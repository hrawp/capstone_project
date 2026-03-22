import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import pipeline

INPUT_FILE = "iran_war_complete_articles_cleaned.csv"
OUTPUT_FILE = "iran_war_complete_articles_sentiment.csv"
TEXT_COLUMN = "full_text"
ID_COLUMN = "link"

MODEL_NAME = "siebert/sentiment-roberta-large-english"
BATCH_SIZE = 32

KEYWORD_GROUPS = {
    "iran": [r"\biran\b"],
    "iranian": [r"\biranian\b"],
    "israel_israeli": [r"\bisrael\b", r"\bisraeli\b"],
    "hormuz": [r"\bhormuz\b"],
    "supreme_leader": [r"\bsupreme leader\b"],
    "tehran": [r"\btehran\b"],
    "trump": [r"\btrump\b"],
    "us_united_states": [r"\bu\.s\.\b", r"\bunited states\b", r"\bus\b"],
}

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text):
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def build_compiled_patterns():
    compiled = {}
    for key, patterns in KEYWORD_GROUPS.items():
        compiled[key] = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    return compiled


def classify_sentiment(score):
    if score is None or pd.isna(score):
        return "no matches"

    if -1.00 <= score <= -0.70:
        return "very negative"
    elif -0.69 <= score <= -0.40:
        return "negative"
    elif -0.39 <= score <= -0.15:
        return "somewhat negative"
    elif -0.14 <= score <= 0.14:
        return "neutral"
    elif 0.15 <= score <= 0.39:
        return "somewhat positive"
    elif 0.40 <= score <= 0.69:
        return "positive"
    elif 0.70 <= score <= 1.00:
        return "very positive"

    return "neutral"


def label_to_signed_score(label, score):
    label = str(label).upper().strip()
    if label == "POSITIVE":
        return float(score)
    if label == "NEGATIVE":
        return -float(score)
    return 0.0


def make_row_id(row):
    if ID_COLUMN in row and pd.notna(row[ID_COLUMN]) and str(row[ID_COLUMN]).strip():
        return str(row[ID_COLUMN]).strip()

    text = str(row.get(TEXT_COLUMN, "") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_matches_for_article(text, compiled_patterns):
    """
    Single-pass extraction:
    split article once, then test each sentence against all keyword groups.
    """
    result = {key: [] for key in compiled_patterns.keys()}
    sentences = split_into_sentences(text)

    for sentence in sentences:
        for key, pattern_list in compiled_patterns.items():
            if any(p.search(sentence) for p in pattern_list):
                result[key].append(sentence)

    return result


def ensure_columns(df):
    for key in KEYWORD_GROUPS.keys():
        count_col = f"{key}_count"
        score_col = f"{key}_score"
        sentiment_col = f"{key}_sentiment"

        if count_col not in df.columns:
            df[count_col] = pd.NA
        if score_col not in df.columns:
            df[score_col] = pd.NA
        if sentiment_col not in df.columns:
            df[sentiment_col] = pd.NA


def load_base_dataframe():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_input = pd.read_csv(input_path)

    if TEXT_COLUMN not in df_input.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found in input. "
            f"Available columns: {list(df_input.columns)}"
        )

    df_input["_row_id"] = df_input.apply(make_row_id, axis=1)

    if output_path.exists():
        print(f"Existing output found: {output_path}")
        df_existing = pd.read_csv(output_path)

        if "_row_id" not in df_existing.columns:
            raise ValueError(
                "Existing output is missing '_row_id'. "
                "Delete or rename the old output once, then rerun."
            )

        # Keep latest cleaned input as base, then bring over existing sentiment cols
        sentiment_cols = [col for col in df_existing.columns if col not in df_input.columns]
        overlapping_cols = [
            col
            for col in df_existing.columns
            if col in df_input.columns and col not in df_input.columns[:]
        ]

        merge_cols = ["_row_id"] + [
            col
            for col in df_existing.columns
            if col.startswith(tuple(KEYWORD_GROUPS.keys())) or col == "_row_id"
        ]

        # safer explicit merge: keep input data current, append prior sentiment results
        sentiment_merge_cols = ["_row_id"]
        for key in KEYWORD_GROUPS.keys():
            sentiment_merge_cols.extend(
                [
                    f"{key}_count",
                    f"{key}_score",
                    f"{key}_sentiment",
                ]
            )

        available_merge_cols = [c for c in sentiment_merge_cols if c in df_existing.columns]
        df = df_input.merge(
            df_existing[available_merge_cols], on="_row_id", how="left", suffixes=("", "_old")
        )
    else:
        df = df_input.copy()

    ensure_columns(df)
    return df


def is_keyword_done(df, key):
    count_col = f"{key}_count"
    score_col = f"{key}_score"
    sentiment_col = f"{key}_sentiment"

    return df[count_col].notna() & df[score_col].notna() & df[sentiment_col].notna()


def score_sentences(sentences, sentiment_model):
    if not sentences:
        return []

    outputs = sentiment_model(
        sentences,
        truncation=True,
        batch_size=BATCH_SIZE,
    )

    return [label_to_signed_score(o["label"], o["score"]) for o in outputs]


def process_keyword(df, key, compiled_patterns, sentiment_model):
    count_col = f"{key}_count"
    score_col = f"{key}_score"
    sentiment_col = f"{key}_sentiment"

    todo_mask = ~is_keyword_done(df, key)
    df_todo = df.loc[todo_mask].copy()

    print(f"\nProcessing keyword group: {key}")
    print(f"Rows needing this keyword: {len(df_todo)}")

    if len(df_todo) == 0:
        print("Nothing to do for this keyword.")
        return df

    matched_sentences_by_index = {}
    flat_sentences = []
    flat_sentence_row_index = []

    patterns = compiled_patterns[key]

    for idx, text in df_todo[TEXT_COLUMN].fillna("").items():
        article_matches = []
        for sentence in split_into_sentences(text):
            if any(p.search(sentence) for p in patterns):
                article_matches.append(sentence)

        matched_sentences_by_index[idx] = article_matches

        for sentence in article_matches:
            flat_sentences.append(sentence)
            flat_sentence_row_index.append(idx)

    print(f"Matched sentences for {key}: {len(flat_sentences)}")

    sentence_scores = score_sentences(flat_sentences, sentiment_model)

    score_buckets = {idx: [] for idx in df_todo.index}
    for idx, score in zip(flat_sentence_row_index, sentence_scores):
        score_buckets[idx].append(score)

    for idx in df_todo.index:
        count = len(matched_sentences_by_index[idx])
        scores = score_buckets[idx]

        if count == 0:
            df.at[idx, count_col] = 0
            df.at[idx, score_col] = pd.NA
            df.at[idx, sentiment_col] = "no matches"
        else:
            avg_score = round(sum(scores) / len(scores), 4)
            df.at[idx, count_col] = count
            df.at[idx, score_col] = avg_score
            df.at[idx, sentiment_col] = classify_sentiment(avg_score)

    return df


def main():
    print("Loading data...")
    df = load_base_dataframe()

    compiled_patterns = build_compiled_patterns()

    print("Loading sentiment model...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
    )

    for key in KEYWORD_GROUPS.keys():
        df = process_keyword(df, key, compiled_patterns, sentiment_model)

        print(f"Saving progress after {key} ...")
        df.to_csv(OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
