import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from ftfy import fix_text
from transformers import pipeline

INPUT_FILE = "iran_war_complete_articles_cleaned.csv"
OUTPUT_FILE = "iran_war_complete_articles_sentiment.csv"
TEXT_COLUMN = "full_text"
ID_COLUMN = "link"

MODEL_NAME = "siebert/sentiment-roberta-large-english"
BATCH_SIZE = 32

TEXT_SENTIMENT_COLUMNS = ["title", "full_text", "start_text", "half_text"]

KEYWORD_GROUPS = {
    "iran": [r"\biran\b"],
    "iranian": [r"\biranian\b"],
    "israel_israeli": [r"\bisrael\b", r"\bisraeli\b"],
    "hormuz": [r"\bhormuz\b"],
    "supreme_leader": [r"\bsupreme leader\b"],
    "tehran": [r"\btehran\b"],
    "trump": [r"\btrump\b"],
    "us_united_states": [r"\bu\.s\.?\b", r"\bunited states\b", r"\bUS\b"],
}

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = fix_text(text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def split_into_sentences(text):
    text = normalize_text(text)
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

    text = normalize_text(str(row.get(TEXT_COLUMN, "") or ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_keyword_columns(df):
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

        df[count_col] = df[count_col].astype("object")
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df[sentiment_col] = df[sentiment_col].astype("object")


def ensure_text_sentiment_columns(df):
    for text_col in TEXT_SENTIMENT_COLUMNS:
        score_col = f"{text_col}_score"
        sentiment_col = f"{text_col}_sentiment"

        if score_col not in df.columns:
            df[score_col] = pd.NA
        if sentiment_col not in df.columns:
            df[sentiment_col] = pd.NA

        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df[sentiment_col] = df[sentiment_col].astype("object")

    if "title_article_diff" not in df.columns:
        df["title_article_diff"] = pd.NA
    if "title_article_diff_label" not in df.columns:
        df["title_article_diff_label"] = pd.NA

    df["title_article_diff"] = pd.to_numeric(df["title_article_diff"], errors="coerce")
    df["title_article_diff_label"] = df["title_article_diff_label"].astype("object")


def coerce_label_columns_to_object(df):
    for col in df.columns:
        if col.endswith("_sentiment") or col == "title_article_diff_label":
            df[col] = df[col].astype("object")
    return df


def load_base_dataframe():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_input = pd.read_csv(input_path, encoding="utf-8")

    if TEXT_COLUMN not in df_input.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found in input. "
            f"Available columns: {list(df_input.columns)}"
        )

    for col in TEXT_SENTIMENT_COLUMNS:
        if col in df_input.columns:
            df_input[col] = df_input[col].fillna("").apply(normalize_text)

    df_input["_row_id"] = df_input.apply(make_row_id, axis=1)

    if output_path.exists():
        print(f"Existing output found: {output_path}")
        df_existing = pd.read_csv(output_path, encoding="utf-8-sig")

        if "_row_id" not in df_existing.columns:
            raise ValueError(
                "Existing output is missing '_row_id'. Add it once or rebuild the output file."
            )

        keep_cols = ["_row_id"]

        for key in KEYWORD_GROUPS.keys():
            keep_cols.extend(
                [
                    f"{key}_count",
                    f"{key}_score",
                    f"{key}_sentiment",
                ]
            )

        for text_col in TEXT_SENTIMENT_COLUMNS:
            keep_cols.extend(
                [
                    f"{text_col}_score",
                    f"{text_col}_sentiment",
                ]
            )

        keep_cols.extend(
            [
                "title_article_diff",
                "title_article_diff_label",
            ]
        )

        keep_cols = [c for c in keep_cols if c in df_existing.columns]
        df = df_input.merge(df_existing[keep_cols], on="_row_id", how="left")
    else:
        df = df_input.copy()

    ensure_keyword_columns(df)
    ensure_text_sentiment_columns(df)
    df = coerce_label_columns_to_object(df)
    return df


def score_sentences(sentences, sentiment_model):
    if not sentences:
        return []

    cleaned_sentences = []
    for s in sentences:
        s_clean = normalize_text(s)
        if s_clean:
            cleaned_sentences.append(s_clean)

    if not cleaned_sentences:
        return []

    outputs = sentiment_model(
        cleaned_sentences,
        truncation=True,
        batch_size=BATCH_SIZE,
    )

    return [label_to_signed_score(o["label"], o["score"]) for o in outputs]


def is_keyword_done(df, key):
    return df[f"{key}_count"].notna() & df[f"{key}_score"].notna() & df[f"{key}_sentiment"].notna()


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

    patterns = compiled_patterns[key]
    matched_sentences_by_index = {}
    flat_sentences = []
    flat_sentence_row_index = []

    for idx, text in df_todo[TEXT_COLUMN].fillna("").items():
        matches = []
        for sentence in split_into_sentences(text):
            if any(p.search(sentence) for p in patterns):
                matches.append(sentence)

        matched_sentences_by_index[idx] = matches

        for sentence in matches:
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
            df.at[idx, score_col] = 0.0
            df.at[idx, sentiment_col] = "no matches"
        else:
            avg_score = round(sum(scores) / len(scores), 4)
            df.at[idx, count_col] = 1
            df.at[idx, score_col] = avg_score
            df.at[idx, sentiment_col] = classify_sentiment(avg_score)

    return df


def is_text_sentiment_done(df, text_col):
    return df[f"{text_col}_score"].notna() & df[f"{text_col}_sentiment"].notna()


def get_rows_missing_any_text_sentiment(df):
    needed_mask = pd.Series(False, index=df.index)

    for text_col in TEXT_SENTIMENT_COLUMNS:
        if text_col not in df.columns:
            continue

        score_col = f"{text_col}_score"
        sentiment_col = f"{text_col}_sentiment"

        missing_mask = df[score_col].isna() | df[sentiment_col].isna()
        needed_mask = needed_mask | missing_mask

    return df.loc[needed_mask].copy()


def build_sentence_score_cache(df, sentiment_model):
    unique_sentences = set()

    for text_col in TEXT_SENTIMENT_COLUMNS:
        if text_col not in df.columns:
            continue

        score_col = f"{text_col}_score"
        sentiment_col = f"{text_col}_sentiment"

        missing_mask = df[score_col].isna() | df[sentiment_col].isna()
        df_missing = df.loc[missing_mask]

        for text in df_missing[text_col].fillna(""):
            for sentence in split_into_sentences(text):
                unique_sentences.add(sentence)

    unique_sentences = sorted(unique_sentences)
    print(f"Unique sentences to score for pending text sentiment: {len(unique_sentences)}")

    scores = score_sentences(unique_sentences, sentiment_model)
    return dict(zip(unique_sentences, scores))


def process_text_column_sentiment(df, text_col, sentence_score_cache):
    score_col = f"{text_col}_score"
    sentiment_col = f"{text_col}_sentiment"

    if text_col not in df.columns:
        print(f"\nSkipping {text_col}: column not found.")
        return df

    todo_mask = ~is_text_sentiment_done(df, text_col)
    df_todo = df.loc[todo_mask].copy()

    print(f"\nProcessing text sentiment for: {text_col}")
    print(f"Rows needing this text sentiment: {len(df_todo)}")

    if len(df_todo) == 0:
        print("Nothing to do for this text column.")
        return df

    for idx, text in df_todo[text_col].fillna("").items():
        sentences = split_into_sentences(text)

        if not sentences:
            df.at[idx, score_col] = pd.NA
            df.at[idx, sentiment_col] = "no matches"
            continue

        scores = [sentence_score_cache[s] for s in sentences if s in sentence_score_cache]

        if not scores:
            df.at[idx, score_col] = pd.NA
            df.at[idx, sentiment_col] = "no matches"
            continue

        avg_score = round(sum(scores) / len(scores), 4)
        df.at[idx, score_col] = avg_score
        df.at[idx, sentiment_col] = classify_sentiment(avg_score)

    return df


def compute_title_article_diff(df):
    if "title_score" not in df.columns or "full_text_score" not in df.columns:
        print("Skipping title-article diff: required columns missing.")
        return df

    both_present = df["title_score"].notna() & df["full_text_score"].notna()

    df.loc[both_present, "title_article_diff"] = (
        df.loc[both_present, "title_score"] - df.loc[both_present, "full_text_score"]
    ).round(4)

    df.loc[~both_present, "title_article_diff"] = pd.NA
    return df


def classify_title_article_diff(value):
    if pd.isna(value):
        return "no data"
    if value <= -0.30:
        return "title much more negative"
    elif value <= -0.10:
        return "title slightly more negative"
    elif value < 0.10:
        return "similar tone"
    elif value < 0.30:
        return "title slightly more positive"
    else:
        return "title much more positive"


def add_title_article_diff_label(df):
    if "title_article_diff" not in df.columns:
        print("Skipping diff label: diff column missing.")
        return df

    df["title_article_diff_label"] = df["title_article_diff"].apply(classify_title_article_diff)
    df["title_article_diff_label"] = df["title_article_diff_label"].astype("object")
    return df


def save_progress(df):
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


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
        print(f"Saving progress after keyword {key} ...")
        save_progress(df)

    print("\nFinding rows still missing any text sentiment...")
    pending_text_df = get_rows_missing_any_text_sentiment(df)
    print(f"Rows still needing text sentiment work: {len(pending_text_df)}")

    if len(pending_text_df) > 0:
        print("\nBuilding shared sentence sentiment cache...")
        sentence_score_cache = build_sentence_score_cache(df, sentiment_model)

        for text_col in TEXT_SENTIMENT_COLUMNS:
            df = process_text_column_sentiment(df, text_col, sentence_score_cache)
            print(f"Saving progress after text column {text_col} ...")
            save_progress(df)
    else:
        print("No text sentiment work needed.")

    print("\nComputing title vs article framing columns...")
    df = compute_title_article_diff(df)
    df = add_title_article_diff_label(df)
    save_progress(df)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
