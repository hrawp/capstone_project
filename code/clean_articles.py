from pathlib import Path
import re
import pandas as pd
import unicodedata
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

try:
    import ftfy
except ImportError:
    ftfy = None


# =========================
# Configuration
# =========================
INPUT_FILE = Path("iran_war_complete_articles_edits.csv")
OUTPUT_FILE = Path("iran_war_complete_articles_cleaned.csv")
AUDIT_FILE = Path("iran_war_cleaning_audit.csv")
REPORT_FILE = Path("iran_war_cleaning_report.csv")

SAVE_AUDIT = False
SAVE_REPORT = False

TEXT_COLUMNS = [
    "title",
    "author",
    "publisher",
    "publisher_country",
    "publisher_type",
    "link",
    "full_text",
    "source_type",
    "keyword_trigger",
]

INVISIBLE_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM
    "\u2060",  # word joiner
}


# =========================
# File reading
# =========================
def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Read file using encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(path, encoding=enc, engine="python")
                print(f"Read file using encoding: {enc} with python engine")
                return df
            except Exception:
                continue

    raise ValueError("Could not read file with utf-8, utf-8-sig, cp1252, or latin1.")


# =========================
# Generic utilities
# =========================
def count_problem_chars(text) -> int:
    if pd.isna(text):
        return 0

    text = str(text)
    patterns = ["â", "€", "™", "�"]

    count = 0
    for p in patterns:
        count += text.count(p)

    return count


def clean_text(value):
    if pd.isna(value):
        return value

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if ftfy is not None:
        text = ftfy.fix_text(text)

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")

    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")

    text = "".join(c for c in text if c in ("\n", "\t") or unicodedata.category(c)[0] != "C")

    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def count_words(text):
    if pd.isna(text):
        return pd.NA

    text = str(text).strip()
    if not text:
        return pd.NA

    words = re.findall(r"\b\w+(?:[-']\w+)*\b", text, flags=re.UNICODE)
    return len(words)


def dedupe_preserve_order(items):
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


# =========================
# Author cleaning helpers
# =========================
def clean_author_generic(value):
    if pd.isna(value):
        return value

    author = clean_text(value)
    if not author:
        return "Unknown"

    lower = author.lower().strip()

    if "reuters" in lower:
        return "Reuters"
    if "al jazeera staff" in lower:
        return "Al Jazeera Staff"
    if "associated press" in lower or lower == "ap":
        return "Associated Press"
    if lower in {"unknown", "none", "nan", "n/a", "na"}:
        return "Unknown"

    author = re.sub(r"^\s*by\s+", "", author, flags=re.IGNORECASE)
    author = re.sub(r"\s+", " ", author).strip(" ;,")

    return author if author else "Unknown"


def clean_author_bbc(value):
    if pd.isna(value):
        return value

    author = clean_text(value)
    if not author:
        return "Unknown"

    lower = author.lower().strip()
    if lower in {"unknown", "none", "nan", "n/a", "na"}:
        return "Unknown"

    author = re.sub(r"^\s*by\s+", "", author, flags=re.IGNORECASE)

    # Remove UI/time
    author = re.sub(r"\bshare\b", "", author, flags=re.IGNORECASE)
    author = re.sub(r"\bsave\b", "", author, flags=re.IGNORECASE)
    author = re.sub(
        r"\b\d+\s*(?:hour|hours|hr|hrs|minute|minutes|min|mins|day|days)\s+ago\b",
        "",
        author,
        flags=re.IGNORECASE,
    )

    # Remove BBC labels/units
    label_patterns = [
        r"\bBBC\s+Persian\b",
        r"\bBBC\s+Afghan\b",
        r"\bBBC\s+Verify\b",
        r"\bBBC\s+Sport\b",
        r"\bBBC\s+News\b",
    ]
    for pat in label_patterns:
        author = re.sub(pat, "", author, flags=re.IGNORECASE)

    # Remove common role/location/unit phrases
    patterns = [
        r"\bMiddle East bureau chief\b",
        r"\bBusiness reporter\b",
        r"\bPolitical reporter\b",
        r"\bSecurity correspondent\b",
        r"\bDiplomatic correspondent\b",
        r"\bMiddle East correspondent\b",
        r"\bSenior reporter\b",
        r"\bSpecial correspondent\b",
        r"\bEurope digital editor\b",
        r"\bNorth America correspondent\b",
        r"\bIndia correspondent\b",
        r"\bDeputy economics editor\b",
        r"\bEnvironment correspondent\b",
        r"\bWashington correspondent\b",
        r"\breporting from Beirut\b",
        r"\bat the White House\b",
        r"\bWest Bloomfield,\s*Michigan\b",
        r"\bNorthern Iraq\b",
        r"\bYounine,\s*northeastern Lebanon\b",
        r"\bTammun,\s*occupied West Bank\b",
        r"\bRamat Gan,\s*Israel\b",
        r"\bRamat Gan\b",
        r"\bWest Bank\b",
        r"\bWhite House\b",
        r"\bRiyadh\b",
        r"\bSydney\b",
        r"\bBeirut\b",
        r"\bKabul\b",
        r"\band\b$",
    ]
    for pat in patterns:
        author = re.sub(pat, "", author, flags=re.IGNORECASE)

    # Remove trailing fragment locations after commas
    author = re.sub(r",\s*(?:in\s+)?[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+)*$", "", author)

    # Normalize separators
    author = re.sub(r"\s+and\s+", "; ", author, flags=re.IGNORECASE)
    author = re.sub(r"\s*,\s*", "; ", author)
    author = re.sub(r"(?:\s*;\s*){2,}", "; ", author)
    author = re.sub(r"\s+", " ", author).strip(" ;,")

    # Extract names
    name_pattern = (
        r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+"
        r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
        r"(?:\s+(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?|Jr))?\b"
    )
    names = re.findall(name_pattern, author)

    bad_tokens = {
        "Middle",
        "Business",
        "Political",
        "Security",
        "Diplomatic",
        "Senior",
        "Special",
        "Europe",
        "North",
        "India",
        "Deputy",
        "Environment",
        "White",
        "House",
        "West",
        "Bank",
        "Ramat",
        "Gan",
        "Beirut",
        "Sydney",
        "Kabul",
        "Persian",
        "Afghan",
        "Verify",
        "Reporter",
        "Correspondent",
        "Editor",
    }

    cleaned = []
    for name in names:
        parts = name.split()
        if any(part in bad_tokens for part in parts):
            continue
        cleaned.append(name.strip())

    cleaned = dedupe_preserve_order(cleaned)
    cleaned = [n for n in cleaned if len(n.split()) >= 2]

    return "; ".join(cleaned) if cleaned else "Unknown"


def clean_author_nbc(value):
    if pd.isna(value):
        return value

    author = clean_text(value)
    if not author:
        return "Unknown"

    lower = author.lower().strip()

    if "associated press" in lower:
        return "Associated Press"
    if lower in {"unknown", "none", "nan", "n/a", "na"}:
        return "Unknown"

    author = author.replace("–", "-").replace("—", "-")

    junk_patterns = [
        r"https?://\S+",
        r"\bmedia[-.]?cldnry\b",
        r"\bmedia[-.]?\w+\b",
        r"\bnewscms\b",
        r"\bimage upload\b",
        r"\bcom\b",
        r"\bjpg\b",
        r"\bjpeg\b",
        r"\bpng\b",
        r"\bnbc news\b",
    ]
    for pat in junk_patterns:
        author = re.sub(pat, "", author, flags=re.IGNORECASE)

    template_patterns = [
        r"\b[A-Z][a-z]+-[A-Z][a-z]+-Circle-Byline-Template\b",
        r"\b[A-Z][a-z]+-[A-Z][a-z]+-Byline-Jm\b",
        r"\bCircle Byline Template\b",
        r"\bByline Jm\b",
    ]
    for pat in template_patterns:
        author = re.sub(pat, "", author)

    # Normalize hyphenated names
    author = re.sub(r"\b([A-Z][a-z]+)-([A-Z][a-z]+)\b", r"\1 \2", author)
    author = re.sub(r"\b([A-Z][a-z]+)-([A-Z][a-z]+)-([A-Z][a-z]+)\b", r"\1 \2 \3", author)

    # Remove descriptive clauses
    desc_patterns = [
        r"\bIs A\b.*",
        r"\bIs An\b.*",
        r"\bIs The\b.*",
        r"\bReports On\b.*",
        r"\bBased In\b.*",
        r"\bCovering\b.*",
        r"\bChief\b.*",
        r"\bCorrespondent\b.*",
        r"\bReporter\b.*",
        r"\bProducer\b.*",
        r"\bEditor\b.*",
        r"\bDirector\b.*",
        r"\bCongress For Nbc\b.*",
        r"\bThe Economy\b.*",
        r"\bTechnology Policy\b.*",
        r"\bMeet The Press\b.*",
        r"\bInvestigative Unit\b.*",
        r"\bThe Military For\b.*",
        r"\bTrends For Nbc\b.*",
        r"\bCovering The State\b.*",
        r"\bPreviously He Was\b.*",
        r"\bPreviously She Was\b.*",
        r"\bAsia Desk Intern\b.*",
        r"\bWho Has Contributed\b.*",
        r"\bSuch As Cnn\b.*",
        r"\bThe South China\b.*",
        r"\bMorning Post\b.*",
    ]
    for pat in desc_patterns:
        author = re.sub(pat, "", author, flags=re.IGNORECASE)

    author = re.sub(r"\s+and\s+", "; ", author, flags=re.IGNORECASE)
    author = re.sub(r"\s*,\s*", "; ", author)
    author = re.sub(r"(?:\s*;\s*){2,}", "; ", author)
    author = re.sub(r"\s+", " ", author).strip(" ;,")

    name_pattern = (
        r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+"
        r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
        r"(?:\s+(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?|V))?\b"
    )
    names = re.findall(name_pattern, author)

    bad_tokens = {
        "Media",
        "Cldnry",
        "Com",
        "News",
        "Meet",
        "Press",
        "Investigative",
        "Unit",
        "Congress",
        "Technology",
        "Policy",
        "Economy",
        "Reporter",
        "Correspondent",
        "Producer",
        "Editor",
        "Director",
        "Chief",
        "Template",
        "Byline",
        "Jm",
        "Covering",
        "State",
        "Military",
        "Trends",
    }

    cleaned = []
    for name in names:
        parts = name.split()
        if any(part in bad_tokens for part in parts):
            continue
        cleaned.append(name.strip())

    cleaned = dedupe_preserve_order(cleaned)
    cleaned = [n for n in cleaned if len(n.split()) >= 2]

    return "; ".join(cleaned) if cleaned else "Unknown"


def clean_author_aljazeera(value):
    if pd.isna(value):
        return value

    author = clean_text(value)
    if not author:
        return "Unknown"

    lower = author.lower().strip()

    if "al jazeera staff" in lower:
        return "Al Jazeera Staff"
    if lower in {"unknown", "none", "nan", "n/a", "na"}:
        return "Unknown"

    # Remove affiliation tails
    author = re.sub(
        r",\s*American University Of Beirut.*$",
        "",
        author,
        flags=re.IGNORECASE,
    )

    # Split on commas first
    parts = [p.strip() for p in re.split(r"\s*,\s*", author) if p.strip()]

    name_pattern = (
        r"\b[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
        r"(?:\s+[A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+){1,2}\b"
    )

    names = []
    for part in parts:
        matches = re.findall(name_pattern, part)
        for m in matches:
            names.append(m.strip())

    names = dedupe_preserve_order(names)

    # Remove concatenated duplicate combos like "Ted Regencia Zaid"
    cleaned = []
    for n in names:
        words = n.split()
        if len(words) == 3:
            if any(
                n != other and (" ".join(words[:2]) == other or " ".join(words[1:]) == other)
                for other in names
            ):
                continue
        cleaned.append(n)

    cleaned = dedupe_preserve_order(cleaned)
    return "; ".join(cleaned) if cleaned else "Unknown"


def clean_author_fox(value):
    if pd.isna(value):
        return value

    author = clean_text(value)
    if not author:
        return "Unknown"

    lower = author.lower().strip()
    if lower in {"unknown", "none", "nan", "n/a", "na"}:
        return "Unknown"

    author = re.sub(r"\s*,\s*", "; ", author)
    author = re.sub(r"\b([A-Z][a-z]+)-([A-Z][a-z]+)\b", r"\1 \2", author)
    author = re.sub(r"(?:\s*;\s*){2,}", "; ", author)
    author = re.sub(r"\s+", " ", author).strip(" ;,")

    return author if author else "Unknown"


def clean_author_by_publisher(row):
    author = row.get("author")
    publisher = str(row.get("publisher", "")).strip().lower()

    if "bbc" in publisher:
        return clean_author_bbc(author)
    if "nbc" in publisher:
        return clean_author_nbc(author)
    if "al jazeera" in publisher:
        return clean_author_aljazeera(author)
    if "fox" in publisher:
        return clean_author_fox(author)

    return clean_author_generic(author)


# =========================
# Dataset-specific cleaning
# =========================
def update_fox_lengths(df: pd.DataFrame) -> pd.DataFrame:
    if "publisher" not in df.columns:
        return df

    fox_mask = df["publisher"].fillna("").astype(str).str.contains("fox", case=False, na=False)

    if "headline_length" in df.columns:
        df["headline_length"] = pd.to_numeric(df["headline_length"], errors="coerce").astype(
            "Int64"
        )

    if "article_word_count" in df.columns:
        df["article_word_count"] = pd.to_numeric(df["article_word_count"], errors="coerce").astype(
            "Int64"
        )

    if "headline_length" in df.columns and "title" in df.columns:
        needs_headline = fox_mask & (df["headline_length"].isna() | (df["headline_length"] == 0))

        if needs_headline.any():
            headline_values = (
                df.loc[needs_headline, "title"].fillna("").astype(str).str.len().astype("Int64")
            )
            df.loc[needs_headline, "headline_length"] = headline_values

    if "article_word_count" in df.columns and "full_text" in df.columns:
        needs_word_count = fox_mask & (
            df["article_word_count"].isna() | (df["article_word_count"] == 0)
        )

        if needs_word_count.any():
            word_counts = df.loc[needs_word_count, "full_text"].apply(count_words).astype("Int64")
            df.loc[needs_word_count, "article_word_count"] = word_counts

    return df


# =========================
# Short and Half text
# =========================


def truncate_text_by_words(text, max_words):
    if pd.isna(text):
        return pd.NA

    text = str(text).strip()
    if not text:
        return pd.NA

    words = re.findall(r"\b\w+(?:[-']\w+)*\b", text, flags=re.UNICODE)

    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words])


# =========================
# Normalize article URLs
# =========================


def canonicalize_link(url):
    """
    Normalize article URLs so tracking/query variants collapse to one canonical link.
    """
    if pd.isna(url):
        return pd.NA

    url = str(url).strip()
    if not url:
        return pd.NA

    parts = urlsplit(url)

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") if parts.path != "/" else parts.path

    drop_params = {
        "traffic_source",
        "at_medium",
        "at_campaign",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
    }

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    kept_pairs = [(k, v) for k, v in query_pairs if k not in drop_params]
    kept_pairs.sort()
    query = urlencode(kept_pairs)

    return urlunsplit((scheme, netloc, path, query, ""))


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate articles using canonical links first, then a fallback.
    Keeps the best row available.
    """
    df = df.copy()

    if "link" in df.columns:
        df["canonical_link"] = df["link"].apply(canonicalize_link)
    else:
        df["canonical_link"] = pd.NA

    # Prefer rows with a real author and longer text
    if "author" in df.columns:
        df["author_quality"] = (
            df["author"].fillna("Unknown").astype(str).str.strip().ne("Unknown")
        ).astype(int)
    else:
        df["author_quality"] = 0

    if "full_text" in df.columns:
        df["text_length_rank"] = df["full_text"].fillna("").astype(str).str.len()
    else:
        df["text_length_rank"] = 0

    # Best rows first
    df = df.sort_values(by=["author_quality", "text_length_rank"], ascending=[False, False])

    # Primary dedupe by canonical URL
    if "canonical_link" in df.columns:
        df = df.drop_duplicates(subset=["canonical_link"], keep="first")

    # Fallback dedupe for same story if URL missing/odd
    fallback_cols = [c for c in ["title", "publisher", "published_date"] if c in df.columns]
    if fallback_cols:
        df = df.drop_duplicates(subset=fallback_cols, keep="first")

    df = df.drop(columns=["author_quality", "text_length_rank"], errors="ignore")
    return df


# =========================
# Main pipeline
# =========================
def main():
    df = read_csv_safe(INPUT_FILE)
    original_df = df.copy(deep=True)

    # Problem chars before
    if "full_text" in df.columns:
        before_counts = df["full_text"].apply(count_problem_chars)
        total_before = int(before_counts.sum())
        rows_with_issues_before = int((before_counts > 0).sum())
    else:
        total_before = 0
        rows_with_issues_before = 0

    # Clean general text columns
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Create truncated text columns
    if "full_text" in df.columns:
        df["start_text"] = df["full_text"].apply(lambda x: truncate_text_by_words(x, 300))
        df["half_text"] = df["full_text"].apply(lambda x: truncate_text_by_words(x, 600))

    # Publisher-specific author cleaning
    if "author" in df.columns and "publisher" in df.columns:
        df["author"] = df.apply(clean_author_by_publisher, axis=1)
    elif "author" in df.columns:
        df["author"] = df["author"].apply(clean_author_generic)

    # FOX lengths
    df = update_fox_lengths(df)

    # Recalculate article word count from full_text for all rows
    if "full_text" in df.columns:
        df["article_word_count"] = df["full_text"].apply(count_words).astype("Int64")

    # Create truncated text columns
    if "full_text" in df.columns:
        df["start_text"] = df["full_text"].apply(lambda x: truncate_text_by_words(x, 300))
        df["half_text"] = df["full_text"].apply(lambda x: truncate_text_by_words(x, 600))

    # Remove low-content articles
    if "article_word_count" in df.columns:
        before_rows = len(df)
        df = df[df["article_word_count"] > 50].copy()
        after_rows = len(df)
        print(f"Removed {before_rows - after_rows} articles with 50 or fewer words.")

    # Remove duplicate articles
    before_dedup = len(df)
    df = deduplicate_articles(df)
    after_dedup = len(df)
    print(f"Removed {before_dedup - after_dedup} duplicate articles.")

    # Problem chars after
    if "full_text" in df.columns:
        after_counts = df["full_text"].apply(count_problem_chars)
        total_after = int(after_counts.sum())
        rows_with_issues_after = int((after_counts > 0).sum())
    else:
        total_after = 0
        rows_with_issues_after = 0

    # Save cleaned output
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Audit
    audit_df = pd.DataFrame()
    if SAVE_AUDIT:
        audit_rows = []
        audit_columns = list(
            set(TEXT_COLUMNS + ["headline_length", "article_word_count", "author"])
        )

        for col in audit_columns:
            if col in df.columns and col in original_df.columns:
                before_series = original_df[col].astype("string")
                after_series = df[col].astype("string")
                changed_mask = before_series.fillna("<NA>") != after_series.fillna("<NA>")

                for idx in df.index[changed_mask]:
                    audit_rows.append(
                        {
                            "row_index": idx,
                            "Key": df.loc[idx, "Key"] if "Key" in df.columns else None,
                            "column": col,
                            "before": original_df.loc[idx, col],
                            "after": df.loc[idx, col],
                        }
                    )

        audit_df = pd.DataFrame(audit_rows)
        audit_df.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    # Report
    if SAVE_REPORT:
        fox_rows = 0
        fox_headlines_filled = 0
        fox_words_filled = 0

        if "publisher" in df.columns:
            fox_mask = (
                df["publisher"].fillna("").astype(str).str.contains("fox", case=False, na=False)
            )
            fox_rows = int(fox_mask.sum())

            if "headline_length" in df.columns and "headline_length" in original_df.columns:
                original_headline = pd.to_numeric(original_df["headline_length"], errors="coerce")
                new_headline = pd.to_numeric(df["headline_length"], errors="coerce")
                fox_headlines_filled = int(
                    (
                        fox_mask
                        & (original_headline.isna() | (original_headline == 0))
                        & new_headline.notna()
                        & (new_headline > 0)
                    ).sum()
                )

            if "article_word_count" in df.columns and "article_word_count" in original_df.columns:
                original_words = pd.to_numeric(original_df["article_word_count"], errors="coerce")
                new_words = pd.to_numeric(df["article_word_count"], errors="coerce")
                fox_words_filled = int(
                    (
                        fox_mask
                        & (original_words.isna() | (original_words == 0))
                        & new_words.notna()
                        & (new_words > 0)
                    ).sum()
                )

        author_changed = 0
        if "author" in df.columns and "author" in original_df.columns:
            author_changed = int(
                (
                    original_df["author"].fillna("").astype(str)
                    != df["author"].fillna("").astype(str)
                ).sum()
            )

        changed_cells = len(audit_df) if SAVE_AUDIT else 0

        report_df = pd.DataFrame(
            {
                "metric": [
                    "total_problem_chars_before",
                    "total_problem_chars_after",
                    "rows_with_issues_before",
                    "rows_with_issues_after",
                    "fox_rows",
                    "fox_headline_length_filled",
                    "fox_article_word_count_filled",
                    "author_rows_changed",
                    "changed_cells",
                ],
                "value": [
                    total_before,
                    total_after,
                    rows_with_issues_before,
                    rows_with_issues_after,
                    fox_rows,
                    fox_headlines_filled,
                    fox_words_filled,
                    author_changed,
                    changed_cells,
                ],
            }
        )
        report_df.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned file: {OUTPUT_FILE}")
    if SAVE_AUDIT:
        print(f"Saved audit file: {AUDIT_FILE}")
    if SAVE_REPORT:
        print(f"Saved report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()
