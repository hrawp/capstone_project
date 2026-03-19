from pathlib import Path
import pandas as pd
import unicodedata

try:
    import ftfy
except ImportError:
    ftfy = None


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

# Known invisible/problematic characters to remove
INVISIBLE_CHARS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # BOM
    "\u2060",  # word joiner
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Try a few encodings so messy CSVs can still be loaded."""
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


def count_problem_chars(text) -> int:
    """Count common mojibake/problem markers for reporting."""
    if pd.isna(text):
        return 0

    text = str(text)
    patterns = ["â", "€", "™", "�"]

    count = 0
    for p in patterns:
        count += text.count(p)

    return count


def clean_text(value):
    """Fix mojibake and remove invisible/control junk without changing wording."""
    if pd.isna(value):
        return value

    text = str(value)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix broken encoding like â€™, â€œ, â€“
    if ftfy is not None:
        text = ftfy.fix_text(text)

    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Convert non-breaking spaces to regular spaces
    text = text.replace("\u00a0", " ")

    # Remove known invisible characters
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")

    # Remove remaining control chars except newline/tab
    text = "".join(c for c in text if c in ("\n", "\t") or unicodedata.category(c)[0] != "C")

    # Trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def main():
    input_file = Path("iran_war_complete_articles_edits.csv")
    output_file = Path("iran_war_complete_articles_textfixed.csv")
    audit_file = Path("iran_war_cleaning_audit.csv")
    report_file = Path("text_cleaning_report.csv")

    df = read_csv_safe(input_file)
    original_df = df.copy(deep=True)

    # Count issues before cleaning
    if "full_text" in df.columns:
        before_counts = df["full_text"].apply(count_problem_chars)
        total_before = int(before_counts.sum())
        rows_with_issues_before = int((before_counts > 0).sum())
    else:
        total_before = 0
        rows_with_issues_before = 0

    # Clean text columns
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Count issues after cleaning
    if "full_text" in df.columns:
        after_counts = df["full_text"].apply(count_problem_chars)
        total_after = int(after_counts.sum())
        rows_with_issues_after = int((after_counts > 0).sum())
    else:
        total_after = 0
        rows_with_issues_after = 0

    # Save cleaned dataset
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # Build audit file of changed cells
    audit_rows = []
    for col in TEXT_COLUMNS:
        if col in df.columns:
            before_series = original_df[col].fillna("").astype(str)
            after_series = df[col].fillna("").astype(str)
            changed_mask = before_series != after_series

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
    audit_df.to_csv(audit_file, index=False, encoding="utf-8-sig")

    # Save report file
    report_df = pd.DataFrame(
        {
            "metric": [
                "total_problem_chars_before",
                "total_problem_chars_after",
                "rows_with_issues_before",
                "rows_with_issues_after",
                "changed_cells",
            ],
            "value": [
                total_before,
                total_after,
                rows_with_issues_before,
                rows_with_issues_after,
                len(audit_df),
            ],
        }
    )
    report_df.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned file: {output_file}")
    print(f"Saved audit file: {audit_file}")
    print(f"Saved report file: {report_file}")
    print(f"Total problem characters BEFORE: {total_before}")
    print(f"Total problem characters AFTER: {total_after}")
    print(f"Rows affected BEFORE: {rows_with_issues_before}")
    print(f"Rows affected AFTER: {rows_with_issues_after}")
    print(f"Changed cells: {len(audit_df)}")


if __name__ == "__main__":
    main()
