import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article


# =========================================================
# CONFIGURATION
# =========================================================

# Output CSV file
csv_file = "iran_war_complete_articles.csv"

# Keywords used for RSS filtering, archive link filtering,
# and archive article text filtering
keywords = [
    "iran",
    "middle east",
    "israel",
    "airstrike",
    "airstrikes",
    "missile",
    "missiles",
    "strike",
    "strikes",
    "hormuz",
]

# RSS feeds
rss_feeds = {
    "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "NBC News": "https://feeds.nbcnews.com/nbcnews/public/world",
    "Fox News": "https://moxie.foxnews.com/google-publisher/world.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
}

# Archive / section pages
archive_urls = {
    "BBC": "https://www.bbc.com/news/world",
    "NBC News": "https://www.nbcnews.com/world",
    "Fox News": "https://www.foxnews.com/world",
    "Al Jazeera": "https://www.aljazeera.com/news/middleeast",
}

# Publisher metadata
publisher_country_map = {
    "BBC": "UK",
    "NBC News": "US",
    "Fox News": "US",
    "Al Jazeera": "Qatar",
}

publisher_type_map = {
    "BBC": "Public Broadcaster",
    "NBC News": "Broadcast Network",
    "Fox News": "Cable News",
    "Al Jazeera": "International Network",
}

# Browser-like headers can help with some requests
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    )
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def get_collection_day() -> str:
    """Return today's date as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def normalize_to_day(date_value) -> str:
    """
    Convert many date formats to YYYY-MM-DD.
    Returns 'Unknown' if parsing fails.
    """
    if not date_value:
        return "Unknown"

    try:
        dt = parsedate_to_datetime(date_value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        return pd.to_datetime(date_value).strftime("%Y-%m-%d")
    except Exception:
        return "Unknown"


def extract_date_from_url(url: str) -> str:
    """
    Fallback for URLs containing dates like /2026/3/14/
    """
    match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", url)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return "Unknown"


def is_valid_url(url: str) -> bool:
    """
    Skip obvious non-article URLs.
    """
    if not isinstance(url, str):
        return False

    lower_url = url.lower()

    if lower_url.startswith("javascript:"):
        return False
    if lower_url.startswith("#"):
        return False
    if "/video/" in lower_url:
        return False
    if "/live/" in lower_url:
        return False

    return lower_url.startswith("http://") or lower_url.startswith("https://")


def get_first_keyword_match(text: str, keyword_list: list[str]) -> str | None:
    """
    Return the first keyword found in text, else None.
    """
    if not text:
        return None

    lower_text = text.lower()
    for kw in keyword_list:
        if kw in lower_text:
            return kw
    return None


def page_date_from_html(url: str, publisher: str) -> str:
    """
    Try to extract the publication day from article HTML.
    Uses common meta tags plus a BBC-specific <time> fallback.
    """
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Common metadata locations for publication time
        candidates = [
            ("meta", {"property": "article:published_time"}, "content"),
            ("meta", {"name": "article:published_time"}, "content"),
            ("meta", {"name": "ptime"}, "content"),
            ("meta", {"property": "og:updated_time"}, "content"),
            ("meta", {"name": "pubdate"}, "content"),
        ]

        for tag_name, attrs, attr_name in candidates:
            tag = soup.find(tag_name, attrs)
            if tag and tag.get(attr_name):
                return normalize_to_day(tag.get(attr_name))

        # BBC often uses a time tag
        if publisher == "BBC":
            time_tag = soup.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                return normalize_to_day(time_tag["datetime"])

    except Exception:
        pass

    # Final fallback: try to get date from the URL itself
    return extract_date_from_url(url)


def extract_bbc_author(soup):
    """
    Extract BBC author/byline using multiple fallback patterns.
    """
    # Meta tags first
    meta_candidates = [
        ("meta", {"name": "byl"}, "content"),
        ("meta", {"property": "article:author"}, "content"),
        ("meta", {"name": "article:author"}, "content"),
    ]

    for tag_name, attrs, attr_name in meta_candidates:
        tag = soup.find(tag_name, attrs)
        if tag and tag.get(attr_name):
            return tag.get(attr_name).replace("By ", "").strip()

    # Broader visible byline search
    byline_selectors = [
        '[data-testid*="byline"]',
        '[data-component*="byline"]',
        '[class*="Contributor"]',
        '[class*="byline"]',
    ]

    for selector in byline_selectors:
        tags = soup.select(selector)
        for tag in tags:
            text = tag.get_text(" ", strip=True)
            if text and len(text.split()) <= 12:
                return text.replace("By ", "").strip()

    return "Unknown"


def extract_bbc_full_text(soup):
    """
    Extract BBC article text using broader paragraph fallbacks.
    """
    selectors = [
        '[data-component="text-block"] p',
        '[data-component="text-block"]',
        'article [data-component="text-block"]',
        'main article p',
        'article p',
        'main p',
    ]

    best_paragraphs = []

    for selector in selectors:
        tags = soup.select(selector)
        paragraphs = []

        for tag in tags:
            text = tag.get_text(" ", strip=True)

            # Skip very short junk
            if text and len(text.split()) >= 5:
                paragraphs.append(text)

        # Keep the selector that yields the most text
        if len(" ".join(paragraphs).split()) > len(" ".join(best_paragraphs).split()):
            best_paragraphs = paragraphs

    # Remove duplicates while preserving order
    seen = set()
    cleaned = []
    for p in best_paragraphs:
        if p not in seen:
            cleaned.append(p)
            seen.add(p)

    return "\n\n".join(cleaned)


def extract_article(url: str, publisher: str, rss_date=None) -> dict:
    if not is_valid_url(url):
        return {
            "success": False,
            "title": "",
            "author": "",
            "published_date": "Unknown",
            "full_text": "",
        }

    try:
        print(f"Downloading article from {publisher}: {url}")

        article = Article(url)
        article.download()
        article.parse()

        title = article.title or ""
        full_text = article.text or ""
        author = ", ".join(article.authors) if article.authors else "Unknown"

        # Fetch page HTML once for fallbacks
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        if publisher == "BBC":
            # Better author fallback
            if author == "Unknown":
                author = extract_bbc_author(soup)

            # Better full-text fallback
            bbc_full_text = extract_bbc_full_text(soup)

            # Always use whichever version is longer
            if len(bbc_full_text.split()) > len(full_text.split()):
                full_text = bbc_full_text

        published_date = normalize_to_day(rss_date)
        if published_date == "Unknown":
            published_date = page_date_from_html(url, publisher)

        return {
            "success": True,
            "title": title,
            "author": author,
            "published_date": published_date,
            "full_text": full_text,
        }

    except Exception as e:
        print(f"Failed to extract {url}: {e}")
        return {
            "success": False,
            "title": "",
            "author": "",
            "published_date": normalize_to_day(rss_date),
            "full_text": "",
        }


def build_record(
    key: int,
    publisher: str,
    link: str,
    source_type: str,
    keyword_trigger: str,
    collection_date: str,
    title: str = "",
    author: str = "",
    published_date: str = "Unknown",
    full_text: str = "",
    failed: bool = False,
) -> dict:
    """
    Build one output row with a consistent schema.
    """
    publisher_country = publisher_country_map.get(publisher, "Unknown")
    publisher_type = publisher_type_map.get(publisher, "Unknown")

    headline_length = len(title) if title else 0
    article_word_count = len(full_text.split()) if full_text else 0

    return {
        "Key": key,
        "title": title if not failed else "Failed download",
        "headline_length": headline_length if not failed else 0,
        "article_word_count": article_word_count if not failed else 0,
        "author": author if not failed else "Unknown",
        "publisher": publisher,
        "publisher_country": publisher_country,
        "publisher_type": publisher_type,
        "published_date": published_date,
        "collection_date": collection_date,
        "link": link,
        "full_text": full_text if not failed else "",
        "source_type": source_type,
        "keyword_trigger": keyword_trigger,
    }


def load_existing_data():
    """
    Load existing CSV data if present.

    Returns:
    - existing dataframe
    - set of existing links
    - next success key
    - next failed key
    """
    try:
        df_existing = pd.read_csv(csv_file)

        if "Key" not in df_existing.columns:
            df_existing["Key"] = range(2000, 2000 + len(df_existing))

        existing_links = set(df_existing["link"].dropna().astype(str))

        success_keys = df_existing.loc[df_existing["Key"] < 7000, "Key"]
        failed_keys = df_existing.loc[df_existing["Key"] >= 7000, "Key"]

        next_success_key = (int(success_keys.max()) + 1) if not success_keys.empty else 2000
        next_failed_key = (int(failed_keys.max()) + 1) if not failed_keys.empty else 7000

        return df_existing, existing_links, next_success_key, next_failed_key

    except FileNotFoundError:
        return pd.DataFrame(), set(), 2000, 7000


def save_articles(records: list[dict]) -> None:
    """
    Save new records to CSV and remove duplicate links.
    """
    df_existing, _, _, _ = load_existing_data()
    df_new = pd.DataFrame(records)

    if df_existing.empty:
        df_combined = df_new
    else:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Keep first occurrence of each URL
    df_combined.drop_duplicates(subset="link", inplace=True)
    df_combined.to_csv(csv_file, index=False)

    print(f"{datetime.now()} - CSV updated. Total rows: {len(df_combined)}")


# =========================================================
# RSS WORKFLOW
# =========================================================


def process_rss(existing_links: set, next_success_key: int, next_failed_key: int):
    """
    Process RSS feeds.

    Rule:
    - Check RSS title + summary for keywords
    - If matched, download the article
    - Save the article
    """
    records = []
    collection_date = get_collection_day()

    stats = {
        "articles_added": 0,
        "failed_downloads": 0,
        "duplicates_skipped": 0,
        "publisher_counts": {publisher: 0 for publisher in rss_feeds.keys()},
    }

    for publisher, feed_url in rss_feeds.items():
        print(f"Fetching RSS from {publisher}")
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            link = getattr(entry, "link", "")

            if not is_valid_url(link):
                continue

            if link in existing_links:
                stats["duplicates_skipped"] += 1
                continue

            rss_text = (getattr(entry, "title", "") + " " + entry.get("summary", "")).lower()

            keyword_trigger = get_first_keyword_match(rss_text, keywords)

            if not keyword_trigger:
                continue

            rss_date = getattr(entry, "published", None) or getattr(entry, "updated", None)
            result = extract_article(link, publisher, rss_date=rss_date)

            if result["success"]:
                record = build_record(
                    key=next_success_key,
                    publisher=publisher,
                    link=link,
                    source_type="RSS",
                    keyword_trigger=keyword_trigger,
                    collection_date=collection_date,
                    title=result["title"],
                    author=result["author"],
                    published_date=result["published_date"],
                    full_text=result["full_text"],
                    failed=False,
                )
                records.append(record)
                existing_links.add(link)
                next_success_key += 1

                stats["articles_added"] += 1
                stats["publisher_counts"][publisher] += 1

            else:
                record = build_record(
                    key=next_failed_key,
                    publisher=publisher,
                    link=link,
                    source_type="FAILED",
                    keyword_trigger=keyword_trigger,
                    collection_date=collection_date,
                    title="",
                    author="",
                    published_date=result["published_date"],
                    full_text="",
                    failed=True,
                )
                records.append(record)
                existing_links.add(link)
                next_failed_key += 1

                stats["failed_downloads"] += 1

    return records, next_success_key, next_failed_key, stats


# =========================================================
# ARCHIVE WORKFLOW
# =========================================================


def process_archive(existing_links: set, next_success_key: int, next_failed_key: int):
    """
    Process archive/section pages.

    Rule:
    - Check only first 50 links per publisher
    - Filter first by link text or URL
    - Download matching links
    - Require a keyword in the article text before saving
    """
    records = []
    collection_date = get_collection_day()

    stats = {
        "articles_added": 0,
        "failed_downloads": 0,
        "duplicates_skipped": 0,
        "publisher_counts": {publisher: 0 for publisher in archive_urls.keys()},
    }

    for publisher, archive_url in archive_urls.items():
        print(f"Scanning archive page for {publisher}")

        try:
            resp = requests.get(archive_url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Only inspect the first 600 links
            archive_links = soup.find_all("a", href=True)[:600]

            for a_tag in archive_links:
                href = a_tag.get("href", "")
                link_text = a_tag.get_text(" ", strip=True)
                full_url = urljoin(archive_url, href)

                if not is_valid_url(full_url):
                    continue

                if full_url in existing_links:
                    stats["duplicates_skipped"] += 1
                    continue

                # First filter: link text or URL must contain a keyword
                keyword_trigger = get_first_keyword_match(
                    f"{link_text} {full_url}",
                    keywords,
                )

                if not keyword_trigger:
                    continue

                result = extract_article(full_url, publisher, rss_date=None)

                if result["success"]:
                    # Second filter: article text must contain a keyword
                    article_text = f"{result['title']} {result['full_text']}".lower()
                    article_keyword = get_first_keyword_match(article_text, keywords)

                    if not article_keyword:
                        continue

                    record = build_record(
                        key=next_success_key,
                        publisher=publisher,
                        link=full_url,
                        source_type="ARCHIVE",
                        keyword_trigger=keyword_trigger,
                        collection_date=collection_date,
                        title=result["title"],
                        author=result["author"],
                        published_date=result["published_date"],
                        full_text=result["full_text"],
                        failed=False,
                    )
                    records.append(record)
                    existing_links.add(full_url)
                    next_success_key += 1

                    stats["articles_added"] += 1
                    stats["publisher_counts"][publisher] += 1

                else:
                    record = build_record(
                        key=next_failed_key,
                        publisher=publisher,
                        link=full_url,
                        source_type="FAILED",
                        keyword_trigger=keyword_trigger,
                        collection_date=collection_date,
                        title="",
                        author="",
                        published_date=result["published_date"],
                        full_text="",
                        failed=True,
                    )
                    records.append(record)
                    existing_links.add(full_url)
                    next_failed_key += 1

                    stats["failed_downloads"] += 1

        except Exception as e:
            print(f"Failed to fetch archive for {publisher}: {e}")

    return records, next_success_key, next_failed_key, stats


# =========================================================
# MAIN WORKFLOW
# =========================================================


def hybrid_fetch():
    """
    Run archive + RSS workflows, save results, and print summary.
    """
    print(f"{datetime.now()} - Starting hybrid fetch...")

    _, existing_links, next_success_key, next_failed_key = load_existing_data()

    archive_records, next_success_key, next_failed_key, archive_stats = process_archive(
        existing_links,
        next_success_key,
        next_failed_key,
    )

    rss_records, next_success_key, next_failed_key, rss_stats = process_rss(
        existing_links,
        next_success_key,
        next_failed_key,
    )

    all_records = archive_records + rss_records

    if all_records:
        save_articles(all_records)
    else:
        print(f"{datetime.now()} - No new records found.")

    # Combine run statistics
    total_articles_added = archive_stats["articles_added"] + rss_stats["articles_added"]
    total_failed_downloads = archive_stats["failed_downloads"] + rss_stats["failed_downloads"]
    total_duplicates_skipped = archive_stats["duplicates_skipped"] + rss_stats["duplicates_skipped"]

    publisher_summary = {publisher: 0 for publisher in rss_feeds.keys()}
    for publisher in publisher_summary:
        publisher_summary[publisher] = archive_stats["publisher_counts"].get(
            publisher, 0
        ) + rss_stats["publisher_counts"].get(publisher, 0)

    # Print run summary
    print("\n================ RUN SUMMARY ================")
    print(f"Articles added:      {total_articles_added}")
    print(f"Failed downloads:    {total_failed_downloads}")
    print(f"Duplicates skipped:  {total_duplicates_skipped}")
    print("Publisher summary:")
    for publisher, count in publisher_summary.items():
        print(f"  {publisher}: {count}")
    print("=============================================\n")


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":
    hybrid_fetch()
