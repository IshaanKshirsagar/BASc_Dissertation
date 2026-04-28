"""
=============================================================================
ProQuest News Article Scraper for Dissertation Research
=============================================================================
Title: "From Tweets to Trades: Analysing Sentiment Signal and Information
        Network for Crypto Market Prediction"

Purpose:
    Scrapes financial news articles from ProQuest (accessed through UCL
    institutional subscription) for sentiment analysis. Targets coverage
    of cryptocurrency markets from the Financial Times, Bloomberg, Reuters,
    The Wall Street Journal, The Guardian, and other major financial news
    sources.

Approach:
    - Uses Selenium with your existing Chrome browser profile so that
      ProQuest recognises you as an authenticated UCL user.
    - Searches are performed in monthly time windows with a per-month
      article cap to ensure even temporal distribution across the
      full 2020-2024 study period.
    - Generous random pauses between requests (15-45 seconds) to
      mimic natural browsing behaviour and avoid triggering rate limits.
    - Full article text is extracted alongside metadata (title, date,
      source, author).
    - Progress is saved after each month so that the scraper can be
      resumed if interrupted.

Prerequisites:
    pip install selenium pandas chromedriver-autoinstaller

    ChromeDriver is handled automatically by chromedriver-autoinstaller,
    which detects your Chrome version and downloads the matching driver.
    No manual ChromeDriver download is needed.

    IMPORTANT: Before running this script, close ALL Chrome windows.
    Selenium needs exclusive access to the Chrome profile.

Usage:
    python scrape_proquest.py

    The script will open a Chrome window. If ProQuest asks you to log in,
    do so manually. The script will wait for you to confirm before
    proceeding with automated collection.

Author: [Candidate Number]
Date: March 2026
=============================================================================
"""

import os
import re
import json
import time
import random
import logging
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# --- Selenium imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException,
        StaleElementReferenceException
    )
except ImportError:
    print("Selenium is not installed. Run: pip install selenium")
    exit(1)

# --- ChromeDriver auto-installer ---
# Automatically downloads and installs the correct ChromeDriver version
# matching your installed Chrome browser. No manual download needed.
try:
    import chromedriver_autoinstaller
    chromedriver_autoinstaller.install()
    print("ChromeDriver installed/verified successfully.")
except ImportError:
    print("chromedriver-autoinstaller not found. Run: pip install chromedriver-autoinstaller")
    print("Alternatively, download ChromeDriver manually from:")
    print("https://googlechromelabs.github.io/chrome-for-testing/")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Output directories ---
OUTPUT_DIR = Path("data/raw/sentiment/news")
PROGRESS_DIR = Path("data/raw/sentiment/news/progress")
DOCS_DIR = Path("docs")

for d in [OUTPUT_DIR, PROGRESS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Study period ---
START_YEAR = 2020
START_MONTH = 1
END_YEAR = 2024
END_MONTH = 12

# --- Search queries per asset ---
# Each query targets a specific crypto asset within ProQuest's search.
# The queries use ProQuest's search syntax.
SEARCH_QUERIES = {
    "bitcoin": '(bitcoin OR btc) AND (price OR market OR trading OR regulation OR crash OR rally)',
    "ethereum": '(ethereum OR ether OR eth) AND (price OR market OR trading OR regulation OR crash OR rally)',
    "dogecoin": '(dogecoin OR doge) AND (price OR market OR trading OR meme OR rally)',
    "shiba_inu": '("shiba inu" OR shib OR shibarmy) AND (price OR market OR trading OR meme OR rally)',
    "crypto_general": '(cryptocurrency OR "crypto market" OR "digital assets") AND (price OR market OR regulation OR crash OR rally OR bull OR bear)',
}

# --- Source filter ---
# Restrict to major financial news sources.
# These are the publication names as they appear in ProQuest.
TARGET_SOURCES = [
    "Financial Times",
    "Bloomberg",
    "Reuters",
    "The Wall Street Journal",
    "The Guardian",
    "The New York Times",
    "The Washington Post",
    "The Telegraph",
    "The Economist",
    "CNBC",
    "MarketWatch",
    "CoinDesk",
    "The Independent",
    "The Times",
]

# --- Rate limiting ---
# Random pause range in seconds between page loads.
MIN_PAUSE = 15
MAX_PAUSE = 45
# Longer pause every N articles to be extra cautious.
LONG_PAUSE_EVERY = 20
LONG_PAUSE_MIN = 60
LONG_PAUSE_MAX = 120

# --- Collection budget ---
MAX_ARTICLES_PER_MONTH_PER_QUERY = 50  # Per search query per month.
# With 5 queries x 60 months x 50 articles = up to 15,000 articles total.
# Adjust this based on how much time you have.

# --- Chrome profile ---
# This uses your default Chrome profile so ProQuest sees you as logged in.
# On Windows, the default profile is typically at:
#   C:\Users\<YourName>\AppData\Local\Google\Chrome\User Data
# On Mac:
#   ~/Library/Application Support/Google/Chrome
# On Linux:
#   ~/.config/google-chrome
#
# IMPORTANT: Close all Chrome windows before running this script.
# Selenium needs exclusive access to the profile directory.

CHROME_PROFILE_DIR = None  # Set to None to auto-detect, or paste your path.

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DOCS_DIR / "proquest_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_chrome_profile_path():
    """Auto-detect the Chrome user data directory."""
    import platform
    system = platform.system()
    home = Path.home()
    if system == "Windows":
        return str(home / "AppData" / "Local" / "Google" / "Chrome" / "User Data")
    elif system == "Darwin":  # macOS
        return str(home / "Library" / "Application Support" / "Google" / "Chrome")
    else:  # Linux
        return str(home / ".config" / "google-chrome")


def random_pause(short=True):
    """Sleep for a random duration to mimic human browsing."""
    if short:
        duration = random.uniform(MIN_PAUSE, MAX_PAUSE)
    else:
        duration = random.uniform(LONG_PAUSE_MIN, LONG_PAUSE_MAX)
    logger.info(f"    Pausing for {duration:.0f}s...")
    time.sleep(duration)


def build_proquest_search_url(query, date_from, date_to):
    """
    Build a ProQuest advanced search URL with date range filter.

    Parameters
    ----------
    query : str
        The search query string.
    date_from : str
        Start date in YYYY-MM-DD format.
    date_to : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    str
        The full ProQuest search URL.
    """
    import urllib.parse

    # ProQuest URL structure for search with date range and news source filter.
    base_url = "https://www.proquest.com/results"
    params = {
        "query": query,
        "filter": "stype(Newspapers)",
        "fromdate": date_from,
        "todate": date_to,
        "FT": "1",  # Full text available
        "language": "English",
        "sortby": "DateDesc",
    }
    param_string = urllib.parse.urlencode(params, safe="()")
    return f"{base_url}?{param_string}"


def load_progress(asset_name):
    """Load the progress file for a given asset to enable resumption."""
    progress_file = PROGRESS_DIR / f"{asset_name}_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"completed_months": [], "total_articles": 0}


def save_progress(asset_name, progress):
    """Save progress for a given asset."""
    progress_file = PROGRESS_DIR / f"{asset_name}_progress.json"
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def generate_monthly_windows(start_year, start_month, end_year, end_month):
    """Generate a list of (start_date, end_date) tuples for each month."""
    windows = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        start_date = f"{year}-{month:02d}-01"
        # Calculate last day of month.
        if month == 12:
            next_year, next_month = year + 1, 1
        else:
            next_year, next_month = year, month + 1
        # End date is the day before the next month starts.
        from datetime import date, timedelta
        end_date_obj = date(next_year, next_month, 1) - timedelta(days=1)
        end_date = end_date_obj.strftime("%Y-%m-%d")
        windows.append((start_date, end_date, f"{year}-{month:02d}"))
        # Move to next month.
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return windows


# =============================================================================
# SELENIUM BROWSER SETUP
# =============================================================================

def create_browser():
    """
    Create a Selenium Chrome browser instance using the user's existing
    Chrome profile for authentication.
    """
    options = Options()

    # Use existing Chrome profile for authentication.
    profile_path = CHROME_PROFILE_DIR or get_chrome_profile_path()
    if os.path.exists(profile_path):
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument("--profile-directory=Default")
        logger.info(f"Using Chrome profile: {profile_path}")
    else:
        logger.warning(
            f"Chrome profile not found at {profile_path}. "
            f"You may need to log in to ProQuest manually."
        )

    # General options.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    # Create the browser.
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)

    return driver


# =============================================================================
# ARTICLE EXTRACTION
# =============================================================================

def extract_article_from_page(driver):
    """
    Extract article metadata and full text from a ProQuest article page.

    Returns
    -------
    dict or None
        Dictionary containing article data, or None if extraction failed.
    """
    try:
        article = {}

        # Title.
        try:
            title_el = driver.find_element(By.CSS_SELECTOR, "h1.titleLink, h1[class*='title']")
            article["title"] = title_el.text.strip()
        except NoSuchElementException:
            try:
                title_el = driver.find_element(By.TAG_NAME, "h1")
                article["title"] = title_el.text.strip()
            except NoSuchElementException:
                article["title"] = ""

        # Publication date.
        try:
            date_el = driver.find_element(
                By.CSS_SELECTOR,
                "span[class*='pub_date'], span[class*='date'], .publicationDate"
            )
            article["pub_date"] = date_el.text.strip()
        except NoSuchElementException:
            article["pub_date"] = ""

        # Source / publication name.
        try:
            source_el = driver.find_element(
                By.CSS_SELECTOR,
                "a[class*='pubTitle'], span[class*='source'], .publicationName"
            )
            article["source"] = source_el.text.strip()
        except NoSuchElementException:
            article["source"] = ""

        # Author.
        try:
            author_el = driver.find_element(
                By.CSS_SELECTOR,
                "span[class*='author'], a[class*='author'], .authorName"
            )
            article["author"] = author_el.text.strip()
        except NoSuchElementException:
            article["author"] = ""

        # Full text.
        try:
            # ProQuest typically wraps full text in a div with specific classes.
            fulltext_el = driver.find_element(
                By.CSS_SELECTOR,
                "div[id*='fullText'], div[class*='fullText'], "
                "div[class*='document_text'], div.fullTextBody, "
                "div[id='fullTextZone']"
            )
            article["full_text"] = fulltext_el.text.strip()
        except NoSuchElementException:
            # Try broader selectors.
            try:
                fulltext_el = driver.find_element(
                    By.CSS_SELECTOR,
                    "div.abstractBody, div[class*='abstract']"
                )
                article["full_text"] = fulltext_el.text.strip()
            except NoSuchElementException:
                article["full_text"] = ""

        # Article URL.
        article["url"] = driver.current_url

        return article if article["title"] or article["full_text"] else None

    except Exception as e:
        logger.error(f"    Error extracting article: {e}")
        return None


def scrape_search_results_page(driver, max_articles=50):
    """
    From a ProQuest search results page, click into each article,
    extract its content, and return to the results page.

    Parameters
    ----------
    driver : webdriver.Chrome
        The Selenium browser instance.
    max_articles : int
        Maximum number of articles to extract from this results page.

    Returns
    -------
    list
        List of article dictionaries.
    """
    articles = []
    article_count = 0

    try:
        # Wait for results to load.
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "li.resultItem, div[class*='result']")
            )
        )
    except TimeoutException:
        logger.warning("    No results found or page took too long to load.")
        return articles

    while article_count < max_articles:
        try:
            # Re-find result links each time (DOM may have changed).
            result_links = driver.find_elements(
                By.CSS_SELECTOR,
                "a.resultTitle, a[class*='titleLink'], "
                "li.resultItem a[href*='docview']"
            )

            if article_count >= len(result_links):
                # Try to go to next page of results.
                try:
                    next_btn = driver.find_element(
                        By.CSS_SELECTOR,
                        "a[class*='nextPage'], button[class*='next'], "
                        "a[aria-label='Next page']"
                    )
                    next_btn.click()
                    random_pause()
                    article_count_on_page = 0
                    continue
                except NoSuchElementException:
                    logger.info("    No more result pages.")
                    break

            # Click on the article link.
            link = result_links[article_count]
            link_text = link.text.strip()[:80]
            logger.info(f"    [{article_count + 1}] {link_text}...")

            # Scroll to element and click.
            driver.execute_script("arguments[0].scrollIntoView(true);", link)
            time.sleep(1)
            link.click()

            # Wait for article page to load.
            time.sleep(random.uniform(3, 6))

            # Check if we need to click "Full text" link.
            try:
                ft_link = driver.find_element(
                    By.XPATH,
                    "//a[contains(text(), 'Full text')] | "
                    "//a[contains(text(), 'Full Text')]"
                )
                ft_link.click()
                time.sleep(random.uniform(2, 4))
            except NoSuchElementException:
                pass  # Already on full text view.

            # Extract article content.
            article = extract_article_from_page(driver)
            if article:
                article["scrape_timestamp"] = datetime.now(
                    timezone.utc
                ).isoformat()
                articles.append(article)
                logger.info(
                    f"    Extracted: {article['source']} | "
                    f"{len(article.get('full_text', ''))} chars"
                )

            # Go back to results.
            driver.back()
            time.sleep(random.uniform(2, 4))
            # If we were on a full text sub-page, go back again.
            if "docview" not in driver.current_url:
                pass  # Already on results
            else:
                driver.back()
                time.sleep(random.uniform(2, 3))

            article_count += 1

            # Rate limiting.
            if article_count % LONG_PAUSE_EVERY == 0:
                random_pause(short=False)
            else:
                random_pause(short=True)

        except StaleElementReferenceException:
            logger.warning("    Stale element, refreshing results...")
            driver.refresh()
            time.sleep(5)
            continue
        except Exception as e:
            logger.error(f"    Error on article {article_count + 1}: {e}")
            article_count += 1
            random_pause()
            continue

    return articles


# =============================================================================
# MAIN SCRAPING PIPELINE
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PROQUEST NEWS ARTICLE SCRAPER")
    logger.info(f"Period: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    logger.info(f"Queries: {len(SEARCH_QUERIES)}")
    logger.info(f"Articles per month per query: {MAX_ARTICLES_PER_MONTH_PER_QUERY}")
    logger.info("=" * 70)

    # Generate monthly windows.
    windows = generate_monthly_windows(
        START_YEAR, START_MONTH, END_YEAR, END_MONTH
    )
    logger.info(f"Total monthly windows: {len(windows)}")

    # Prompt user to close Chrome.
    print("\n" + "=" * 60)
    print("IMPORTANT: Close ALL Chrome windows before continuing.")
    print("The scraper needs exclusive access to your Chrome profile")
    print("so that ProQuest recognises you as a UCL user.")
    print("=" * 60)
    input("\nPress ENTER when all Chrome windows are closed...")

    # Create browser.
    logger.info("Starting Chrome browser...")
    driver = create_browser()

    # Navigate to ProQuest to verify authentication.
    logger.info("Navigating to ProQuest...")
    driver.get("https://www.proquest.com/")
    time.sleep(5)

    print("\n" + "=" * 60)
    print("CHECK: Does the ProQuest page show")
    print("'Access provided by UNIVERSITY COLLEGE LONDON'?")
    print("If not, log in manually now through the browser window.")
    print("=" * 60)
    input("\nPress ENTER when you can see ProQuest with UCL access...")

    logger.info("Authentication confirmed. Starting collection.\n")

    # --- Main collection loop ---
    for query_name, query_text in SEARCH_QUERIES.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"QUERY: {query_name}")
        logger.info(f"{'=' * 60}")

        # Load progress for this query.
        progress = load_progress(query_name)
        completed_months = set(progress["completed_months"])

        all_articles = []

        # Load any previously collected articles.
        existing_file = OUTPUT_DIR / f"{query_name}_articles.csv"
        if existing_file.exists():
            existing_df = pd.read_csv(existing_file)
            all_articles = existing_df.to_dict("records")
            logger.info(f"Loaded {len(all_articles)} previously collected articles.")

        for date_from, date_to, month_label in windows:
            if month_label in completed_months:
                logger.info(f"  {month_label}: Already completed. Skipping.")
                continue

            logger.info(f"\n  --- {month_label} ({date_from} to {date_to}) ---")

            # Build search URL.
            search_url = build_proquest_search_url(
                query_text, date_from, date_to
            )
            logger.info(f"  URL: {search_url[:100]}...")

            try:
                driver.get(search_url)
                time.sleep(random.uniform(5, 8))

                # Extract articles from this search.
                month_articles = scrape_search_results_page(
                    driver, max_articles=MAX_ARTICLES_PER_MONTH_PER_QUERY
                )

                # Tag articles with query metadata.
                for art in month_articles:
                    art["query_name"] = query_name
                    art["month"] = month_label
                    art["date_from"] = date_from
                    art["date_to"] = date_to

                all_articles.extend(month_articles)
                logger.info(
                    f"  {month_label}: Collected {len(month_articles)} articles "
                    f"(total: {len(all_articles)})"
                )

                # Mark month as completed and save progress.
                completed_months.add(month_label)
                progress["completed_months"] = list(completed_months)
                progress["total_articles"] = len(all_articles)
                save_progress(query_name, progress)

                # Save articles after each month (for safety).
                df = pd.DataFrame(all_articles)
                df.to_csv(existing_file, index=False)

            except Exception as e:
                logger.error(f"  Error for {month_label}: {e}")
                random_pause(short=False)
                continue

            # Pause between months.
            random_pause()

        # Final save for this query.
        if all_articles:
            df = pd.DataFrame(all_articles)
            df.to_csv(existing_file, index=False)
            logger.info(
                f"\nQuery '{query_name}' complete: "
                f"{len(all_articles)} articles saved to {existing_file}"
            )

    # --- Combine all queries into master file ---
    logger.info("\n" + "=" * 60)
    logger.info("COMBINING ALL QUERIES")
    all_files = list(OUTPUT_DIR.glob("*_articles.csv"))
    if all_files:
        master = pd.concat(
            [pd.read_csv(f) for f in all_files], ignore_index=True
        )
        master = master.drop_duplicates(subset=["url"], keep="first")
        master_path = OUTPUT_DIR / "all_proquest_articles.csv"
        master.to_csv(master_path, index=False)
        logger.info(f"Master file: {master_path} ({len(master)} unique articles)")
    logger.info("=" * 60)

    # Clean up.
    driver.quit()
    logger.info("Browser closed. ProQuest scraping COMPLETE.")


if __name__ == "__main__":
    main()
