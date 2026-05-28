"""
Crumbl Cookies — Data Collection
=================================
Scrapes weekly menu history from CrumblCookieFlavors.com,
builds a flavor catalog with category assignments, and pulls
Reddit sentiment data from r/Crumbl_Cookies.

Output files (in data/):
  - menu_history.csv     Weekly flavor lineups (week × flavor)
  - flavor_catalog.csv   Unique flavors with categories and metrics
  - reddit_sentiment.csv Per-flavor Reddit sentiment scores
"""

import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

MENU_URL = "https://crumblcookieflavors.com/all-weeks/"
HEADERS = {"User-Agent": "CrumblAnalysis/1.0 (academic research project)"}


# ============================================================
# SECTION 1: MENU HISTORY SCRAPING
# ============================================================

def scrape_menu_history():
    """Scrape weekly flavor lineups from CrumblCookieFlavors.com."""

    cache_path = os.path.join(DATA_DIR, "menu_history.csv")
    if os.path.exists(cache_path):
        print(f"Found cached menu history: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"  {len(df):,} rows, {df['flavor_name'].nunique()} unique flavors")
        return df

    print("Scraping menu history from CrumblCookieFlavors.com...")
    resp = requests.get(MENU_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    weeks = []
    h2_tags = soup.find_all("h2")

    for h2 in h2_tags:
        text = h2.get_text(strip=True)
        if "for the week of" not in text.lower():
            continue

        week_start, week_end = parse_week_dates(text)
        if week_start is None:
            continue

        flavors = []
        widget = h2.find_parent(class_="elementor-widget")
        if widget is None:
            continue
        flavor_widget = widget.find_next_sibling()
        if flavor_widget is None:
            continue
        rank = 1
        for item in flavor_widget.find_all(class_="jet-listing-dynamic-repeater__item"):
            flavor = parse_flavor(item.get_text(strip=True))
            if flavor:
                flavors.append((*flavor, rank))
                rank += 1

        n_flavors_this_week = len(flavors)
        for flavor_name, is_new, rank in flavors:
            weeks.append({
                "week_start": week_start,
                "week_end": week_end,
                "year": week_start.year,
                "month": week_start.month,
                "flavor_name": flavor_name,
                "is_new": is_new,
                "rank": rank,
                "n_flavors": n_flavors_this_week,
            })

    df = pd.DataFrame(weeks)
    if len(df) == 0:
        print("ERROR: No menu data scraped. Check the site structure.")
        return df
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = pd.to_datetime(df["week_end"])
    df = df.sort_values(["week_start", "flavor_name"]).reset_index(drop=True)

    n_weeks = df["week_start"].nunique()
    n_flavors = df["flavor_name"].nunique()
    print(f"\nScraped {n_weeks} weeks, {len(df):,} total rows, {n_flavors} unique flavors")
    print(f"Date range: {df['week_start'].min().date()} to {df['week_start'].max().date()}")

    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}\n")
    return df


def parse_week_dates(header_text):
    """Extract start and end dates from a week header like 'For the week of May 25-30, 2026'."""

    header_text = header_text.strip()
    # Fix known typos on the source site
    header_text = re.sub(r"\b(\d)\d{4}\b", lambda m: m.group()[-4:], header_text)

    # Pattern: "Month Day-Day, Year" or "Month Day – Day, Year"
    m = re.search(
        r"(\w+)\s+(\d{1,2})\s*[-–]\s*(\d{1,2}),?\s*(\d{4})",
        header_text,
    )
    if m:
        month_str, start_day, end_day, year = m.groups()
        try:
            start = datetime.strptime(f"{month_str} {start_day} {year}", "%B %d %Y")
            end = datetime.strptime(f"{month_str} {end_day} {year}", "%B %d %Y")
            if end < start:
                end = end.replace(month=end.month + 1) if end.month < 12 else end.replace(year=end.year + 1, month=1)
            return start.date(), end.date()
        except ValueError:
            pass

    # Pattern: "Month Day - Month Day, Year" (cross-month weeks)
    m = re.search(
        r"(\w+)\s+(\d{1,2})\s*[-–]\s*(\w+)\s+(\d{1,2}),?\s*(\d{4})",
        header_text,
    )
    if m:
        month1, day1, month2, day2, year = m.groups()
        try:
            start = datetime.strptime(f"{month1} {day1} {year}", "%B %d %Y")
            end = datetime.strptime(f"{month2} {day2} {year}", "%B %d %Y")
            return start.date(), end.date()
        except ValueError:
            pass

    return None, None


def parse_flavor(raw_text):
    """Parse a flavor name from a list item, stripping emoji, numbers, and tags."""

    text = raw_text.strip()

    # Remove leading numbers and dots/parentheses
    text = re.sub(r"^\d+[\.\)]\s*", "", text)

    # Remove emoji and ???? placeholders
    text = re.sub(r"\?{2,}", "", text)
    text = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F\U0001FAD0-\U0001FAFF\U0000FE00-\U0000FE0F\U0000200D]+",
        "",
        text,
    )

    text = text.strip()
    if not text:
        return None

    is_new = bool(re.search(r"New", text, re.IGNORECASE))

    # Remove tags (including when stuck to word end like "Rollnew")
    text = re.sub(r"(?i)\s*new\s*$", "", text)
    text = re.sub(r"(?i)\bnew\b", "", text)
    text = re.sub(r"\(?\s*NCD\s*\)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", "", text)
    text = text.strip().strip(".")

    # Normalize: title case, collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_flavor_name(text)

    if len(text) < 2:
        return None

    return text, is_new


NAME_MERGES = {
    "Milk Chocolate Chip Cookie": "Milk Chocolate Chip",
    "Semi-Sweet Chocolate Chunk Cookie": "Semi-Sweet Chocolate Chunk",
    "Classic Pink Sugar": "Pink Sugar",
    "Pink Sugar Cookie": "Pink Sugar",
    "Chilled Sugar": "Chilled Sugar Cookie",
    "Brownie Batter Cookie": "Brownie Batter",
    "Snickerdoodle Cookie": "Snickerdoodle",
    "Oatmeal Raisin Cookie": "Oatmeal Raisin",
    "Chocolate Chip Cookie": "Chocolate Chip",
    "S'Mores Cookie": "S'mores",
    "S'mores Cookie": "S'mores",
    "Lemon Poppy Seed Cookie": "Lemon Poppy Seed",
    "Strawberry Cheesecake Cookie": "Strawberry Cheesecake",
}


def normalize_flavor_name(name):
    """Normalize flavor name to consistent title case, preserving brand names."""

    brands = {
        "OREO": "Oreo",
        "REESE'S": "Reese's",
        "REESES": "Reese's",
        "BISCOFF": "Biscoff",
        "SNICKERS": "Snickers",
        "TWIX": "Twix",
        "M&M": "M&M",
        "M&M'S": "M&M's",
        "KIT KAT": "Kit Kat",
        "BUTTERFINGER": "Butterfinger",
        "NUTTER BUTTER": "Nutter Butter",
        "NUTELLA": "Nutella",
        "DULCE DE LECHE": "Dulce de Leche",
        "FT.": "ft.",
        "FT": "ft.",
    }

    result = name.title()
    for brand_upper, brand_correct in brands.items():
        pattern = re.compile(re.escape(brand_upper), re.IGNORECASE)
        result = pattern.sub(brand_correct, result)

    result = result.replace("ft..", "ft.").replace("Ft..", "ft.")
    result = NAME_MERGES.get(result, result)
    return result


# ============================================================
# SECTION 2: FLAVOR CATALOG
# ============================================================

CATEGORY_RULES = [
    # Brand collaborations first
    (lambda n: any(b in n.lower() for b in ["oreo", "reese", "snickers", "twix", "kit kat",
     "butterfinger", "nutter butter", "nutella", "biscoff", "m&m"]), "Brand Collab"),

    # Specific types
    (lambda n: "brownie" in n.lower(), "Chocolate"),
    (lambda n: "chocolate" in n.lower() and "chip" in n.lower(), "Chocolate Chip"),
    (lambda n: "chocolate" in n.lower(), "Chocolate"),

    (lambda n: any(f in n.lower() for f in ["lemon", "lime", "orange", "berry", "strawberry",
     "blueberry", "raspberry", "cherry", "peach", "mango", "banana", "apple",
     "coconut", "pineapple", "watermelon", "grape"]), "Fruit"),

    (lambda n: "peanut butter" in n.lower(), "Peanut Butter"),

    (lambda n: any(w in n.lower() for w in ["cinnamon", "pumpkin", "maple", "gingerbread",
     "eggnog", "peppermint", "caramel apple"]), "Seasonal/Fall-Winter"),

    (lambda n: any(w in n.lower() for w in ["sugar cookie", "pink sugar", "sugar"]) and
     "brown sugar" not in n.lower(), "Sugar Cookie"),

    (lambda n: any(w in n.lower() for w in ["cheesecake", "pie", "cake", "cupcake",
     "tiramisu", "tart"]), "Cake/Cheesecake"),

    (lambda n: any(w in n.lower() for w in ["caramel", "toffee", "butterscotch",
     "dulce", "brown sugar"]), "Caramel/Toffee"),

    (lambda n: any(w in n.lower() for w in ["s'more", "smore", "campfire",
     "marshmallow"]), "S'mores"),

    (lambda n: any(w in n.lower() for w in ["cookie dough", "dough"]), "Cookie Dough"),

    (lambda n: any(w in n.lower() for w in ["oatmeal", "granola", "cereal",
     "cornbread", "waffle", "pancake", "french toast", "cinnamon roll"]), "Breakfast/Oat"),

    (lambda n: any(w in n.lower() for w in ["ice cream", "milkshake", "sundae",
     "frozen", "sherbet", "sorbet"]), "Frozen/Ice Cream"),

    (lambda n: any(w in n.lower() for w in ["mint", "peppermint", "andes"]), "Mint"),

    (lambda n: any(w in n.lower() for w in ["red velvet"]), "Red Velvet"),

    (lambda n: "cookie" in n.lower() or "snickerdoodle" in n.lower(), "Classic Cookie"),

    (lambda n: True, "Specialty"),
]


SEASON_RULES = [
    (lambda n: any(w in n.lower() for w in ["pumpkin", "gingerbread", "eggnog",
     "peppermint", "candy cane", "christmas", "holiday", "thanksgiving"]), "Holiday"),
    (lambda n: any(w in n.lower() for w in ["valentine", "heart", "cupid",
     "love"]), "Valentine's"),
    (lambda n: any(w in n.lower() for w in ["watermelon", "lemonade", "tropical",
     "popsicle", "ice cream"]), "Summer"),
    (lambda n: any(w in n.lower() for w in ["apple", "maple", "pumpkin spice",
     "fall", "harvest"]), "Fall"),
    (lambda n: True, "Year-Round"),
]


def categorize_flavor(name):
    """Assign a category to a flavor based on its name."""
    for rule_fn, category in CATEGORY_RULES:
        if rule_fn(name):
            return category
    return "Specialty"


def assign_season(name):
    """Assign a seasonal tag to a flavor."""
    for rule_fn, season in SEASON_RULES:
        if rule_fn(name):
            return season
    return "Year-Round"


def build_flavor_catalog(menu_df):
    """Build a catalog of unique flavors with categories and rotation metrics."""

    cache_path = os.path.join(DATA_DIR, "flavor_catalog.csv")
    if os.path.exists(cache_path):
        print(f"Found cached flavor catalog: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"  {len(df)} unique flavors")
        return df

    print("Building flavor catalog...")

    flavor_stats = (
        menu_df.groupby("flavor_name")
        .agg(
            first_appeared=("week_start", "min"),
            last_appeared=("week_start", "max"),
            times_appeared=("week_start", "nunique"),
            avg_rank=("rank", "mean"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )
    flavor_stats["avg_rank"] = flavor_stats["avg_rank"].round(1)

    # Compute average gap between appearances (for flavors that appeared 2+ times)
    gaps = []
    for flavor in flavor_stats["flavor_name"]:
        weeks = sorted(menu_df.loc[menu_df["flavor_name"] == flavor, "week_start"].unique())
        if len(weeks) >= 2:
            week_gaps = [(weeks[i+1] - weeks[i]).days / 7 for i in range(len(weeks) - 1)]
            gaps.append({"flavor_name": flavor, "avg_gap_weeks": round(sum(week_gaps) / len(week_gaps), 1)})
        else:
            gaps.append({"flavor_name": flavor, "avg_gap_weeks": None})

    gap_df = pd.DataFrame(gaps)
    flavor_stats = flavor_stats.merge(gap_df, on="flavor_name")

    flavor_stats["category"] = flavor_stats["flavor_name"].apply(categorize_flavor)
    flavor_stats["season_tag"] = flavor_stats["flavor_name"].apply(assign_season)

    # Complexity: count words in the name as a rough proxy
    flavor_stats["name_length"] = flavor_stats["flavor_name"].str.split().str.len()
    flavor_stats["has_collab"] = flavor_stats["flavor_name"].str.contains(
        r"\bft\b\.?", case=False, regex=True
    ).astype(int)

    flavor_stats = flavor_stats.sort_values("times_appeared", ascending=False).reset_index(drop=True)

    print(f"\nFlavor catalog: {len(flavor_stats)} unique flavors")
    print(f"\nCategory distribution:")
    cat_counts = flavor_stats["category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat:25s} {count:>4}")

    print(f"\nTop 10 most-rotated flavors:")
    for _, row in flavor_stats.head(10).iterrows():
        print(f"  {row['flavor_name']:40s} {row['times_appeared']:>3} appearances")

    one_timers = (flavor_stats["times_appeared"] == 1).sum()
    print(f"\nOne-hit wonders: {one_timers} ({one_timers / len(flavor_stats) * 100:.0f}%)")

    flavor_stats.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}\n")
    return flavor_stats


# ============================================================
# SECTION 3: REDDIT SENTIMENT
# ============================================================

def collect_reddit_sentiment(flavor_catalog):
    """Pull Reddit sentiment data for flavors from r/Crumbl_Cookies."""

    cache_path = os.path.join(DATA_DIR, "reddit_sentiment.csv")
    if os.path.exists(cache_path):
        print(f"Found cached Reddit sentiment: {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"  {len(df)} flavors with sentiment data")
        return df

    # Check for credentials
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print("WARNING: No .env file found. Skipping Reddit sentiment collection.")
        print("Create a .env file with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        print("See README.md for setup instructions.\n")
        return pd.DataFrame()

    import praw
    from dotenv import load_dotenv
    load_dotenv(env_path)

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "crumbl-analysis/1.0")

    if not client_id or not client_secret:
        print("WARNING: Reddit credentials not found in .env. Skipping sentiment collection.\n")
        return pd.DataFrame()

    print("Collecting Reddit sentiment from r/Crumbl_Cookies...")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    subreddit = reddit.subreddit("Crumbl_Cookies")

    POSITIVE_WORDS = {"love", "amazing", "incredible", "best", "fire", "perfect",
                      "delicious", "bussin", "goated", "phenomenal", "outstanding",
                      "addicting", "addictive", "heavenly", "obsessed", "favorite"}
    NEGATIVE_WORDS = {"mid", "dry", "disappointing", "worst", "bad", "bland",
                      "overrated", "gross", "terrible", "awful", "stale", "meh",
                      "mediocre", "trash", "nasty", "disgusting", "hard pass"}

    # Search for top flavors (limit to top 80 by appearances to stay within API limits)
    top_flavors = flavor_catalog.nlargest(80, "times_appeared")["flavor_name"].tolist()

    results = []
    for i, flavor in enumerate(top_flavors):
        search_term = flavor.replace(" ft.", "").replace(" Ft.", "")
        if len(search_term) > 40:
            search_term = " ".join(search_term.split()[:4])

        mentions = 0
        total_score = 0
        positive = 0
        negative = 0
        neutral = 0

        try:
            for post in subreddit.search(search_term, limit=25, sort="relevance"):
                mentions += 1
                total_score += post.score

                text = (post.title + " " + (post.selftext or "")).lower()
                pos_hits = sum(1 for w in POSITIVE_WORDS if w in text)
                neg_hits = sum(1 for w in NEGATIVE_WORDS if w in text)

                if pos_hits > neg_hits:
                    positive += 1
                elif neg_hits > pos_hits:
                    negative += 1
                else:
                    neutral += 1

        except Exception as e:
            print(f"  Error searching '{flavor}': {e}")

        total = positive + negative + neutral
        results.append({
            "flavor_name": flavor,
            "mention_count": mentions,
            "avg_score": round(total_score / mentions, 1) if mentions > 0 else 0,
            "positive_pct": round(positive / total * 100, 1) if total > 0 else 0,
            "negative_pct": round(negative / total * 100, 1) if total > 0 else 0,
            "sentiment_score": round((positive - negative) / total, 3) if total > 0 else 0,
        })

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(top_flavors)} flavors...")
        time.sleep(1)

    df = pd.DataFrame(results)
    df = df.sort_values("sentiment_score", ascending=False).reset_index(drop=True)

    print(f"\nReddit sentiment: {len(df)} flavors analyzed")
    print(f"  Avg mentions per flavor: {df['mention_count'].mean():.1f}")
    print(f"  Top sentiment: {df.iloc[0]['flavor_name']} ({df.iloc[0]['sentiment_score']:.3f})")
    print(f"  Lowest sentiment: {df.iloc[-1]['flavor_name']} ({df.iloc[-1]['sentiment_score']:.3f})")

    df.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}\n")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CRUMBL COOKIES — DATA COLLECTION")
    print("=" * 60)
    print(f"Source: {MENU_URL}")
    print(f"Output: {DATA_DIR}/\n")

    menu_df = scrape_menu_history()
    catalog_df = build_flavor_catalog(menu_df)
    sentiment_df = collect_reddit_sentiment(catalog_df)

    print("=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    if len(sentiment_df) == 0:
        print("Note: Reddit sentiment was skipped (no credentials).")
        print("Set up .env file and delete data/reddit_sentiment.csv to collect.\n")
