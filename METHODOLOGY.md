# Methodology

Plain-English explanations of every analytical method used in this project.

---

## Data Collection

### Menu History Scraping

The primary dataset is scraped from CrumblCookieFlavors.com, a fan-maintained archive of every weekly Crumbl menu since August 2019. The site uses Elementor's dynamic repeater widget to display each week's flavors as numbered, ranked lists.

The scraper extracts:
- **Week dates** from `<h2>` headers (e.g., "For the week of May 25-30, 2026")
- **Flavor names** from `jet-listing-dynamic-repeater__item` divs, with emoji, ranking numbers, and tags ("New", "NCD") stripped
- **Rank position** (1–10) preserved as a popularity proxy — higher-ranked flavors are listed first

Flavor names are normalized to title case with brand name corrections (e.g., "OREO" → "Oreo") and known duplicates merged (e.g., "Milk Chocolate Chip Cookie" → "Milk Chocolate Chip").

### Flavor Categorization

Each of the 721 unique flavors is assigned a category using keyword-based rules applied to the flavor name. Rules are checked in priority order (first match wins):

1. **Brand Collab** — contains a known brand name (Oreo, Reese's, Snickers, etc.)
2. **Chocolate** — contains "brownie" or "chocolate" (but not "chocolate chip")
3. **Chocolate Chip** — contains both "chocolate" and "chip"
4. **Fruit** — contains a fruit name (lemon, strawberry, berry, etc.)
5. **Peanut Butter** — contains "peanut butter"
6. **Seasonal/Fall-Winter** — contains seasonal keywords (pumpkin, gingerbread, peppermint, etc.)
7. **Sugar Cookie** — contains "sugar" (excluding "brown sugar")
8. **Cake/Cheesecake** — contains "cheesecake," "cake," "pie," or "tart"
9. **Caramel/Toffee** — contains "caramel," "toffee," or "butterscotch"
10. And so on for S'mores, Cookie Dough, Breakfast/Oat, Frozen/Ice Cream, Mint, Red Velvet, Classic Cookie
11. **Specialty** — catch-all for anything that doesn't match above rules

Each flavor also gets a **season tag** (Holiday, Valentine's, Summer, Fall, Year-Round) based on name keywords.

---

## Analysis Methods

### Rotation Overview (Section 1)

Basic descriptive statistics: total weeks, unique flavors, flavors per week, and new flavor introduction rate by quarter. Flavors are grouped into rotation tiers:
- **Permanent** (50+ weeks): the classic standbys
- **Regular** (10–49): fan favorites that return frequently
- **Occasional** (3–9): mid-tier rotation
- **Rare** (2): appeared twice then disappeared
- **One-hit wonder** (1): never returned after debut

### Category Distribution Analysis (Section 2)

Measures what percentage of weekly lineup slots each category fills, tracked year-over-year. This reveals strategic shifts — for example, Chocolate Chip's share dropped from 21% (2019) to 5% (2026) as Crumbl diversified into more Chocolate and Fruit flavors.

Categories are also ranked by average position in the weekly lineup (rank 1 = listed first = most popular). This reveals which categories customers prefer vs. which categories Crumbl is over- or under-serving.

### Flavor Return Analysis (Section 3)

For flavors that appeared at least twice, we compute the average gap (in weeks) between appearances. Combined with rank-based popularity, this creates a four-quadrant classification:

- **Stars**: Popular AND frequently rotated — the core rotation
- **Underused Gems**: Popular but rarely rotated — missed opportunities
- **Overrotated**: Unpopular but frequently rotated — potential menu fatigue
- **Niche**: Unpopular and rarely rotated — correctly deprioritized

"Popular" is defined as above-median popularity score (inverse of average rank). "Frequently rotated" is defined as above-median number of appearances. The quadrant analysis identifies specific flavors in each group.

### Popularity vs. Frequency Correlation (Section 4)

Pearson correlation between rank-based popularity and return frequency. A strong negative correlation (avg rank vs. appearances) would mean Crumbl is already data-driven in rotation decisions. A weak correlation means there are optimization opportunities.

Mismatch analysis identifies the largest gaps between where a flavor ranks in popularity and how often it's rotated — the biggest under-rotated and over-rotated flavors.

### Flavor Return Prediction Model (Section 5)

**Goal**: Predict whether a flavor will return within 12 weeks of its last appearance.

**Approach**: Binary classification using Random Forest (200 trees, max depth 10).

**Training data**: For each flavor × week it appeared, we record whether it returned within the next 12 weeks (target variable). This creates ~1,400 training observations from flavors that appeared at least twice.

**Features**:
- `avg_gap_weeks` — historical average gap between appearances
- `avg_rank` — average weekly rank position
- `total_appearances` — lifetime appearance count
- `appearances_so_far` — appearances up to that point in time
- `month` and `week_of_year` — seasonal timing
- `category` and `season_tag` — one-hot encoded

**Evaluation**: 75/25 train-test split with stratified sampling. Reports accuracy, precision, recall, and F1-score for both classes.

**Prediction output**: For all flavors not seen in the last 12 weeks, the model outputs a return probability, creating a ranked recommendation list.

### Lineup Optimization Scoring (Section 6)

Each week's lineup receives a composite score (0 to 1) based on two factors:

1. **Diversity score** (50% weight): What fraction of available categories are represented? A week with 6 different categories out of 6 flavors scores 1.0; a week with only 2 categories scores lower.

2. **Popularity score** (50% weight): Inverse of the average rank position. A lineup of all #1-ranked flavors would score 1.0.

Tracking the composite score over time reveals whether Crumbl's rotation strategy is improving or declining. Year-over-year averages show the trend.

---

## Visualization Methods

All plots use a Crumbl-branded dark theme (deep brown background `#2B1B17`, warm brown panels `#3B2418`, pink accents `#FF87B2`, cream text `#FFF5E6`).

- **Flavor Frequency**: Horizontal bar chart colored by category
- **Category Distribution**: Stacked area chart showing share shifts over years
- **Popularity vs. Frequency**: Scatter plot with four quadrants (Star, Underused Gem, Overrotated, Niche) and labeled notable flavors
- **Seasonal Heatmap**: seaborn heatmap of category percentage by month (RdPu colormap)
- **Feature Importance**: Horizontal bar chart of Random Forest feature importances
- **Lineup Scores**: Time series scatter with 90-day rolling average trend line
