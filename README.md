# Crumbl Cookies — Menu Performance Analytics

A data-driven analysis of Crumbl's weekly rotating menu using 330+ weeks of real scraped data. Built as a targeted demonstration for data analyst roles at Crumbl Cookies.

## What This Project Does

Scrapes Crumbl's full menu rotation history (Aug 2019 – May 2026) and analyzes flavor performance, category trends, and rotation optimization:

1. **Rotation Overview** — 721 unique flavors across 330 weeks; 57% are one-hit wonders that never return
2. **Category Distribution** — Tracks how the menu mix shifted over time (Chocolate Chip dropped from 21% to 5% while Chocolate rose from 8% to 23%)
3. **Flavor Return Analysis** — Identifies "Stars," "Underused Gems," "Overrotated," and "Niche" flavors using rank-based popularity vs. rotation frequency
4. **Popularity vs. Frequency** — Pinpoints mismatches: high-ranked flavors that rarely return and low-ranked flavors that keep appearing
5. **Return Prediction Model** — Random Forest classifier predicts whether a flavor will return within 12 weeks (89.4% accuracy)
6. **Lineup Optimization** — Scores each week's lineup on category diversity and popularity, revealing a declining trend (0.672 → 0.571)

## Key Findings

| Metric | Value |
|---|---|
| Total unique flavors | 721 |
| One-hit wonders | 57% of all flavors |
| Most overrotated flavor | Milk Chocolate Chip (248 appearances, avg rank 6.2) |
| Most popular category | Mint (avg rank 2.6) |
| Prediction model accuracy | 89.4% |
| Top return predictor | Average gap between appearances (0.37 importance) |
| Lineup optimization trend | Declining from 2019 to 2026 |

## Data Source

**CrumblCookieFlavors.com** — a fan-maintained archive of every weekly Crumbl menu since August 2019. Each week's flavors are ranked 1–10 (or 1–6 in earlier years), providing a built-in popularity signal. No synthetic data — every number in this analysis comes from real public data.

See [METHODOLOGY.md](METHODOLOGY.md) for detailed explanations of every analytical method.

## Project Structure

```
crumbl/
├── data_collection.py      # Scrape menu history, build flavor catalog
├── analysis.py              # Full analysis pipeline + visualizations
├── METHODOLOGY.md           # Plain-English method explanations
├── README.md
├── .gitignore
├── data/                    # CSVs (gitignored)
│   ├── menu_history.csv
│   ├── flavor_catalog.csv
│   └── analysis_exports/    # 6 Tableau-ready CSVs
└── plots/                   # 6 visualization PNGs
    ├── flavor_frequency.png
    ├── category_distribution.png
    ├── popularity_vs_frequency.png
    ├── seasonal_heatmap.png
    ├── feature_importance.png
    └── lineup_scores.png
```

## Running the Analysis

```bash
# Install dependencies
pip install pandas numpy requests beautifulsoup4 scikit-learn matplotlib seaborn

# Step 1: Scrape menu data (~30 seconds)
python data_collection.py

# Step 2: Run full analysis (~15 seconds)
python analysis.py
```

## Output

### Visualizations

| Plot | Description |
|---|---|
| Flavor Frequency | Top 20 most-rotated flavors with category coloring |
| Category Distribution | Stacked area chart of category mix shifts 2019–2026 |
| Popularity vs. Frequency | Scatter plot with quadrant labels (Stars, Underused Gems, Overrotated, Niche) |
| Seasonal Heatmap | Category frequency by month — when each type appears |
| Feature Importance | What drives flavor returns (Random Forest) |
| Lineup Scores | Weekly optimization score trend over time |

### Tableau Data

Six CSVs designed for direct import into Tableau Public, covering flavor summary, weekly lineups with categories, category trends, model predictions, feature importance, and lineup optimization scores.

## Technical Stack

- **Python** — pandas, numpy, matplotlib, seaborn
- **Web Scraping** — requests, BeautifulSoup4
- **Machine Learning** — scikit-learn (Random Forest)
- **Visualization** — Tableau Public, matplotlib
