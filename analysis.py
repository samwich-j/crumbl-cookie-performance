"""
Crumbl Cookies — Menu Performance Analytics
=============================================
Analyzes flavor rotation patterns, category distribution, popularity
signals, and builds a flavor return prediction model using real
scraped data from CrumblCookieFlavors.com.

Input: data/menu_history.csv, data/flavor_catalog.csv
Output: plots/, data/analysis_exports/
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
EXPORT_DIR = os.path.join(DATA_DIR, "analysis_exports")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Crumbl brand palette
PINK = "#FF87B2"
CREAM = "#FFF5E6"
DARK = "#2B1B17"
BROWN = "#3B2418"
ACCENT_PINK = "#FF4081"
LIGHT_PINK = "#FFB6C1"
GOLD = "#D4A76A"

CATEGORY_COLORS = {
    "Brand Collab": "#FF4081",
    "Fruit": "#FF8C42",
    "Cake/Cheesecake": "#FFD166",
    "Chocolate": "#6B3A2A",
    "Specialty": "#9B59B6",
    "Seasonal/Fall-Winter": "#E07B54",
    "Sugar Cookie": "#FF87B2",
    "Classic Cookie": "#D4A76A",
    "Chocolate Chip": "#8B5E3C",
    "Breakfast/Oat": "#C4A882",
    "Peanut Butter": "#D4943A",
    "Frozen/Ice Cream": "#5BC0EB",
    "Caramel/Toffee": "#B8860B",
    "Mint": "#4A7C59",
    "Red Velvet": "#C8102E",
    "S'mores": "#8B4513",
    "Cookie Dough": "#DEB887",
}


def setup_style():
    """Set dark Crumbl-branded plot style."""
    plt.rcParams.update({
        "figure.facecolor": DARK,
        "axes.facecolor": BROWN,
        "text.color": CREAM,
        "axes.labelcolor": CREAM,
        "xtick.color": CREAM,
        "ytick.color": CREAM,
        "axes.edgecolor": "#5A3D2E",
        "legend.facecolor": BROWN,
        "legend.edgecolor": "#5A3D2E",
        "font.family": "sans-serif",
        "font.size": 10,
    })


# ============================================================
# SECTION 1: MENU ROTATION OVERVIEW
# ============================================================

def rotation_overview(menu_df, catalog_df):
    """Print summary stats on menu rotation patterns."""

    print("=" * 60)
    print("SECTION 1: MENU ROTATION OVERVIEW")
    print("=" * 60)

    n_weeks = menu_df["week_start"].nunique()
    n_flavors = menu_df["flavor_name"].nunique()
    n_rows = len(menu_df)
    date_min = menu_df["week_start"].min()
    date_max = menu_df["week_start"].max()

    print(f"Total weeks:        {n_weeks}")
    print(f"Total flavor slots:  {n_rows:,}")
    print(f"Unique flavors:     {n_flavors}")
    print(f"Date range:         {date_min.date()} to {date_max.date()}")

    # Flavors per week over time
    fpw = menu_df.groupby("week_start")["flavor_name"].nunique()
    print(f"\nFlavors per week:   {fpw.mean():.1f} avg, {fpw.min()} min, {fpw.max()} max")

    # New flavors per quarter
    new_flavors = menu_df[menu_df["is_new"]].copy()
    new_flavors["quarter"] = new_flavors["week_start"].dt.to_period("Q")
    new_per_q = new_flavors.groupby("quarter")["flavor_name"].nunique()
    print(f"\nNew flavors per quarter:")
    for q in new_per_q.index[-8:]:
        print(f"  {q}: {new_per_q[q]}")

    # Rotation tiers
    print(f"\nRotation tiers:")
    tiers = [
        ("Permanent (50+ weeks)", catalog_df["times_appeared"] >= 50),
        ("Regular (10-49 weeks)", (catalog_df["times_appeared"] >= 10) & (catalog_df["times_appeared"] < 50)),
        ("Occasional (3-9 weeks)", (catalog_df["times_appeared"] >= 3) & (catalog_df["times_appeared"] < 10)),
        ("Rare (2 weeks)", catalog_df["times_appeared"] == 2),
        ("One-hit wonder (1 week)", catalog_df["times_appeared"] == 1),
    ]
    for label, mask in tiers:
        count = mask.sum()
        pct = count / len(catalog_df) * 100
        print(f"  {label:30s} {count:>4} ({pct:4.1f}%)")

    print()
    return fpw


# ============================================================
# SECTION 2: CATEGORY DISTRIBUTION ANALYSIS
# ============================================================

def category_analysis(menu_df, catalog_df):
    """Analyze category mix in weekly lineups over time."""

    print("=" * 60)
    print("SECTION 2: CATEGORY DISTRIBUTION ANALYSIS")
    print("=" * 60)

    menu_cats = menu_df.merge(
        catalog_df[["flavor_name", "category"]], on="flavor_name", how="left"
    )

    # Overall category share
    cat_share = menu_cats["category"].value_counts(normalize=True) * 100
    print("\nOverall category share (% of all flavor slots):")
    for cat, pct in cat_share.items():
        print(f"  {cat:25s} {pct:5.1f}%")

    # Category trends by year
    menu_cats["year"] = menu_cats["week_start"].dt.year
    yearly_cats = (
        menu_cats.groupby(["year", "category"])
        .size()
        .unstack(fill_value=0)
    )
    yearly_pcts = yearly_cats.div(yearly_cats.sum(axis=1), axis=0) * 100

    print("\nCategory share by year (top 5 categories):")
    top5_cats = cat_share.head(5).index.tolist()
    print(f"  {'Year':>6}", end="")
    for cat in top5_cats:
        print(f"  {cat:>15s}", end="")
    print()
    for year in yearly_pcts.index:
        if year > 2026:
            continue
        print(f"  {year:>6}", end="")
        for cat in top5_cats:
            val = yearly_pcts.loc[year, cat] if cat in yearly_pcts.columns else 0
            print(f"  {val:>14.1f}%", end="")
        print()

    # Category by avg rank (popularity)
    cat_rank = menu_cats.groupby("category")["rank"].mean().sort_values()
    print("\nCategory by avg rank (lower = more popular):")
    for cat, rank in cat_rank.items():
        print(f"  {cat:25s} {rank:5.1f}")

    print()
    return menu_cats, yearly_pcts


# ============================================================
# SECTION 3: FLAVOR RETURN ANALYSIS
# ============================================================

def flavor_return_analysis(menu_df, catalog_df):
    """Analyze which flavors return, how often, and when."""

    print("=" * 60)
    print("SECTION 3: FLAVOR RETURN ANALYSIS")
    print("=" * 60)

    returners = catalog_df[catalog_df["times_appeared"] >= 2].copy()
    print(f"\nFlavors that returned at least once: {len(returners)} / {len(catalog_df)}")

    # Gap analysis
    print(f"\nAverage gap between appearances:")
    print(f"  Mean:   {returners['avg_gap_weeks'].mean():.1f} weeks")
    print(f"  Median: {returners['avg_gap_weeks'].median():.1f} weeks")
    print(f"  Min:    {returners['avg_gap_weeks'].min():.1f} weeks")
    print(f"  Max:    {returners['avg_gap_weeks'].max():.1f} weeks")

    # Overplayed vs underused (for flavors with enough data)
    qualified = catalog_df[catalog_df["times_appeared"] >= 3].copy()
    qualified["popularity_score"] = 1 / qualified["avg_rank"]
    median_pop = qualified["popularity_score"].median()
    median_appearances = qualified["times_appeared"].median()

    qualified["quadrant"] = "Average"
    qualified.loc[
        (qualified["popularity_score"] > median_pop) & (qualified["times_appeared"] < median_appearances),
        "quadrant",
    ] = "Underused Gem"
    qualified.loc[
        (qualified["popularity_score"] < median_pop) & (qualified["times_appeared"] > median_appearances),
        "quadrant",
    ] = "Overrotated"
    qualified.loc[
        (qualified["popularity_score"] > median_pop) & (qualified["times_appeared"] >= median_appearances),
        "quadrant",
    ] = "Star"
    qualified.loc[
        (qualified["popularity_score"] < median_pop) & (qualified["times_appeared"] <= median_appearances),
        "quadrant",
    ] = "Niche"

    print(f"\nQuadrant analysis ({len(qualified)} flavors with 3+ appearances):")
    for quad in ["Star", "Underused Gem", "Overrotated", "Niche"]:
        q_df = qualified[qualified["quadrant"] == quad]
        print(f"\n  {quad} ({len(q_df)} flavors):")
        for _, row in q_df.head(5).iterrows():
            print(f"    {row['flavor_name']:40s}  rank {row['avg_rank']:4.1f}  x{row['times_appeared']}")

    print()
    return qualified


# ============================================================
# SECTION 4: POPULARITY vs RETURN FREQUENCY
# ============================================================

def popularity_vs_frequency(catalog_df):
    """Analyze correlation between rank-based popularity and return frequency."""

    print("=" * 60)
    print("SECTION 4: POPULARITY vs RETURN FREQUENCY")
    print("=" * 60)

    qualified = catalog_df[catalog_df["times_appeared"] >= 2].copy()
    qualified["popularity_score"] = 1 / qualified["avg_rank"]

    corr = qualified["popularity_score"].corr(qualified["times_appeared"])
    rank_corr = qualified["avg_rank"].corr(qualified["times_appeared"])
    print(f"\nCorrelation (popularity score vs appearances): {corr:.3f}")
    print(f"Correlation (avg rank vs appearances):          {rank_corr:.3f}")
    print("  (Negative = higher-ranked flavors appear more often — expected)")

    # Top mismatches: popular but rarely returning
    qualified_enough = qualified[qualified["times_appeared"] >= 3].copy()
    qualified_enough["rank_percentile"] = qualified_enough["avg_rank"].rank(pct=True)
    qualified_enough["freq_percentile"] = qualified_enough["times_appeared"].rank(pct=True)
    qualified_enough["mismatch"] = qualified_enough["freq_percentile"] - qualified_enough["rank_percentile"]

    print(f"\nBiggest mismatches — popular but under-rotated:")
    under = qualified_enough.nsmallest(5, "mismatch")
    for _, r in under.iterrows():
        print(f"  {r['flavor_name']:40s}  rank {r['avg_rank']:4.1f}  x{r['times_appeared']}")

    print(f"\nBiggest mismatches — unpopular but over-rotated:")
    over = qualified_enough.nlargest(5, "mismatch")
    for _, r in over.iterrows():
        print(f"  {r['flavor_name']:40s}  rank {r['avg_rank']:4.1f}  x{r['times_appeared']}")

    print()
    return qualified_enough


# ============================================================
# SECTION 5: FLAVOR RETURN PREDICTION MODEL
# ============================================================

def build_prediction_model(menu_df, catalog_df):
    """Predict whether a flavor will return within 12 weeks."""

    print("=" * 60)
    print("SECTION 5: FLAVOR RETURN PREDICTION MODEL")
    print("=" * 60)

    # Build training data: for each flavor × week it appeared,
    # did it return within the next 12 weeks?
    qualified = catalog_df[catalog_df["times_appeared"] >= 2].copy()
    qualified_names = set(qualified["flavor_name"])

    cat_map = catalog_df.set_index("flavor_name")["category"].to_dict()
    season_map = catalog_df.set_index("flavor_name")["season_tag"].to_dict()
    appearances_map = catalog_df.set_index("flavor_name")["times_appeared"].to_dict()
    avg_gap_map = catalog_df.set_index("flavor_name")["avg_gap_weeks"].to_dict()
    avg_rank_map = catalog_df.set_index("flavor_name")["avg_rank"].to_dict()

    all_weeks = sorted(menu_df["week_start"].unique())

    records = []
    for flavor in qualified_names:
        flavor_weeks = sorted(menu_df.loc[menu_df["flavor_name"] == flavor, "week_start"].unique())

        for i, week in enumerate(flavor_weeks[:-1]):
            next_appearance = flavor_weeks[i + 1]
            gap_weeks = (next_appearance - week).days / 7
            returned_within_12 = 1 if gap_weeks <= 12 else 0

            week_idx = list(all_weeks).index(week)
            prev_appearances = sum(1 for w in flavor_weeks if w < week)

            records.append({
                "flavor_name": flavor,
                "week": week,
                "month": week.month,
                "category": cat_map.get(flavor, "Specialty"),
                "season_tag": season_map.get(flavor, "Year-Round"),
                "total_appearances": appearances_map.get(flavor, 1),
                "avg_gap_weeks": avg_gap_map.get(flavor),
                "avg_rank": avg_rank_map.get(flavor, 5),
                "appearances_so_far": prev_appearances + 1,
                "week_of_year": week.isocalendar()[1],
                "returned_within_12": returned_within_12,
            })

    model_df = pd.DataFrame(records)
    model_df = model_df.dropna()

    # Encode categoricals
    model_df = pd.get_dummies(model_df, columns=["category", "season_tag"], drop_first=True)

    feature_cols = [c for c in model_df.columns if c not in
                    ["flavor_name", "week", "returned_within_12"]]
    X = model_df[feature_cols]
    y = model_df["returned_within_12"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel: Random Forest (200 trees, max_depth=10)")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Accuracy:         {accuracy:.1%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Return", "Returns <12wk"]))

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("Top 10 features:")
    for _, r in importance.head(10).iterrows():
        print(f"  {r['feature']:35s} {r['importance']:.4f}")

    # Predict return probability for all flavors not recently seen
    latest_week = menu_df["week_start"].max()
    recent_cutoff = latest_week - pd.Timedelta(weeks=12)
    recent_flavors = set(
        menu_df.loc[menu_df["week_start"] >= recent_cutoff, "flavor_name"]
    )

    candidates = catalog_df[
        (catalog_df["times_appeared"] >= 2) &
        (~catalog_df["flavor_name"].isin(recent_flavors))
    ].copy()

    if len(candidates) > 0:
        pred_records = []
        for _, row in candidates.iterrows():
            rec = {
                "month": latest_week.month,
                "total_appearances": row["times_appeared"],
                "avg_gap_weeks": row["avg_gap_weeks"],
                "avg_rank": row["avg_rank"],
                "appearances_so_far": row["times_appeared"],
                "week_of_year": latest_week.isocalendar()[1],
            }
            for col in feature_cols:
                if col not in rec:
                    if col.startswith("category_"):
                        cat_val = col.replace("category_", "")
                        rec[col] = 1 if row["category"] == cat_val else 0
                    elif col.startswith("season_tag_"):
                        season_val = col.replace("season_tag_", "")
                        rec[col] = 1 if row["season_tag"] == season_val else 0
                    else:
                        rec[col] = 0
            pred_records.append(rec)

        pred_X = pd.DataFrame(pred_records)[feature_cols]
        candidates["return_probability"] = clf.predict_proba(pred_X)[:, 1]
        candidates = candidates.sort_values("return_probability", ascending=False)

        print(f"\nTop 10 flavors most likely to return:")
        for _, r in candidates.head(10).iterrows():
            print(f"  {r['flavor_name']:40s}  {r['return_probability']:.1%}  (rank {r['avg_rank']:.1f}, x{r['times_appeared']})")

    print()
    return importance, clf, candidates


# ============================================================
# SECTION 6: OPTIMAL LINEUP RECOMMENDATION
# ============================================================

def lineup_optimization(menu_df, catalog_df):
    """Score actual lineups against an optimal category mix framework."""

    print("=" * 60)
    print("SECTION 6: OPTIMAL LINEUP RECOMMENDATION")
    print("=" * 60)

    menu_cats = menu_df.merge(
        catalog_df[["flavor_name", "category", "avg_rank"]], on="flavor_name", how="left"
    )

    # Determine "ideal" mix from highest-ranked category averages
    cat_pop = menu_cats.groupby("category")["rank"].mean().sort_values()
    print("\nCategory popularity ranking (lower avg rank = more popular):")
    for cat, r in cat_pop.items():
        print(f"  {cat:25s} {r:5.1f}")

    # For a 6-flavor weekly lineup, recommended mix based on popularity + variety
    # Weight categories by inverse popularity rank
    cat_weights = (1 / cat_pop).to_dict()
    total_weight = sum(cat_weights.values())
    recommended_mix = {cat: w / total_weight for cat, w in cat_weights.items()}

    # Score each week's actual lineup
    weekly_scores = []
    for week, group in menu_cats.groupby("week_start"):
        if week.year > 2026:
            continue
        actual_mix = group["category"].value_counts(normalize=True).to_dict()
        n_categories = group["category"].nunique()
        n_flavors = len(group)
        avg_rank = group["rank"].mean()

        # Diversity score: how many different categories represented
        max_possible = min(n_flavors, len(cat_pop))
        diversity_score = n_categories / max_possible

        # Popularity score: inverse of avg rank
        max_rank = group["n_flavors"].iloc[0]
        popularity_score = 1 - (avg_rank - 1) / (max_rank - 1) if max_rank > 1 else 0.5

        # Combined score
        combined = diversity_score * 0.5 + popularity_score * 0.5

        weekly_scores.append({
            "week_start": week,
            "n_flavors": n_flavors,
            "n_categories": n_categories,
            "avg_rank": round(avg_rank, 1),
            "diversity_score": round(diversity_score, 3),
            "popularity_score": round(popularity_score, 3),
            "lineup_score": round(combined, 3),
        })

    scores_df = pd.DataFrame(weekly_scores)
    scores_df["week_start"] = pd.to_datetime(scores_df["week_start"])

    print(f"\nLineup optimization scores ({len(scores_df)} weeks):")
    print(f"  Avg diversity score:   {scores_df['diversity_score'].mean():.3f}")
    print(f"  Avg popularity score:  {scores_df['popularity_score'].mean():.3f}")
    print(f"  Avg combined score:    {scores_df['lineup_score'].mean():.3f}")

    # Best and worst weeks
    best = scores_df.nlargest(3, "lineup_score")
    worst = scores_df.nsmallest(3, "lineup_score")
    print(f"\nBest-optimized weeks:")
    for _, r in best.iterrows():
        print(f"  {r['week_start'].date()}  score={r['lineup_score']:.3f}  ({r['n_categories']} categories, avg rank {r['avg_rank']})")
    print(f"\nWorst-optimized weeks:")
    for _, r in worst.iterrows():
        print(f"  {r['week_start'].date()}  score={r['lineup_score']:.3f}  ({r['n_categories']} categories, avg rank {r['avg_rank']})")

    # Trend over time
    scores_df["year"] = scores_df["week_start"].dt.year
    yearly_avg = scores_df.groupby("year")["lineup_score"].mean()
    print(f"\nAvg lineup score by year:")
    for year, score in yearly_avg.items():
        if year <= 2026:
            print(f"  {year}: {score:.3f}")

    print()
    return scores_df


# ============================================================
# SECTION 7: VISUALIZATIONS
# ============================================================

def plot_flavor_frequency(catalog_df):
    """Top 20 most-rotated flavors bar chart."""
    setup_style()
    top20 = catalog_df.nlargest(20, "times_appeared")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [CATEGORY_COLORS.get(cat, PINK) for cat in top20["category"]]
    bars = ax.barh(
        range(len(top20)), top20["times_appeared"],
        color=colors, edgecolor=DARK, height=0.7
    )

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["flavor_name"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Weekly Appearances")
    ax.set_title("Top 20 Most-Rotated Crumbl Flavors", fontsize=14, fontweight="bold", pad=12)

    for bar, val in zip(bars, top20["times_appeared"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(int(val)), va="center", fontsize=9, color=CREAM)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "flavor_frequency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/flavor_frequency.png")


def plot_category_distribution(menu_cats, yearly_pcts):
    """Category mix over time stacked area chart."""
    setup_style()

    yearly_pcts_clean = yearly_pcts[yearly_pcts.index <= 2026]
    top_cats = yearly_pcts_clean.sum().nlargest(8).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CATEGORY_COLORS.get(cat, "#999999") for cat in top_cats]
    ax.stackplot(
        yearly_pcts_clean.index,
        [yearly_pcts_clean[cat] for cat in top_cats],
        labels=top_cats,
        colors=colors,
        alpha=0.85,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of Weekly Lineup (%)")
    ax.set_title("Category Mix Over Time", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_xlim(yearly_pcts_clean.index.min(), yearly_pcts_clean.index.max())

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "category_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/category_distribution.png")


def plot_popularity_vs_frequency(catalog_df):
    """Scatter plot: avg rank vs return frequency with quadrant labels."""
    setup_style()

    qualified = catalog_df[catalog_df["times_appeared"] >= 3].copy()

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {"Star": "#4A7C59", "Underused Gem": "#FF4081",
                  "Overrotated": "#C8102E", "Niche": "#999999", "Average": "#D4A76A"}

    if "quadrant" not in qualified.columns:
        qualified["popularity_score"] = 1 / qualified["avg_rank"]
        median_pop = qualified["popularity_score"].median()
        median_app = qualified["times_appeared"].median()
        qualified["quadrant"] = "Average"
        qualified.loc[(qualified["popularity_score"] > median_pop) & (qualified["times_appeared"] < median_app), "quadrant"] = "Underused Gem"
        qualified.loc[(qualified["popularity_score"] < median_pop) & (qualified["times_appeared"] > median_app), "quadrant"] = "Overrotated"
        qualified.loc[(qualified["popularity_score"] > median_pop) & (qualified["times_appeared"] >= median_app), "quadrant"] = "Star"
        qualified.loc[(qualified["popularity_score"] < median_pop) & (qualified["times_appeared"] <= median_app), "quadrant"] = "Niche"

    for quad, color in colors_map.items():
        mask = qualified["quadrant"] == quad
        ax.scatter(
            qualified.loc[mask, "times_appeared"],
            qualified.loc[mask, "avg_rank"],
            c=color, label=quad, alpha=0.7, s=50, edgecolors=DARK, linewidth=0.5
        )

    # Label notable points
    for _, row in qualified.nlargest(5, "times_appeared").iterrows():
        ax.annotate(row["flavor_name"], (row["times_appeared"], row["avg_rank"]),
                    fontsize=7, color=CREAM, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    for _, row in qualified[qualified["quadrant"] == "Underused Gem"].nsmallest(3, "avg_rank").iterrows():
        ax.annotate(row["flavor_name"], (row["times_appeared"], row["avg_rank"]),
                    fontsize=7, color=ACCENT_PINK, alpha=0.9,
                    xytext=(5, -10), textcoords="offset points")

    ax.set_xlabel("Number of Appearances")
    ax.set_ylabel("Average Rank (lower = more popular)")
    ax.invert_yaxis()
    ax.set_title("Flavor Popularity vs. Rotation Frequency", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "popularity_vs_frequency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/popularity_vs_frequency.png")


def plot_seasonal_heatmap(menu_df, catalog_df):
    """Heatmap of category frequency by month."""
    setup_style()

    menu_cats = menu_df.merge(
        catalog_df[["flavor_name", "category"]], on="flavor_name", how="left"
    )
    menu_cats = menu_cats[menu_cats["week_start"].dt.year <= 2026]

    top_cats = menu_cats["category"].value_counts().head(10).index.tolist()
    menu_cats_top = menu_cats[menu_cats["category"].isin(top_cats)]

    pivot = menu_cats_top.groupby(
        [menu_cats_top["week_start"].dt.month, "category"]
    ).size().unstack(fill_value=0)

    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot.T, cmap="RdPu", annot=True, fmt=".0f",
        linewidths=0.5, linecolor=DARK,
        cbar_kws={"label": "% of Month's Lineup"},
        ax=ax,
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticklabels(month_labels, rotation=0)
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    ax.set_title("Category Seasonality — When Does Each Type Appear?",
                 fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "seasonal_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/seasonal_heatmap.png")


def plot_feature_importance(importance_df):
    """Feature importance from the return prediction model."""
    setup_style()
    top15 = importance_df.head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(
        range(len(top15)), top15["importance"],
        color=PINK, edgecolor=DARK, height=0.6
    )
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].str.replace("category_", "").str.replace("season_tag_", ""), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("What Drives Flavor Returns?", fontsize=14, fontweight="bold", pad=12)

    for bar, val in zip(bars, top15["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8, color=CREAM)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/feature_importance.png")


def plot_lineup_scores(scores_df):
    """Lineup optimization score over time."""
    setup_style()
    scores_clean = scores_df[scores_df["week_start"].dt.year <= 2026].copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(scores_clean["week_start"], scores_clean["lineup_score"],
               c=PINK, alpha=0.3, s=15, edgecolors="none")

    rolling = scores_clean.set_index("week_start")["lineup_score"].rolling("90D").mean()
    ax.plot(rolling.index, rolling.values, color=ACCENT_PINK, linewidth=2, label="90-day avg")

    ax.set_xlabel("Date")
    ax.set_ylabel("Lineup Score")
    ax.set_title("Weekly Lineup Optimization Score Over Time",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lineup_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/lineup_scores.png")


# ============================================================
# SECTION 8: TABLEAU EXPORTS
# ============================================================

def export_tableau_data(menu_df, catalog_df, importance_df, candidates_df, scores_df, menu_cats):
    """Export analysis-ready CSVs for Tableau."""

    print("=" * 60)
    print("SECTION 8: TABLEAU EXPORTS")
    print("=" * 60)

    # 1. Flavor summary
    catalog_df.to_csv(os.path.join(EXPORT_DIR, "flavor_summary.csv"), index=False)

    # 2. Weekly lineups with categories
    lineup_export = menu_cats[["week_start", "year", "month", "flavor_name",
                               "rank", "n_flavors", "category", "is_new"]].copy()
    lineup_export = lineup_export[lineup_export["week_start"].dt.year <= 2026]
    lineup_export.to_csv(os.path.join(EXPORT_DIR, "weekly_lineups.csv"), index=False)

    # 3. Category trends by month
    menu_cats_clean = menu_cats[menu_cats["week_start"].dt.year <= 2026]
    cat_trends = (
        menu_cats_clean.groupby([menu_cats_clean["week_start"].dt.to_period("M"), "category"])
        .size()
        .reset_index(name="count")
    )
    cat_trends.columns = ["month", "category", "count"]
    cat_trends["month"] = cat_trends["month"].astype(str)
    cat_trends.to_csv(os.path.join(EXPORT_DIR, "category_trends.csv"), index=False)

    # 4. Model predictions
    if len(candidates_df) > 0:
        pred_export = candidates_df[["flavor_name", "category", "season_tag",
                                      "times_appeared", "avg_rank", "avg_gap_weeks",
                                      "return_probability"]].copy()
        pred_export.to_csv(os.path.join(EXPORT_DIR, "model_predictions.csv"), index=False)

    # 5. Feature importance
    importance_df.to_csv(os.path.join(EXPORT_DIR, "feature_importance.csv"), index=False)

    # 6. Lineup scores
    scores_df.to_csv(os.path.join(EXPORT_DIR, "lineup_scores.csv"), index=False)

    exported = os.listdir(EXPORT_DIR)
    print(f"\nExported {len(exported)} files to {EXPORT_DIR}/:")
    for f in sorted(exported):
        size = os.path.getsize(os.path.join(EXPORT_DIR, f))
        print(f"  {f:30s} {size:>8,} bytes")
    print()


# ============================================================
# SECTION 9: EXECUTIVE SUMMARY
# ============================================================

def executive_summary(catalog_df, importance_df, scores_df, candidates_df):
    """Print executive summary of findings."""

    print("=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    n_flavors = len(catalog_df)
    one_timers = (catalog_df["times_appeared"] == 1).sum()
    one_timer_pct = one_timers / n_flavors * 100

    top_cat = catalog_df["category"].value_counts().index[0]
    top_cat_pct = catalog_df["category"].value_counts(normalize=True).iloc[0] * 100

    avg_score = scores_df["lineup_score"].mean()

    top_feature = importance_df.iloc[0]["feature"]

    print(f"""
┌──────────────────────────────────────────────────────────┐
│  CRUMBL COOKIES — MENU PERFORMANCE INSIGHTS              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  SCALE:  {n_flavors} unique flavors across 330+ weeks          │
│                                                          │
│  KEY FINDING #1: {one_timer_pct:.0f}% of flavors never return       │
│  → Over half of all flavors are one-hit wonders.         │
│    Opportunity: test flavors in smaller markets before   │
│    committing to a full national rotation slot.          │
│                                                          │
│  KEY FINDING #2: Category concentration risk             │
│  → {top_cat} dominates at {top_cat_pct:.0f}% of the catalog.     │
│    Diversifying into underrepresented categories could   │
│    reduce menu fatigue.                                  │
│                                                          │
│  KEY FINDING #3: Popularity-rotation mismatches          │
│  → Several high-ranked flavors are under-rotated while   │
│    lower-ranked flavors keep appearing. Data-driven      │
│    rotation scheduling could improve customer            │
│    satisfaction.                                         │
│                                                          │
│  KEY FINDING #4: Lineup diversity = {avg_score:.0%}          │
│  → Average weekly lineup scores {avg_score:.3f}/1.000 on the     │
│    optimization index. Room to improve category          │
│    diversity in weekly selections.                       │
│                                                          │
│  PREDICTION MODEL: {top_feature:35s}  │
│  is the strongest predictor of whether a flavor returns. │
│                                                          │
│  RECOMMENDATION: Implement a data-driven rotation        │
│  framework that balances category diversity, popularity  │
│  signals, and seasonal timing to maximize customer       │
│  engagement and reduce menu fatigue.                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CRUMBL COOKIES — MENU PERFORMANCE ANALYTICS")
    print("=" * 60)
    print()

    # Load data
    menu_df = pd.read_csv(os.path.join(DATA_DIR, "menu_history.csv"), parse_dates=["week_start", "week_end"])
    catalog_df = pd.read_csv(os.path.join(DATA_DIR, "flavor_catalog.csv"))

    # Analysis
    fpw = rotation_overview(menu_df, catalog_df)
    menu_cats, yearly_pcts = category_analysis(menu_df, catalog_df)
    qualified_df = flavor_return_analysis(menu_df, catalog_df)
    mismatch_df = popularity_vs_frequency(catalog_df)
    importance_df, model, candidates_df = build_prediction_model(menu_df, catalog_df)
    scores_df = lineup_optimization(menu_df, catalog_df)

    # Visualizations
    print("=" * 60)
    print("SECTION 7: VISUALIZATIONS")
    print("=" * 60)
    plot_flavor_frequency(catalog_df)
    plot_category_distribution(menu_cats, yearly_pcts)
    plot_popularity_vs_frequency(catalog_df)
    plot_seasonal_heatmap(menu_df, catalog_df)
    plot_feature_importance(importance_df)
    plot_lineup_scores(scores_df)
    print()

    # Exports
    export_tableau_data(menu_df, catalog_df, importance_df, candidates_df, scores_df, menu_cats)

    # Summary
    executive_summary(catalog_df, importance_df, scores_df, candidates_df)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
