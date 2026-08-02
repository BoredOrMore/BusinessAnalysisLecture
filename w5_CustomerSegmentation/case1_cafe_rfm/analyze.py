#!/usr/bin/env python3
"""
Calculate and evaluate RFM customer segments for the synthetic cafe case study.

Business context: the cafe manager needs reproducible customer segments for
retention, offer design, and menu strategy. Source archetypes remain withheld
until after the clustering model is selected and fitted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import config

# Keep Matplotlib's cache inside the writable, gitignored project workspace.
MATPLOTLIB_CACHE = config.BASE_DIR.parent / ".tmp" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

TRANSACTIONS_PATH = config.DATA_DIR / "cafe_transactions_example.csv"
GROUND_TRUTH_PATH = config.DATA_DIR / "cafe_customer_archetypes_validation.csv"
RFM_PATH = config.OUTPUT_DIR / "customer_rfm_segments.csv"
CANDIDATES_PATH = config.OUTPUT_DIR / "cluster_candidate_metrics.csv"
PROFILES_PATH = config.OUTPUT_DIR / "segment_profiles.csv"
VALIDATION_PATH = config.OUTPUT_DIR / "archetype_segment_crosstab.csv"
RUN_METRICS_PATH = config.OUTPUT_DIR / "run_metrics.json"

MODEL_FEATURES = ["recency_days", "frequency", "monetary"]

plt.style.use("default")
sns.set_theme(style="whitegrid", palette="deep")


def load_and_validate_transactions(path: Path) -> pd.DataFrame:
    """Load the visit log and fail loudly on schema or value violations."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run generate.py first.")

    transactions = pd.read_csv(path, parse_dates=["visit_datetime"])
    expected_columns = [
        "transaction_id",
        "customer_id",
        "visit_datetime",
        "items",
        "total_spend",
    ]
    assert list(transactions.columns) == expected_columns, "Transaction schema changed."
    assert len(transactions) > 0, "Transaction file is empty."
    assert not transactions.isna().any().any(), "Transaction data contains nulls."
    assert transactions["transaction_id"].is_unique, "Transaction IDs are not unique."
    assert transactions["customer_id"].nunique() == config.EXPECTED_CUSTOMERS
    assert transactions["total_spend"].gt(0).all(), "Spend must be positive."
    assert transactions["visit_datetime"].between(
        pd.Timestamp(config.OBSERVATION_START),
        pd.Timestamp(config.GENERATOR_REFERENCE_DATE),
        inclusive="left",
    ).all()

    def calculate_basket_spend(basket: str) -> float:
        items = basket.split(" | ")
        unknown_items = set(items) - set(config.MENU_PRICES)
        assert not unknown_items, f"Unknown menu items: {sorted(unknown_items)}"
        return sum(config.MENU_PRICES[item] for item in items)

    expected_spend = transactions["items"].map(calculate_basket_spend)
    assert np.allclose(expected_spend, transactions["total_spend"]), "Basket prices disagree."
    return transactions


def calculate_rfm(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create one customer row using the locked RFM definitions."""
    reference_date = transactions["visit_datetime"].max().normalize() + pd.Timedelta(days=1)
    rfm = (
        transactions.groupby("customer_id", as_index=False)
        .agg(
            last_visit=("visit_datetime", "max"),
            frequency=("transaction_id", "nunique"),
            monetary=("total_spend", "sum"),
        )
        .sort_values("customer_id", ignore_index=True)
    )
    rfm["recency_days"] = (reference_date - rfm["last_visit"].dt.normalize()).dt.days
    rfm["average_transaction_value"] = rfm["monetary"] / rfm["frequency"]
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm["average_transaction_value"] = rfm["average_transaction_value"].round(2)

    assert len(rfm) == config.EXPECTED_CUSTOMERS
    assert rfm["recency_days"].ge(1).all()
    assert rfm["frequency"].gt(0).all()
    assert rfm["monetary"].gt(0).all()
    assert np.isclose(rfm["monetary"].sum(), transactions["total_spend"].sum())
    return rfm, reference_date


def prepare_model_matrix(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Reduce RFM skew with log1p, then standardize all three dimensions."""
    transformed = np.log1p(rfm[MODEL_FEATURES].to_numpy(dtype=float))
    scaler = StandardScaler()
    return scaler.fit_transform(transformed), scaler


def evaluate_candidates(model_matrix: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Evaluate k by separation, compactness, cluster size, and seed stability."""
    records: list[dict[str, float | int | bool]] = []

    for cluster_count in config.CANDIDATE_K:
        baseline = KMeans(
            n_clusters=cluster_count,
            random_state=config.RANDOM_SEED,
            n_init=30,
        ).fit(model_matrix)
        cluster_sizes = np.bincount(baseline.labels_, minlength=cluster_count)
        stability_scores = []
        for seed in config.STABILITY_SEEDS:
            comparison_labels = KMeans(
                n_clusters=cluster_count,
                random_state=seed,
                n_init=20,
            ).fit_predict(model_matrix)
            stability_scores.append(adjusted_rand_score(baseline.labels_, comparison_labels))

        stability = float(np.mean(stability_scores))
        minimum_size = int(cluster_sizes.min())
        records.append(
            {
                "k": cluster_count,
                "silhouette": float(silhouette_score(model_matrix, baseline.labels_)),
                "calinski_harabasz": float(
                    calinski_harabasz_score(model_matrix, baseline.labels_)
                ),
                "davies_bouldin": float(
                    davies_bouldin_score(model_matrix, baseline.labels_)
                ),
                "stability_ari": stability,
                "minimum_cluster_size": minimum_size,
                "eligible": minimum_size >= config.MIN_CLUSTER_CUSTOMERS
                and stability >= config.MIN_STABILITY_ARI,
            }
        )

    candidates = pd.DataFrame(records)
    eligible = candidates[candidates["eligible"]]
    assert not eligible.empty, "No candidate passed cluster-size and stability guardrails."
    selected_k = int(
        eligible.sort_values(
            ["silhouette", "stability_ari", "davies_bouldin"],
            ascending=[False, False, True],
        ).iloc[0]["k"]
    )
    return candidates, selected_k


def assign_business_labels(rfm: pd.DataFrame, selected_k: int) -> dict[int, str]:
    """Name four clusters from their measured profiles, without reading ground truth."""
    if selected_k != 4:
        return {cluster: f"Cluster {cluster + 1}" for cluster in sorted(rfm["cluster"].unique())}

    profiles = rfm.groupby("cluster").agg(
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        average_transaction_value=("average_transaction_value", "mean"),
    )
    remaining = set(profiles.index)
    churned = int(profiles.loc[list(remaining), "recency_days"].idxmax())
    remaining.remove(churned)

    standardized = (profiles - profiles.mean()) / profiles.std(ddof=0)
    vip_score = standardized["frequency"] + standardized["monetary"]
    vip = int(vip_score.loc[list(remaining)].idxmax())
    remaining.remove(vip)

    budget = int(profiles.loc[list(remaining), "average_transaction_value"].idxmin())
    remaining.remove(budget)
    occasional = int(remaining.pop())

    return {
        churned: "Churned Customers",
        vip: "VIP Daily Drinkers",
        budget: "Budget Students",
        occasional: "Occasional Treaters",
    }


def fit_final_model(
    rfm: pd.DataFrame, model_matrix: np.ndarray, selected_k: int
) -> tuple[pd.DataFrame, KMeans]:
    """Fit the selected model and apply data-derived business labels."""
    model = KMeans(
        n_clusters=selected_k,
        random_state=config.RANDOM_SEED,
        n_init=50,
    ).fit(model_matrix)
    result = rfm.copy()
    result["cluster"] = model.labels_
    label_map = assign_business_labels(result, selected_k)
    result["segment"] = result["cluster"].map(label_map)
    assert result["segment"].notna().all()
    return result, model


def validate_against_withheld_labels(
    segmented_rfm: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float]:
    """Open simulation ground truth only after fitting and calculate recovery metrics."""
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Missing withheld labels: {GROUND_TRUTH_PATH}")
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    assert list(ground_truth.columns) == ["customer_id", "archetype"]
    assert ground_truth["customer_id"].is_unique

    validation = segmented_rfm.merge(ground_truth, on="customer_id", validate="one_to_one")
    ari = float(adjusted_rand_score(validation["archetype"], validation["cluster"]))
    business_label_accuracy = float((validation["segment"] == validation["archetype"]).mean())

    contingency = pd.crosstab(validation["archetype"], validation["cluster"])
    row_indexes, column_indexes = linear_sum_assignment(-contingency.to_numpy())
    best_match_accuracy = float(
        contingency.to_numpy()[row_indexes, column_indexes].sum() / len(validation)
    )
    crosstab = pd.crosstab(
        validation["archetype"], validation["segment"], margins=True, margins_name="Total"
    )
    return crosstab, ari, best_match_accuracy, business_label_accuracy


def build_profiles(segmented_rfm: pd.DataFrame) -> pd.DataFrame:
    """Summarize segment scale and RFM behavior for business interpretation."""
    profiles = (
        segmented_rfm.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            recency_days_mean=("recency_days", "mean"),
            frequency_mean=("frequency", "mean"),
            monetary_mean=("monetary", "mean"),
            average_transaction_value_mean=("average_transaction_value", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .sort_values("monetary_mean", ascending=False, ignore_index=True)
    )
    numeric_columns = profiles.select_dtypes(include="number").columns
    profiles[numeric_columns] = profiles[numeric_columns].round(2)
    return profiles


def save_figures(
    candidates: pd.DataFrame,
    selected_k: int,
    segmented_rfm: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    """Save cluster-selection, customer-map, and standardized-profile figures."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    segment_order = [
        "VIP Daily Drinkers",
        "Budget Students",
        "Occasional Treaters",
        "Churned Customers",
    ]
    present_segments = set(profiles["segment"])
    segment_order = [segment for segment in segment_order if segment in present_segments]
    segment_order.extend(
        segment for segment in profiles["segment"] if segment not in segment_order
    )
    segment_colors = dict(
        zip(segment_order, sns.color_palette("colorblind", n_colors=len(segment_order)))
    )

    dashboard = profiles.set_index("segment").loc[segment_order].reset_index()
    dashboard["revenue_share"] = dashboard["total_revenue"] / dashboard["total_revenue"].sum()
    dashboard["short_segment"] = dashboard["segment"].replace(
        {
            "VIP Daily Drinkers": "VIP",
            "Budget Students": "Budget",
            "Occasional Treaters": "Occasional",
            "Churned Customers": "Churned",
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    dashboard_specs = [
        ("customers", "Customers by Segment", "Customers", lambda value: f"{value:.0f}"),
        (
            "total_revenue",
            "Revenue Contribution by Segment",
            "Revenue ($)",
            lambda value: f"${value / 1000:.1f}K",
        ),
        (
            "frequency_mean",
            "Average Annual Visit Frequency",
            "Visits per customer",
            lambda value: f"{value:.1f}",
        ),
        (
            "recency_days_mean",
            "Average Days Since Last Visit",
            "Days",
            lambda value: f"{value:.1f}",
        ),
    ]
    dashboard_palette = [segment_colors[segment] for segment in dashboard["segment"]]
    for axis, (column, title, ylabel, formatter) in zip(axes.flat, dashboard_specs):
        bars = axis.bar(
            dashboard["short_segment"],
            dashboard[column],
            color=dashboard_palette,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set(title=title, xlabel="", ylabel=ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.bar_label(bars, labels=[formatter(value) for value in dashboard[column]], padding=3)
        axis.margins(y=0.14)
    figure.suptitle("Cafe RFM Segment Business Dashboard", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(config.FIGURES_DIR / "segment_business_dashboard.png", dpi=220)
    plt.close(figure)

    figure, left_axis = plt.subplots(figsize=(9, 5.5))
    right_axis = left_axis.twinx()
    left_axis.plot(candidates["k"], candidates["silhouette"], marker="o", label="Silhouette")
    right_axis.plot(
        candidates["k"],
        candidates["stability_ari"],
        marker="s",
        color="#D55E00",
        label="Stability ARI",
    )
    left_axis.axvline(selected_k, color="#333333", linestyle="--", label=f"Selected k={selected_k}")
    left_axis.set(title="Cafe RFM Cluster Selection", xlabel="Number of clusters (k)", ylabel="Silhouette")
    right_axis.set_ylabel("Mean stability ARI")
    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "cluster_selection.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6.5))
    sns.scatterplot(
        data=segmented_rfm,
        x="frequency",
        y="monetary",
        hue="segment",
        hue_order=segment_order,
        palette=segment_colors,
        size="recency_days",
        sizes=(35, 180),
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )
    axis.set(
        title="Cafe Customers by Frequency and Monetary Value",
        xlabel="Visits during observation year",
        ylabel="Annual spend ($)",
    )
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "rfm_segments.png", dpi=220)
    plt.close(figure)

    profile_values = profiles.set_index("segment")[
        [
            "recency_days_mean",
            "frequency_mean",
            "monetary_mean",
            "average_transaction_value_mean",
        ]
    ]
    standardized = (profile_values - profile_values.mean()) / profile_values.std(ddof=0)
    standardized.columns = ["Recency", "Frequency", "Monetary", "Avg transaction"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(
        standardized,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Standard deviations from segment mean"},
        ax=axis,
    )
    axis.set(title="Standardized Cafe Segment Profiles", xlabel="RFM measure", ylabel="Segment")
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "segment_profiles.png", dpi=220)
    plt.close(figure)


def write_outputs(
    segmented_rfm: pd.DataFrame,
    candidates: pd.DataFrame,
    profiles: pd.DataFrame,
    crosstab: pd.DataFrame,
    metrics: dict[str, object],
) -> None:
    """Persist compact, reviewable evidence from the successful analysis run."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segmented_rfm.to_csv(RFM_PATH, index=False)
    candidates.round(6).to_csv(CANDIDATES_PATH, index=False)
    profiles.to_csv(PROFILES_PATH, index=False)
    crosstab.to_csv(VALIDATION_PATH)
    RUN_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def print_report(profiles: pd.DataFrame, metrics: dict[str, object]) -> None:
    """Print the complete measured summary used to write the executive report."""
    print("\n" + "=" * 92)
    print("CASE STUDY 1: CAFE RFM SEGMENTATION - MEASURED RESULTS")
    print("=" * 92)
    print(f"Customers:              {metrics['customers']:,}")
    print(f"Transactions:           {metrics['transactions']:,}")
    print(f"Reference date:         {metrics['reference_date']}")
    print(f"Selected clusters:      {metrics['selected_k']}")
    print(f"Selected silhouette:    {metrics['selected_silhouette']:.3f}")
    print(f"Selected stability ARI: {metrics['selected_stability_ari']:.3f}")
    print(f"Withheld-label ARI:     {metrics['withheld_label_ari']:.3f}")
    print(f"Best-match accuracy:    {metrics['best_match_accuracy']:.1%}")
    print(f"Business-label accuracy:{metrics['business_label_accuracy']:>6.1%}")
    print("-" * 92)
    display = profiles.copy()
    display["monetary_mean"] = display["monetary_mean"].map(lambda value: f"${value:,.2f}")
    display["average_transaction_value_mean"] = display[
        "average_transaction_value_mean"
    ].map(lambda value: f"${value:,.2f}")
    display["total_revenue"] = display["total_revenue"].map(lambda value: f"${value:,.2f}")
    print(display.to_string(index=False))
    print("=" * 92 + "\n")


def main() -> None:
    LOGGER.info("Loading and validating %s", TRANSACTIONS_PATH)
    transactions = load_and_validate_transactions(TRANSACTIONS_PATH)
    rfm, reference_date = calculate_rfm(transactions)
    model_matrix, _ = prepare_model_matrix(rfm)
    candidates, selected_k = evaluate_candidates(model_matrix)
    segmented_rfm, _ = fit_final_model(rfm, model_matrix, selected_k)
    profiles = build_profiles(segmented_rfm)
    crosstab, ari, best_match_accuracy, business_label_accuracy = (
        validate_against_withheld_labels(segmented_rfm)
    )

    selected_row = candidates.loc[candidates["k"] == selected_k].iloc[0]
    metrics: dict[str, object] = {
        "random_seed": config.RANDOM_SEED,
        "customers": int(len(segmented_rfm)),
        "transactions": int(len(transactions)),
        "reference_date": reference_date.date().isoformat(),
        "rfm_definitions": {
            "recency": "days from reference date to latest visit date",
            "frequency": "distinct transaction IDs",
            "monetary": "sum of transaction total_spend",
        },
        "candidate_k": list(config.CANDIDATE_K),
        "selected_k": selected_k,
        "selected_silhouette": float(selected_row["silhouette"]),
        "selected_stability_ari": float(selected_row["stability_ari"]),
        "selected_minimum_cluster_size": int(selected_row["minimum_cluster_size"]),
        "withheld_label_ari": ari,
        "best_match_accuracy": best_match_accuracy,
        "business_label_accuracy": business_label_accuracy,
        "total_revenue": round(float(transactions["total_spend"].sum()), 2),
    }
    write_outputs(segmented_rfm, candidates, profiles, crosstab, metrics)
    save_figures(candidates, selected_k, segmented_rfm, profiles)
    print_report(profiles, metrics)
    LOGGER.info("Analysis outputs: %s", config.OUTPUT_DIR)
    LOGGER.info("Figures: %s", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
