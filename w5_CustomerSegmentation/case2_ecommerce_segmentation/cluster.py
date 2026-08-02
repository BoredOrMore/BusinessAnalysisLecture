#!/usr/bin/env python3
"""Select, fit, profile, and visualize ecommerce purchaser clusters in batches."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from time import monotonic

import config

MATPLOTLIB_CACHE = config.BASE_DIR.parent / ".tmp" / "matplotlib_case2"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(config.DUCKDB_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(config.DUCKDB_THREADS))

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from resources import ResourceWatchdog, configured_connection, preflight_snapshot


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
plt.style.use("default")
sns.set_theme(style="whitegrid", palette="colorblind")


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def transform_features(values: np.ndarray) -> np.ndarray:
    """Apply locked nonnegative transformations before standardization."""
    transformed = values.astype(np.float64, copy=True)
    assert np.isfinite(transformed).all(), "Model features contain non-finite values."
    assert (transformed >= 0).all(), "Model features must be nonnegative."
    for index, feature in enumerate(config.MODEL_FEATURES):
        if feature in config.LOG1P_FEATURES:
            transformed[:, index] = np.log1p(transformed[:, index])
    return transformed


def feature_batches(path: Path, batch_size: int = config.MODEL_BATCH_ROWS):
    """Yield Arrow record batches and numeric feature matrices from Parquet."""
    parquet = pq.ParquetFile(path)
    columns = ["user_id", *config.MODEL_FEATURES]
    missing = set(columns) - set(parquet.schema_arrow.names)
    assert not missing, f"Feature schema missing: {sorted(missing)}"
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        matrix = np.column_stack(
            [batch.column(feature).to_numpy(zero_copy_only=False) for feature in config.MODEL_FEATURES]
        )
        yield batch, matrix


def fit_scaler(path: Path, watchdog: ResourceWatchdog) -> tuple[StandardScaler, int]:
    """Fit scaling statistics across every customer without materializing the table."""
    scaler = StandardScaler()
    customer_count = 0
    for _, values in feature_batches(path):
        watchdog.raise_if_violated()
        transformed = transform_features(values)
        scaler.partial_fit(transformed)
        customer_count += len(values)
    assert customer_count > 0
    return scaler, customer_count


def load_model_sample(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    customer_count: int,
) -> pl.DataFrame:
    """Load a fixed-size reproducible reservoir sample for model selection."""
    sample_rows = min(config.MODEL_SAMPLE_ROWS, customer_count)
    columns = ", ".join(["user_id", *config.MODEL_FEATURES])
    arrow_table = connection.execute(
        f"""
        SELECT {columns}
        FROM read_parquet({sql_string(path)})
        USING SAMPLE reservoir({sample_rows} ROWS) REPEATABLE({config.RANDOM_SEED})
        """
    ).to_arrow_table()
    return pl.from_arrow(arrow_table)


def evaluate_candidates(
    sample_matrix: np.ndarray,
) -> tuple[pl.DataFrame, int]:
    """Select k using silhouette, stability, minimum cluster size, and compactness."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    silhouette_rows = min(config.SILHOUETTE_SAMPLE_ROWS, len(sample_matrix))
    silhouette_indexes = rng.choice(len(sample_matrix), size=silhouette_rows, replace=False)
    records: list[dict[str, object]] = []
    minimum_allowed = max(10, int(np.ceil(len(sample_matrix) * 0.005)))

    for cluster_count in config.CANDIDATE_K:
        baseline = KMeans(
            n_clusters=cluster_count,
            random_state=config.RANDOM_SEED,
            n_init=20,
        ).fit(sample_matrix)
        baseline_labels = baseline.labels_
        cluster_sizes = np.bincount(baseline_labels, minlength=cluster_count)
        stability_scores = []
        for seed in config.STABILITY_SEEDS:
            comparison = KMeans(
                n_clusters=cluster_count,
                random_state=seed,
                n_init=10,
            ).fit_predict(sample_matrix)
            stability_scores.append(adjusted_rand_score(baseline_labels, comparison))

        sample_labels = baseline.predict(sample_matrix[silhouette_indexes])
        stability = float(np.mean(stability_scores))
        minimum_size = int(cluster_sizes.min())
        records.append(
            {
                "k": cluster_count,
                "silhouette": float(
                    silhouette_score(sample_matrix[silhouette_indexes], sample_labels)
                ),
                "davies_bouldin": float(davies_bouldin_score(sample_matrix, baseline_labels)),
                "stability_ari": stability,
                "minimum_cluster_size": minimum_size,
                "eligible": minimum_size >= minimum_allowed
                and stability >= config.MIN_STABILITY_ARI,
            }
        )

    candidates = pl.DataFrame(records)
    eligible = candidates.filter(pl.col("eligible"))
    assert not eligible.is_empty(), (
        "No k candidate passed stability and size guardrails. Candidate diagnostics:\n"
        + str(candidates)
    )
    selected_k = int(
        eligible.sort(
            ["silhouette", "stability_ari", "davies_bouldin"],
            descending=[True, True, False],
        ).item(0, "k")
    )
    return candidates, selected_k


def fit_full_model(
    path: Path,
    scaler: StandardScaler,
    selected_k: int,
    watchdog: ResourceWatchdog,
) -> MiniBatchKMeans:
    """Train the selected model through bounded full-data batches."""
    model = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=config.RANDOM_SEED,
        batch_size=8192,
        n_init=3,
    )
    for _, values in feature_batches(path):
        watchdog.raise_if_violated()
        model.partial_fit(scaler.transform(transform_features(values)))
    return model


def assign_clusters(
    input_path: Path,
    output_path: Path,
    scaler: StandardScaler,
    model: MiniBatchKMeans,
    watchdog: ResourceWatchdog,
) -> int:
    """Predict every customer in batches while preserving the source feature schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        source_parquet = pq.ParquetFile(input_path)
        for source_batch in source_parquet.iter_batches(batch_size=config.MODEL_BATCH_ROWS):
            watchdog.raise_if_violated()
            values = np.column_stack(
                [
                    source_batch.column(feature).to_numpy(zero_copy_only=False)
                    for feature in config.MODEL_FEATURES
                ]
            )
            labels = model.predict(scaler.transform(transform_features(values))).astype(np.int16)
            table = pa.Table.from_batches([source_batch]).append_column("cluster", pa.array(labels))
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                    compression_level=1,
                )
            writer.write_table(table, row_group_size=config.PARQUET_ROW_GROUP_SIZE)
            rows_written += len(source_batch)
    finally:
        if writer is not None:
            writer.close()
    return rows_written


def profile_clusters(
    connection: duckdb.DuckDBPyConnection,
    clustered_path: Path,
) -> pl.DataFrame:
    """Create business-readable customer and feature summaries for each cluster."""
    averages = ",\n".join(
        f"round(avg({feature}), 4) AS {feature}_mean" for feature in config.MODEL_FEATURES
    )
    arrow_table = connection.execute(
        f"""
        SELECT
            cluster,
            count(*) AS customers,
            round(sum(revenue), 2) AS total_revenue,
            {averages}
        FROM read_parquet({sql_string(clustered_path)})
        GROUP BY cluster
        ORDER BY cluster
        """
    ).to_arrow_table()
    return pl.from_arrow(arrow_table)


def assign_business_names(profiles: pl.DataFrame) -> pl.DataFrame:
    """Name the three measured behavioral patterns without using source labels."""
    named = profiles.clone()
    if named.height != 3:
        segments = [f"Cluster {value}" for value in named["cluster"].to_list()]
        return named.insert_column(1, pl.Series("segment", segments))

    remaining = {int(value) for value in named["cluster"].to_list()}
    repeat_cluster = int(
        named.sort("purchase_sessions_mean", descending=True).item(0, "cluster")
    )
    remaining.remove(repeat_cluster)
    efficient_candidates = named.filter(pl.col("cluster").is_in(list(remaining)))
    efficient_cluster = int(
        efficient_candidates.sort("average_order_value_mean", descending=True).item(0, "cluster")
    )
    remaining.remove(efficient_cluster)
    direct_cluster = int(remaining.pop())
    names = {
        repeat_cluster: "Engaged Repeat Shoppers",
        efficient_cluster: "High-Value Efficient Buyers",
        direct_cluster: "One-Time Direct Buyers",
    }
    segments = [names[int(value)] for value in named["cluster"].to_list()]
    return named.insert_column(1, pl.Series("segment", segments))


def save_figures(candidates: pl.DataFrame, selected_k: int, profiles: pl.DataFrame) -> None:
    """Plot selection evidence and a four-panel cluster business dashboard."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, left = plt.subplots(figsize=(9, 5.5))
    right = left.twinx()
    left.plot(
        candidates["k"].to_numpy(),
        candidates["silhouette"].to_numpy(),
        marker="o",
        label="Silhouette",
    )
    right.plot(
        candidates["k"].to_numpy(),
        candidates["stability_ari"].to_numpy(),
        marker="s",
        color="#D55E00",
        label="Stability ARI",
    )
    left.axvline(selected_k, color="#333333", linestyle="--", label=f"Selected k={selected_k}")
    left.set(title="E-commerce Cluster Selection", xlabel="Number of clusters (k)", ylabel="Silhouette")
    right.set_ylabel("Mean stability ARI")
    handles_a, labels_a = left.get_legend_handles_labels()
    handles_b, labels_b = right.get_legend_handles_labels()
    left.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    figure.tight_layout()
    figure.savefig(config.FIGURES_DIR / "cluster_selection.png", dpi=220)
    plt.close(figure)

    dashboard = profiles.with_columns(
        pl.col("segment").alias("cluster_name"),
        (pl.col("total_revenue") / pl.col("total_revenue").sum()).alias("revenue_share"),
    )
    specs = [
        ("customers", "Purchasers by Segment", "Customers", lambda value: f"{value:,.0f}"),
        ("revenue_share", "Purchase Value Share by Segment", "Share", lambda value: f"{value:.1%}"),
        ("purchase_sessions_mean", "Mean Purchase Sessions", "Sessions", lambda value: f"{value:.2f}"),
        ("recency_days_mean", "Mean Purchase Recency", "Days", lambda value: f"{value:.1f}"),
    ]
    colors = sns.color_palette("colorblind", n_colors=dashboard.height)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for axis, (column, title, ylabel, formatter) in zip(axes.flat, specs):
        x_values = dashboard["cluster_name"].to_list()
        y_values = dashboard[column].to_numpy()
        bars = axis.bar(x_values, y_values, color=colors)
        axis.set(title=title, xlabel="", ylabel=ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.bar_label(bars, labels=[formatter(value) for value in y_values], padding=3)
        axis.margins(y=0.15)
    figure.suptitle("E-commerce Purchaser Segmentation Dashboard", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0.02, 0.04, 1, 0.96))
    figure.savefig(config.FIGURES_DIR / "cluster_business_dashboard.png", dpi=220)
    plt.close(figure)


def run(input_path: Path, clustered_path: Path, prefix: str) -> dict[str, object]:
    """Execute every bounded clustering pass and persist compact evidence."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    connection = configured_connection()
    watchdog = ResourceWatchdog(connection, stage="cluster purchaser features")
    try:
        with watchdog:
            scaler, customer_count = fit_scaler(input_path, watchdog)
            sample = load_model_sample(connection, input_path, customer_count)
            sample_values = sample.select(config.MODEL_FEATURES).to_numpy().astype(float)
            sample_matrix = scaler.transform(transform_features(sample_values))
            candidates, selected_k = evaluate_candidates(sample_matrix)
            watchdog.raise_if_violated()
            model = fit_full_model(input_path, scaler, selected_k, watchdog)
            rows_written = assign_clusters(input_path, clustered_path, scaler, model, watchdog)
            assert rows_written == customer_count
            profiles = assign_business_names(profile_clusters(connection, clustered_path))
    finally:
        connection.close()

    selected = candidates.filter(pl.col("k") == selected_k).row(0, named=True)
    candidate_path = config.OUTPUT_DIR / f"{prefix}_cluster_candidate_metrics.csv"
    profile_path = config.OUTPUT_DIR / f"{prefix}_cluster_profiles.csv"
    metrics_path = config.OUTPUT_DIR / f"{prefix}_cluster_run_metrics.json"
    candidates.with_columns(pl.col(pl.Float64).round(6)).write_csv(candidate_path)
    profiles.write_csv(profile_path)
    save_figures(candidates, selected_k, profiles)
    metrics: dict[str, object] = {
        "input": str(input_path),
        "clustered_output": str(clustered_path),
        "customers": customer_count,
        "sample_rows": len(sample),
        "silhouette_rows": min(config.SILHOUETTE_SAMPLE_ROWS, len(sample)),
        "selected_k": selected_k,
        "selected_silhouette": float(selected["silhouette"]),
        "selected_stability_ari": float(selected["stability_ari"]),
        "selected_minimum_cluster_size_in_sample": int(selected["minimum_cluster_size"]),
        "random_seed": config.RANDOM_SEED,
        "features": config.MODEL_FEATURES,
        "elapsed_seconds": round(monotonic() - started, 3),
        "resource_metrics": {
            "peak_rss_gb": round(watchdog.metrics.peak_rss_gb, 3),
            "minimum_free_disk_gb": round(watchdog.metrics.minimum_free_disk_gb, 3),
            "peak_temp_gb": round(watchdog.metrics.peak_temp_gb, 3),
        },
        "machine_preflight": preflight_snapshot(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.DERIVED_DIR / "smoke_customer_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DERIVED_DIR / "smoke_clustered_customers.parquet",
    )
    parser.add_argument("--prefix", default="smoke")
    args = parser.parse_args()
    LOGGER.info("Preflight: %s", json.dumps(preflight_snapshot(), sort_keys=True))
    metrics = run(args.input.resolve(), args.output.resolve(), args.prefix)
    LOGGER.info(
        "Selected k=%d with silhouette=%.3f and stability ARI=%.3f across %s purchasers",
        metrics["selected_k"],
        metrics["selected_silhouette"],
        metrics["selected_stability_ari"],
        f"{metrics['customers']:,}",
    )
    if args.prefix == "smoke":
        LOGGER.warning("Smoke clusters validate code only and must not enter the final report.")


if __name__ == "__main__":
    main()
