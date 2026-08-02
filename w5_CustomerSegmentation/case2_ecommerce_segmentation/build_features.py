#!/usr/bin/env python3
"""Aggregate monthly ecommerce events into one purchaser-level behavioral feature row."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb

import config
from resources import ResourceWatchdog, configured_connection, preflight_snapshot


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def parquet_source(paths: list[Path]) -> str:
    """Build a DuckDB read_parquet expression from validated paths."""
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed inputs: {missing}")
    path_list = ", ".join(sql_string(path.resolve()) for path in paths)
    return f"read_parquet([{path_list}], union_by_name=false)"


def feature_query(paths: list[Path]) -> str:
    """Build purchase RFM plus browsing, session, and funnel features."""
    source = parquet_source(paths)
    return f"""
        WITH events AS (
            SELECT * FROM {source}
        ),
        observation AS (
            SELECT date_trunc('day', max(event_time)) + INTERVAL 1 DAY AS reference_date
            FROM events
        ),
        user_events AS (
            SELECT
                user_id,
                count(*) FILTER (WHERE event_type = 'view') AS views,
                count(*) FILTER (WHERE event_type = 'cart') AS cart_events,
                count(*) FILTER (WHERE event_type = 'remove_from_cart') AS remove_events,
                count(DISTINCT CAST(event_time AS DATE)) AS active_days,
                count(DISTINCT user_session) AS session_count,
                count(DISTINCT category_code) FILTER (
                    WHERE category_code IS NOT NULL
                ) AS category_diversity
            FROM events
            GROUP BY user_id
        ),
        purchases AS (
            SELECT
                user_id,
                max(event_time) AS last_purchase,
                count(*) AS purchase_items,
                count(DISTINCT user_session) AS purchase_sessions,
                sum(price) AS revenue
            FROM events
            WHERE event_type = 'purchase'
            GROUP BY user_id
        ),
        session_funnel AS (
            SELECT
                user_id,
                user_session,
                bool_or(event_type = 'cart') AS had_cart,
                bool_or(event_type = 'purchase') AS had_purchase
            FROM events
            WHERE user_session IS NOT NULL
            GROUP BY user_id, user_session
        ),
        user_funnel AS (
            SELECT
                user_id,
                count(*) FILTER (WHERE had_cart) AS cart_sessions,
                count(*) FILTER (WHERE had_cart AND had_purchase) AS converted_cart_sessions
            FROM session_funnel
            GROUP BY user_id
        )
        SELECT
            p.user_id,
            date_diff(
                'day', CAST(p.last_purchase AS DATE), CAST(o.reference_date AS DATE)
            )::INTEGER AS recency_days,
            p.purchase_sessions::BIGINT AS purchase_sessions,
            round(p.revenue, 2) AS revenue,
            round(p.revenue / p.purchase_sessions, 4) AS average_order_value,
            u.views::BIGINT AS views,
            u.cart_events::BIGINT AS cart_events,
            u.remove_events::BIGINT AS remove_events,
            p.purchase_items::BIGINT AS purchase_items,
            u.active_days::BIGINT AS active_days,
            u.session_count::BIGINT AS session_count,
            u.category_diversity::BIGINT AS category_diversity,
            coalesce(f.cart_sessions, 0)::BIGINT AS cart_sessions,
            coalesce(f.converted_cart_sessions, 0)::BIGINT AS converted_cart_sessions,
            CASE
                WHEN coalesce(f.cart_sessions, 0) = 0 THEN 0.0
                ELSE f.converted_cart_sessions::DOUBLE / f.cart_sessions
            END AS cart_to_purchase_rate,
            p.last_purchase,
            o.reference_date
        FROM purchases p
        JOIN user_events u USING (user_id)
        LEFT JOIN user_funnel f USING (user_id)
        CROSS JOIN observation o
        WHERE p.purchase_sessions > 0
    """


def validate_features(
    connection: duckdb.DuckDBPyConnection,
    inputs: list[Path],
    output_path: Path,
    copy_rows: int,
) -> dict[str, object]:
    """Reconcile purchase revenue and enforce feature-domain constraints."""
    source = f"read_parquet({sql_string(output_path)})"
    feature_names = [
        "recency_days",
        "purchase_sessions",
        "revenue",
        "average_order_value",
        "views",
        "cart_events",
        "remove_events",
        "purchase_items",
        "active_days",
        "session_count",
        "category_diversity",
        "cart_sessions",
        "converted_cart_sessions",
        "cart_to_purchase_rate",
    ]
    null_expression = " + ".join(
        f"count(*) FILTER (WHERE {name} IS NULL)" for name in feature_names
    )
    negative_expression = " + ".join(
        f"count(*) FILTER (WHERE {name} < 0)" for name in feature_names
    )
    summary = connection.execute(
        f"""
        SELECT
            count(*) AS customers,
            count(DISTINCT user_id) AS unique_customers,
            sum(revenue) AS feature_revenue,
            min(reference_date) AS reference_date_min,
            max(reference_date) AS reference_date_max,
            ({null_expression}) AS feature_null_cells,
            ({negative_expression}) AS negative_feature_cells,
            count(*) FILTER (WHERE cart_to_purchase_rate > 1) AS rates_above_one,
            count(*) FILTER (WHERE purchase_sessions > session_count) AS invalid_session_counts
        FROM {source}
        """
    ).fetchone()
    names = [item[0] for item in connection.description]
    report = dict(zip(names, summary))

    event_source = parquet_source(inputs)
    event_revenue = connection.execute(
        f"SELECT sum(price) FROM {event_source} WHERE event_type = 'purchase'"
    ).fetchone()[0]
    report["copy_rows"] = copy_rows
    report["event_purchase_revenue"] = float(event_revenue or 0)
    report["feature_revenue"] = float(report["feature_revenue"] or 0)
    for key in ("reference_date_min", "reference_date_max"):
        value = report[key]
        report[key] = value.isoformat(sep=" ") if value is not None else None

    assert report["customers"] == copy_rows > 0
    assert report["customers"] == report["unique_customers"]
    assert report["feature_null_cells"] == 0
    assert report["negative_feature_cells"] == 0
    assert report["rates_above_one"] == 0
    assert report["invalid_session_counts"] == 0
    assert report["reference_date_min"] == report["reference_date_max"]
    revenue_difference = abs(report["event_purchase_revenue"] - report["feature_revenue"])
    assert revenue_difference <= 0.01, f"Purchase revenue mismatch: {revenue_difference:.4f}"
    return report


def build_features(inputs: list[Path], output_path: Path) -> dict[str, object]:
    """Execute guarded aggregation and return its audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = configured_connection()
    watchdog = ResourceWatchdog(connection, stage="build purchaser features")
    try:
        with watchdog:
            copy_rows = int(
                connection.execute(
                    f"""
                    COPY ({feature_query(inputs)})
                    TO {sql_string(output_path)} (
                        FORMAT parquet,
                        COMPRESSION zstd,
                        COMPRESSION_LEVEL 1,
                        ROW_GROUP_SIZE {config.PARQUET_ROW_GROUP_SIZE}
                    )
                    """
                ).fetchone()[0]
            )
            report = validate_features(connection, inputs, output_path, copy_rows)
    finally:
        connection.close()

    report.update(
        {
            "inputs": [str(path.resolve()) for path in inputs],
            "output": str(output_path.resolve()),
            "output_bytes": output_path.stat().st_size,
            "resource_metrics": {
                "elapsed_seconds": round(watchdog.metrics.elapsed_seconds, 3),
                "peak_rss_gb": round(watchdog.metrics.peak_rss_gb, 3),
                "minimum_free_disk_gb": round(watchdog.metrics.minimum_free_disk_gb, 3),
                "peak_temp_gb": round(watchdog.metrics.peak_temp_gb, 3),
            },
            "machine_preflight": preflight_snapshot(),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=config.DERIVED_DIR / "smoke_customer_features.parquet")
    args = parser.parse_args()
    inputs = args.input or [config.PROCESSED_DIR / "smoke-2019-Oct.parquet"]
    report_path = config.OUTPUT_DIR / f"{args.output.stem}_quality_report.json"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preflight: %s", json.dumps(preflight_snapshot(), sort_keys=True))
    LOGGER.info("Building purchaser features from %d monthly file(s)", len(inputs))
    report = build_features([path.resolve() for path in inputs], args.output.resolve())
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Validated %s purchaser rows and %s total purchase value",
        f"{report['customers']:,}",
        f"{report['feature_revenue']:,.2f}",
    )
    LOGGER.info("Feature report: %s", report_path)


if __name__ == "__main__":
    main()
