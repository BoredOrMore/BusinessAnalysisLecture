#!/usr/bin/env python3
"""Validate a monthly ecommerce CSV/GZIP file and convert it to typed Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import duckdb

import config
from resources import ResourceWatchdog, configured_connection, preflight_snapshot


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def sql_string(value: str | Path) -> str:
    """Quote a value for a DuckDB string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    """Hash the immutable source file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_expression(input_path: Path) -> str:
    """Build an explicit-schema DuckDB CSV reader expression."""
    columns = ", ".join(
        f"{sql_string(name)}: {sql_string(data_type)}"
        for name, data_type in config.RAW_COLUMNS.items()
    )
    return (
        f"read_csv({sql_string(input_path)}, header=true, auto_detect=false, "
        f"columns={{ {columns} }}, nullstr='')"
    )


def typed_select(input_path: Path) -> str:
    """Return the canonical typed projection used for Parquet conversion."""
    return f"""
        SELECT
            try_strptime(event_time, '%Y-%m-%d %H:%M:%S UTC') AS event_time,
            lower(event_type) AS event_type,
            product_id,
            category_id,
            category_code,
            lower(brand) AS brand,
            price,
            user_id,
            user_session
        FROM {read_csv_expression(input_path)}
    """


def quality_report(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    expected_month: str,
    input_rows: int,
) -> dict[str, object]:
    """Calculate schema, null, range, event, and duplicate evidence from Parquet."""
    source = f"read_parquet({sql_string(parquet_path)})"
    summary = connection.execute(
        f"""
        SELECT
            count(*) AS output_rows,
            min(event_time) AS min_event_time,
            max(event_time) AS max_event_time,
            count(*) FILTER (WHERE event_time IS NULL) AS invalid_event_times,
            count(*) FILTER (WHERE product_id IS NULL) AS null_product_ids,
            count(*) FILTER (WHERE category_id IS NULL) AS null_category_ids,
            count(*) FILTER (WHERE category_code IS NULL) AS null_category_codes,
            count(*) FILTER (WHERE brand IS NULL) AS null_brands,
            count(*) FILTER (WHERE price IS NULL) AS null_prices,
            count(*) FILTER (WHERE price < 0) AS negative_prices,
            count(*) FILTER (WHERE user_id IS NULL) AS null_user_ids,
            count(*) FILTER (WHERE user_session IS NULL) AS null_user_sessions,
            count(*) FILTER (
                WHERE event_type = 'purchase' AND user_session IS NULL
            ) AS purchase_events_without_session,
            count(*) FILTER (
                WHERE event_time IS NOT NULL
                  AND strftime(event_time, '%Y-%m') <> {sql_string(expected_month)}
            ) AS rows_outside_expected_month
        FROM {source}
        """
    ).fetchone()
    summary_names = [item[0] for item in connection.description]
    summary_dict = dict(zip(summary_names, summary))

    event_counts = dict(
        connection.execute(
            f"SELECT event_type, count(*) FROM {source} GROUP BY event_type ORDER BY event_type"
        ).fetchall()
    )
    column_list = ", ".join(config.RAW_COLUMNS)
    duplicate_rows = connection.execute(
        f"""
        SELECT coalesce(sum(group_count - 1), 0)
        FROM (
            SELECT {column_list}, count(*) AS group_count
            FROM {source}
            GROUP BY ALL
            HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    schema = [
        {"column": row[0], "type": row[1], "nullable": row[2]}
        for row in connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    ]

    report: dict[str, object] = {
        "expected_month": expected_month,
        "input_rows_from_copy": input_rows,
        **summary_dict,
        "exact_duplicate_rows": int(duplicate_rows),
        "event_type_counts": event_counts,
        "schema": schema,
    }
    for key in ("min_event_time", "max_event_time"):
        value = report[key]
        report[key] = value.isoformat(sep=" ") if value is not None else None
    return report


def enforce_quality(report: dict[str, object]) -> None:
    """Stop on contract violations while allowing documented optional-field nulls."""
    assert report["input_rows_from_copy"] == report["output_rows"], "Row reconciliation failed."
    assert report["output_rows"] > 0, "No rows were written."
    for key in (
        "invalid_event_times",
        "null_product_ids",
        "null_category_ids",
        "null_prices",
        "negative_prices",
        "null_user_ids",
        "purchase_events_without_session",
        "rows_outside_expected_month",
    ):
        assert report[key] == 0, f"Quality contract failed: {key}={report[key]}"
    unknown_types = set(report["event_type_counts"]) - config.ALLOWED_EVENT_TYPES
    assert not unknown_types, f"Unknown event types: {sorted(unknown_types)}"
    assert "purchase" in report["event_type_counts"], "No purchase events found."


def ingest(input_path: Path, output_path: Path, month: str) -> dict[str, object]:
    """Run guarded conversion and return its complete audit record."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if len(month) != 7 or month[4] != "-":
        raise ValueError("Month must use YYYY-MM format.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    connection = configured_connection()
    watchdog = ResourceWatchdog(connection, stage=f"ingest {input_path.name}")
    try:
        with watchdog:
            copy_result = connection.execute(
                f"""
                COPY ({typed_select(input_path)})
                TO {sql_string(output_path)} (
                    FORMAT parquet,
                    COMPRESSION zstd,
                    COMPRESSION_LEVEL 1,
                    ROW_GROUP_SIZE {config.PARQUET_ROW_GROUP_SIZE}
                )
                """
            ).fetchone()
            input_rows = int(copy_result[0])
            report = quality_report(connection, output_path, month, input_rows)
            enforce_quality(report)
    finally:
        connection.close()

    report.update(
        {
            "source_path": str(input_path),
            "source_bytes": input_path.stat().st_size,
            "source_sha256": sha256_file(input_path),
            "output_path": str(output_path),
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
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RAW_DIR / "smoke-2019-Oct.csv.gz",
    )
    parser.add_argument("--month", default="2019-10")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve()
    stem = input_path.name.removesuffix(".gz").removesuffix(".csv")
    output_path = (args.output or config.PROCESSED_DIR / f"{stem}.parquet").resolve()
    quality_path = config.OUTPUT_DIR / f"{stem}_quality_report.json"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Preflight: %s", json.dumps(preflight_snapshot(), sort_keys=True))
    LOGGER.info("Converting %s to %s", input_path, output_path)
    report = ingest(input_path, output_path, args.month)
    quality_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Validated %s rows; duplicates=%s; output=%.1f MiB",
        f"{report['output_rows']:,}",
        report["exact_duplicate_rows"],
        report["output_bytes"] / 1024**2,
    )
    LOGGER.info("Quality report: %s", quality_path)


if __name__ == "__main__":
    main()
