"""Data ingestion and hygiene enforcement script for Case Study 2 (Online Retail).

Downloads/verifies the raw dataset, applies RULE_01, RULE_02, and RULE_03,
and generates the data hygiene validation report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import pandas as pd

from config import (
    CLEAN_DATA_PATH,
    DATA_DIR,
    HYGIENE_REPORT_PATH,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
    RAW_DATA_URL,
    SMOKE_DATA_PATH,
)


def ensure_dataset(use_smoke: bool = False) -> pd.DataFrame:
    """Ensure raw or smoke dataset exists on disk and load it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_smoke:
        if not SMOKE_DATA_PATH.exists():
            raise FileNotFoundError(f"Smoke dataset not found at {SMOKE_DATA_PATH}. Run generate_smoke_data.py first.")
        print(f"Loading smoke dataset from: {SMOKE_DATA_PATH}")
        return pd.read_csv(SMOKE_DATA_PATH)

    if not RAW_DATA_PATH.exists():
        print(f"Raw dataset missing at {RAW_DATA_PATH}. Downloading from validated URL...")
        cmd = ["curl", "-L", "-o", str(RAW_DATA_PATH), RAW_DATA_URL]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to download dataset: {res.stderr}")
        print(f"Downloaded raw dataset to {RAW_DATA_PATH} ({RAW_DATA_PATH.stat().st_size / (1024*1024):.1f} MB)")

    return pd.read_csv(RAW_DATA_PATH, encoding="latin-1")


def apply_data_hygiene(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply standard Data Hygiene Protocol (Slides 33-34)."""
    total_raw_rows = len(raw_df)

    # RULE_01: CustomerID IS NOT NULL
    r1_mask = raw_df["CustomerID"].notna()
    r1_dropped = int((~r1_mask).sum())
    r1_df = raw_df[r1_mask].copy()

    # RULE_02: Quantity > 0
    r2_mask = r1_df["Quantity"] > 0
    r2_dropped = int((~r2_mask).sum())
    r2_df = r1_df[r2_mask].copy()

    # RULE_03: UnitPrice > 0.00
    r3_mask = r2_df["UnitPrice"] > 0.0
    r3_dropped = int((~r3_mask).sum())
    clean_df = r2_df[r3_mask].copy()

    # Standardize data types: CustomerID as string
    clean_df["CustomerID"] = clean_df["CustomerID"].apply(
        lambda x: str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x)
    )
    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], format="mixed")
    clean_df["TotalSpend"] = clean_df["Quantity"] * clean_df["UnitPrice"]

    hygiene_report = {
        "raw_row_count": total_raw_rows,
        "rule_01_null_customer_dropped": r1_dropped,
        "rule_02_negative_qty_dropped": r2_dropped,
        "rule_03_zero_price_dropped": r3_dropped,
        "clean_row_count": len(clean_df),
        "data_retention_pct": float((len(clean_df) / total_raw_rows) * 100.0),
        "unique_customers": int(clean_df["CustomerID"].nunique()),
        "date_min": str(clean_df["InvoiceDate"].min()),
        "date_max": str(clean_df["InvoiceDate"].max()),
        "total_revenue": float(clean_df["TotalSpend"].sum()),
    }

    with open(HYGIENE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(hygiene_report, f, indent=2)

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    return clean_df, hygiene_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and clean Online Retail dataset.")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic smoke dataset for fast testing")
    args = parser.parse_args()

    raw_df = ensure_dataset(use_smoke=args.smoke)
    clean_df, report = apply_data_hygiene(raw_df)

    print("=" * 80)
    print("        CASE STUDY 2: DATA INGESTION & HYGIENE EXECUTION REPORT        ")
    print("=" * 80)
    print(f"Raw Input Rows:               {report['raw_row_count']:,}")
    print(f"RULE_01 (Null CustomerID):    -{report['rule_01_null_customer_dropped']:,}")
    print(f"RULE_02 (Quantity <= 0):      -{report['rule_02_negative_qty_dropped']:,}")
    print(f"RULE_03 (UnitPrice <= 0):     -{report['rule_03_zero_price_dropped']:,}")
    print(f"Clean Transaction Rows:       {report['clean_row_count']:,} ({report['data_retention_pct']:.1f}%)")
    print(f"Unique Verified Customers:    {report['unique_customers']:,}")
    print(f"Date Range:                   {report['date_min']} to {report['date_max']}")
    print(f"Total Gross Transaction Base: ${report['total_revenue']:,.2f}")
    print("=" * 80)
    print(f"Cleaned dataset saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
