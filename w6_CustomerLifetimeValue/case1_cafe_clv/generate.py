"""Data generation and validation script for Case Study 1 (Artisanal Cafe CLV).

Validates the baseline 10-customer sample and derives initial RFM-T vectors.
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd

from config import DATA_DIR, OUTPUTS_DIR, RAW_DATA_PATH, DERIVED_RFMT_PATH


def validate_and_prepare_data() -> pd.DataFrame:
    """Validate baseline data integrity and compute derived RFM-T features."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data file at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    assert len(df) == 10, f"Expected 10 rows in baseline dataset, found {len(df)}"
    assert (df["Frequency_Orders"] >= 1).all(), "All customers must have at least 1 order"
    assert (df["Tenure_Days"] >= df["Recency_Days"]).all(), "Tenure must be >= Recency Days"

    # Derive standard RFM-T columns according to Slide 35
    df["frequency"] = df["Frequency_Orders"] - 1
    df["recency"] = df["Tenure_Days"] - df["Recency_Days"]
    df["T"] = df["Tenure_Days"]
    df["monetary_value"] = df["Avg_Order_Value"]

    df.to_csv(DERIVED_RFMT_PATH, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline cafe dataset and derive RFM-T.")
    parser.parse_args()

    df = validate_and_prepare_data()
    print("=" * 80)
    print("CASE STUDY 1: DATA INGESTION & RFM-T DERIVATION SUCCESSFUL")
    print("=" * 80)
    print(df[["CustomerID", "frequency", "recency", "T", "monetary_value", "Historical_Margin"]].to_string(index=False))
    print(f"\nDerived RFM-T saved to: {DERIVED_RFMT_PATH}")


if __name__ == "__main__":
    main()
