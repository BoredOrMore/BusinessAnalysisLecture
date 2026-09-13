"""Feature engineering module for Case Study 2 (Online Retail).

Transforms clean line-item transactions into customer-level RFM-T summary vectors.
"""

from __future__ import annotations

import argparse
import pandas as pd

from config import CLEAN_DATA_PATH, OUTPUTS_DIR, RFMT_SUMMARY_PATH


def build_rfmt_summary(clean_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate transactional stream into RFM-T summary grid (Slide 35, 41)."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if clean_df is None:
        if not CLEAN_DATA_PATH.exists():
            raise FileNotFoundError(f"Clean dataset missing at {CLEAN_DATA_PATH}. Run ingest.py first.")
        clean_df = pd.read_csv(CLEAN_DATA_PATH, dtype={"CustomerID": str})
        clean_df["CustomerID"] = clean_df["CustomerID"].astype(str)
        clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], format="mixed")

    # Group line items into daily customer baskets
    daily_orders = clean_df.groupby(["CustomerID", clean_df["InvoiceDate"].dt.floor("D")]).agg(
        daily_spend=("TotalSpend", "sum"),
        items_count=("Quantity", "sum"),
    ).reset_index()

    max_observation_date = daily_orders["InvoiceDate"].max()

    # Aggregate by customer
    rfmt = daily_orders.groupby("CustomerID").agg(
        first_order=("InvoiceDate", "min"),
        last_order=("InvoiceDate", "max"),
        order_count=("InvoiceDate", "count"),
        total_historical_spend=("daily_spend", "sum"),
        mean_order_spend=("daily_spend", "mean"),
    ).reset_index()

    rfmt["CustomerID"] = rfmt["CustomerID"].astype(str)
    rfmt["frequency"] = rfmt["order_count"] - 1
    rfmt["recency"] = (rfmt["last_order"] - rfmt["first_order"]).dt.days
    rfmt["T"] = (max_observation_date - rfmt["first_order"]).dt.days
    rfmt["monetary_value"] = rfmt["mean_order_spend"]

    # Invariant assertions
    assert (rfmt["frequency"] >= 0).all(), "Frequency must be non-negative"
    assert (rfmt["T"] >= rfmt["recency"]).all(), "Tenure T must be >= Recency"
    assert (rfmt["monetary_value"] > 0).all(), "Monetary value must be positive"

    rfmt.to_csv(RFMT_SUMMARY_PATH, index=False)
    return rfmt


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RFM-T features from clean transactions.")
    parser.parse_args()

    rfmt = build_rfmt_summary()
    print("=" * 80)
    print("           CASE STUDY 2: RFM-T FEATURE ENGINEERING SUMMARY            ")
    print("=" * 80)
    print(f"Total Customer Accounts:      {len(rfmt):,}")
    print(f"Repeat Buyers (freq > 0):     {(rfmt['frequency'] > 0).sum():,} ({(rfmt['frequency'] > 0).mean():.1%})")
    print(f"One-Time Buyers (freq == 0):  {(rfmt['frequency'] == 0).sum():,} ({(rfmt['frequency'] == 0).mean():.1%})")
    print(f"Average Orders / Customer:    {rfmt['order_count'].mean():.2f}")
    print(f"Average Order Spend ($):      ${rfmt['monetary_value'].mean():,.2f}")
    print(f"Average Tenure (Days):        {rfmt['T'].mean():.1f} days")
    print("=" * 80)
    print(f"RFM-T summary saved to: {RFMT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
