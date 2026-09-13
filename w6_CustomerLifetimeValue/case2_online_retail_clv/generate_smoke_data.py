"""Deterministic synthetic smoke dataset generator for fast CI and offline testing."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED, SMOKE_DATA_PATH


def generate_smoke_dataset(num_rows: int = 1500, num_customers: int = 120) -> pd.DataFrame:
    """Generate synthetic retail transactions with realistic RFM distributions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    base_date = datetime(2011, 1, 1)
    customer_ids = np.random.choice([f"CUST_{x}" for x in range(10000, 10000 + num_customers)], size=num_rows)
    # Inject 5% null customer IDs to test Rule 1
    null_idx = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
    
    quantities = np.random.choice([1, 2, 4, 6, 12, 24, -2, -5], size=num_rows, p=[0.35, 0.25, 0.15, 0.1, 0.08, 0.03, 0.02, 0.02])
    unit_prices = np.round(np.random.exponential(scale=5.0, size=num_rows) + 0.5, 2)
    # Inject a few zero prices to test Rule 3
    unit_prices[np.random.choice(num_rows, size=10, replace=False)] = 0.0

    days_offset = np.random.randint(0, 360, size=num_rows)
    invoice_dates = [base_date + timedelta(days=int(d), hours=np.random.randint(8, 18)) for d in days_offset]
    stock_codes = np.random.choice(["85123A", "71053", "84406B", "84029G", "22752", "21730"], size=num_rows)
    invoice_nos = [f"INV_{i//5 + 500000}" for i in range(num_rows)]

    cust_col = customer_ids.astype(object)
    cust_col[null_idx] = np.nan

    df = pd.DataFrame({
        "InvoiceNo": invoice_nos,
        "StockCode": stock_codes,
        "Description": "Specialty Retail Item",
        "Quantity": quantities,
        "InvoiceDate": [d.strftime("%Y-%m-%d %H:%M:%S") for d in invoice_dates],
        "UnitPrice": unit_prices,
        "CustomerID": cust_col,
        "Country": "United Kingdom",
    })

    df.to_csv(SMOKE_DATA_PATH, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic smoke data for testing.")
    parser.add_argument("--rows", type=int, default=1500, help="Number of rows (default: 1500)")
    parser.add_argument("--customers", type=int, default=120, help="Number of unique customers (default: 120)")
    args = parser.parse_args()

    df = generate_smoke_dataset(num_rows=args.rows, num_customers=args.customers)
    print(f"Generated smoke dataset with {len(df)} rows at: {SMOKE_DATA_PATH}")


if __name__ == "__main__":
    main()
