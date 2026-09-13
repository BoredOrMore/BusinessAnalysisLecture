"""Configuration and assumptions for Case Study 2: Enterprise Online Retail CLV Mining.

Frames business parameters, data hygiene rules, ML hyperparameters, and path anchors.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
)
RAW_DATA_PATH = DATA_DIR / "Online_Retail.csv"
SMOKE_DATA_PATH = DATA_DIR / "Online_Retail_smoke.csv"
CLEAN_DATA_PATH = OUTPUTS_DIR / "clean_transactions.csv"
RFMT_SUMMARY_PATH = OUTPUTS_DIR / "rfmt_summary.csv"
SCORED_CUSTOMERS_PATH = OUTPUTS_DIR / "clv_scored_customers.csv"
SEGMENT_IMPACT_PATH = OUTPUTS_DIR / "segment_financial_impact.csv"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
HYGIENE_REPORT_PATH = OUTPUTS_DIR / "data_hygiene_report.json"

# Financial & Valuation Constants (Slide 39)
RANDOM_SEED = 42
GROSS_MARGIN_PCT = 0.30          # 30% gross profit margin
ANNUAL_DISCOUNT_RATE = 0.10      # 10% annual discount rate
MONTHLY_DISCOUNT_RATE = 0.0083   # 0.83% monthly discount rate
HORIZON_DAYS = 365               # 12-Month prediction window (365 days)
HORIZON_MONTHS = 12

# Model Penalizers
BGNBD_PENALIZER = 0.001
GG_PENALIZER = 0.001
