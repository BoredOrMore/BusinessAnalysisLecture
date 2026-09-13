"""Configuration and business constants for Case Study 1: Artisanal Cafe CLV.

Frames the business question, financial valuation parameters, and path anchors.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_PATH = DATA_DIR / "CLV_Retail_Transaction_example.csv"
DERIVED_RFMT_PATH = OUTPUTS_DIR / "derived_rfmt_matrix.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "clv_model_predictions.csv"
TOP3_PROFILES_PATH = OUTPUTS_DIR / "top3_customer_profiles.csv"
METRICS_PATH = OUTPUTS_DIR / "case1_metrics.json"

# Financial & Modeling Assumptions
RANDOM_SEED = 42
GROSS_MARGIN_PCT = 0.30          # 30% gross margin from Slide 39
ANNUAL_DISCOUNT_RATE = 0.10      # 10% annual discount rate from Slide 39
MONTHLY_DISCOUNT_RATE = 0.0083   # 0.83% monthly discount rate (10% / 12)
PREDICTION_DAYS = 365            # 12-Month horizon (365 days)
PREDICTION_MONTHS = 12           # 12 Months

# Fitted Model Parameters from Slide 37
BGNBD_R = 0.55
BGNBD_ALPHA = 10.8
BGNBD_A = 0.78
BGNBD_B = 2.45
