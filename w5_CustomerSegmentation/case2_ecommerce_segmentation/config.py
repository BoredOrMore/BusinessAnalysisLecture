#!/usr/bin/env python3
"""Locked configuration for out-of-core ecommerce customer segmentation."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"
TEMP_DIR = BASE_DIR.parent / ".tmp" / "duckdb_case2"

DATASET_SLUG = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
DATASET_URL = "https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store"
RANDOM_SEED = 42

DUCKDB_MEMORY_LIMIT = "10GB"
DUCKDB_THREADS = 4
DUCKDB_TEMP_LIMIT = "35GB"
PARQUET_ROW_GROUP_SIZE = 250_000

RSS_WARNING_GB = 12.0
RSS_ABORT_GB = 14.0
DISK_WARNING_GB = 60.0
DISK_ABORT_GB = 50.0
SYSTEM_AVAILABLE_WARNING_GB = 4.0
SYSTEM_AVAILABLE_ABORT_GB = 2.0
WATCHDOG_INTERVAL_SECONDS = 2.0

RAW_COLUMNS = {
    "event_time": "VARCHAR",
    "event_type": "VARCHAR",
    "product_id": "BIGINT",
    "category_id": "BIGINT",
    "category_code": "VARCHAR",
    "brand": "VARCHAR",
    "price": "DOUBLE",
    "user_id": "BIGINT",
    "user_session": "VARCHAR",
}
ALLOWED_EVENT_TYPES = {"view", "cart", "remove_from_cart", "purchase"}

MODEL_FEATURES = [
    "recency_days",
    "purchase_sessions",
    "revenue",
    "average_order_value",
    "views",
    "cart_events",
    "purchase_items",
    "active_days",
    "session_count",
    "category_diversity",
    "cart_to_purchase_rate",
]
LOG1P_FEATURES = set(MODEL_FEATURES) - {"cart_to_purchase_rate"}
MODEL_SAMPLE_ROWS = 100_000
SILHOUETTE_SAMPLE_ROWS = 25_000
MODEL_BATCH_ROWS = 100_000
CANDIDATE_K = range(3, 9)
STABILITY_SEEDS = (19, 43, 71)
MIN_STABILITY_ARI = 0.80
