#!/usr/bin/env python3
"""Locked configuration for the synthetic Cafe RFM case study."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR / "outputs"

RANDOM_SEED = 42
CUSTOMER_PREFIX = "CUST"
OBSERVATION_START = "2025-01-01"
GENERATOR_REFERENCE_DATE = "2026-01-01"

MENU_PRICES = {
    "Drip Coffee": 2.50,
    "Americano": 3.00,
    "Tea": 3.00,
    "Muffin": 3.50,
    "Croissant": 4.00,
    "Latte": 4.50,
    "Cake Slice": 5.50,
    "Breakfast Sandwich": 6.50,
    "Salad": 8.50,
    "Pasta": 11.00,
}

# Each archetype controls visit frequency, last-visit recency, basket size, and menu mix.
# Counts sum to exactly 150 customers. Weights follow MENU_PRICES insertion order.
ARCHETYPES = {
    "VIP Daily Drinkers": {
        "customer_count": 38,
        "visit_range": (48, 72),
        "last_recency_days": (1, 14),
        "basket_size": (3, 5),
        "item_weights": (0.10, 0.08, 0.04, 0.08, 0.09, 0.16, 0.10, 0.17, 0.09, 0.09),
    },
    "Budget Students": {
        "customer_count": 40,
        "visit_range": (24, 44),
        "last_recency_days": (1, 21),
        "basket_size": (1, 2),
        "item_weights": (0.34, 0.18, 0.17, 0.17, 0.07, 0.03, 0.02, 0.01, 0.005, 0.005),
    },
    "Occasional Treaters": {
        "customer_count": 36,
        "visit_range": (7, 15),
        "last_recency_days": (35, 110),
        "basket_size": (2, 4),
        "item_weights": (0.03, 0.03, 0.04, 0.04, 0.08, 0.15, 0.18, 0.16, 0.14, 0.15),
    },
    "Churned Customers": {
        "customer_count": 36,
        "visit_range": (3, 10),
        "last_recency_days": (190, 330),
        "basket_size": (1, 2),
        "item_weights": (0.20, 0.12, 0.12, 0.13, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05),
    },
}

EXPECTED_CUSTOMERS = sum(spec["customer_count"] for spec in ARCHETYPES.values())
CANDIDATE_K = range(2, 9)
MIN_CLUSTER_CUSTOMERS = 8
MIN_STABILITY_ARI = 0.80
STABILITY_SEEDS = (7, 19, 31, 43, 59)
