#!/usr/bin/env python3
"""
config.py

Top-level configuration constants for Mock Cafe Market Basket Data Generation & Mining.
Adheres strictly to decisions locked in plan.md.
"""

# Decision #6: Random seed for 100% reproducibility across all operations
RANDOM_SEED = 42

# Decision #1: Item catalog (Keep 5 core items)
CATALOG = ["Coffee", "Bread", "Butter", "Tea", "Jam"]

# Decision #2 & #3: Customer archetypes and target association probabilities
# Coffee-crowd (45%): strong Coffee<->Butter bond, repels Jam/Tea
# Tea-crowd (35%): strong Tea<->Jam bond, repels Coffee/Butter
# Mixed (20%): moderate across the board to add realistic background noise
ARCHETYPES = {
    "Coffee-crowd": {
        "weight": 0.45,
        "probs": {
            "Coffee": 0.95,
            "Butter": 0.90,
            "Bread": 0.60,
            "Jam": 0.05,
            "Tea": 0.02
        }
    },
    "Tea-crowd": {
        "weight": 0.35,
        "probs": {
            "Tea": 0.90,
            "Jam": 0.85,
            "Bread": 0.50,
            "Coffee": 0.05,
            "Butter": 0.03
        }
    },
    "Mixed": {
        "weight": 0.20,
        "probs": {
            "Coffee": 0.35,
            "Bread": 0.40,
            "Butter": 0.35,
            "Tea": 0.35,
            "Jam": 0.35
        }
    }
}

# Total transactions to generate
NUM_TRANSACTIONS = 5000
START_TX_ID = 101

# Decision #5: Mining thresholds
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.50
MIN_LIFT = 1.0

# Phase 3: Ground Truth Verification Assertions
GROUND_TRUTH_ASSERTIONS = {
    ("Coffee", "Butter"): {"min_lift": 1.3, "direction": "bonded"},
    ("Tea", "Jam"): {"min_lift": 1.3, "direction": "bonded"},
    ("Coffee", "Jam"): {"max_lift": 1.0, "direction": "repelling"}
}
