#!/usr/bin/env python3
"""
generate.py

Data Generator & Ground Truth Verifier for Mock Cafe Market Basket Analysis.
Adheres strictly to plan.md Phase 2 and Phase 3.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def generate_transactions():
    """
    Generate 5,000 transactions using the latent-class archetype method.
    Guarantees no empty baskets.
    Outputs long format (transactions.csv) and wide one-hot format (transactions_onehot.csv).
    """
    logging.info(f"Initializing random generator with fixed seed = {config.RANDOM_SEED}")
    rng = np.random.default_rng(config.RANDOM_SEED)

    archetype_names = list(config.ARCHETYPES.keys())
    archetype_weights = [config.ARCHETYPES[name]["weight"] for name in archetype_names]

    logging.info(f"Sampling {config.NUM_TRANSACTIONS:,} transactions across archetypes: {config.ARCHETYPES}")

    long_rows = []
    onehot_rows = []

    for idx in range(config.NUM_TRANSACTIONS):
        tx_id = config.START_TX_ID + idx
        
        # Guarantee no empty basket (resample if all items are 0)
        basket_items = []
        while not basket_items:
            chosen_archetype = rng.choice(archetype_names, p=archetype_weights)
            probs = config.ARCHETYPES[chosen_archetype]["probs"]
            
            for item in config.CATALOG:
                if rng.random() < probs[item]:
                    basket_items.append(item)

        # Long format row
        for item in basket_items:
            long_rows.append({"TxID": tx_id, "Item": item})

        # One-hot row
        onehot_dict = {"TxID": tx_id}
        for item in config.CATALOG:
            onehot_dict[item] = bool(item in basket_items)
        onehot_rows.append(onehot_dict)

    long_df = pd.DataFrame(long_rows)
    onehot_df = pd.DataFrame(onehot_rows)

    long_path = "transactions.csv"
    onehot_path = "transactions_onehot.csv"

    long_df.to_csv(long_path, index=False)
    onehot_df.to_csv(onehot_path, index=False)

    logging.info(f"Saved long format transactions to {long_path} ({len(long_df):,} rows)")
    logging.info(f"Saved wide one-hot format to {onehot_path} ({len(onehot_df):,} rows)")

    return long_df, onehot_df


def verify_ground_truth(onehot_df: pd.DataFrame):
    """
    Phase 3: Verify Ground Truth.
    Compute support, confidence, and lift for designed pairs.
    Assert they match intended direction or fail loud.
    """
    logging.info("=== PHASE 3: GROUND TRUTH VERIFICATION ===")
    n = len(onehot_df)

    # Compute item supports
    supports = {item: onehot_df[item].mean() for item in config.CATALOG}

    print("\n" + "-"*80)
    print("ITEM SUPPORTS (Base Penetration):")
    for item, sup in supports.items():
        print(f"  {item:<10}: {sup*100:6.2f}%")
    print("-"*80)

    print("\nDESIGNED PAIRWISE ASSOCIATIONS (Ground Truth Verification):")
    print(f"{'Pair (A -> B)':<22} | {'Support(A ∩ B)':<15} | {'Conf(A -> B)':<14} | {'Lift':<8} | {'Status'}")
    print("-"*80)

    for (item_a, item_b), rules_spec in config.GROUND_TRUTH_ASSERTIONS.items():
        sup_a = supports[item_a]
        sup_b = supports[item_b]
        sup_ab = (onehot_df[item_a] & onehot_df[item_b]).mean()

        conf_ab = sup_ab / sup_a if sup_a > 0 else 0.0
        lift_ab = conf_ab / sup_b if sup_b > 0 else 0.0

        status = "PASS"
        try:
            if "min_lift" in rules_spec:
                assert lift_ab > rules_spec["min_lift"], f"FAIL: Expected Lift({item_a}->{item_b}) > {rules_spec['min_lift']}, got {lift_ab:.3f}"
            if "max_lift" in rules_spec:
                assert lift_ab < rules_spec["max_lift"], f"FAIL: Expected Lift({item_a}->{item_b}) < {rules_spec['max_lift']}, got {lift_ab:.3f}"
        except AssertionError as e:
            status = f"FAIL ({e})"
            logging.error(status)
            raise e

        print(f"{item_a + ' -> ' + item_b:<22} | {sup_ab*100:6.2f}%         | {conf_ab*100:6.2f}%       | {lift_ab:6.3f}x | {status}")

    print("-"*80 + "\n")
    logging.info("All Phase 3 ground truth assertions passed successfully!")


def main():
    logging.info("Starting Phase 2 generation...")
    long_df, onehot_df = generate_transactions()
    verify_ground_truth(onehot_df)


if __name__ == "__main__":
    main()
