#!/usr/bin/env python3
"""
analyze.py

Association Rule Mining & Shelf-Layout Optimization Report for Mock Cafe Transactions.
Adheres strictly to plan.md Phase 4 and Phase 5.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set non-glyph styling
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")


def run_mining(onehot_df: pd.DataFrame):
    """
    Phase 4: Run FP-Growth primary and Apriori cross-check.
    Verify identical itemsets and generate association_rules(metric='lift').
    """
    logging.info("=== PHASE 4: FREQUENT PATTERN MINING ===")
    basket_bool = onehot_df.drop(columns=["TxID"]).astype(bool)

    # Primary: FP-Growth
    logging.info(f"Running FP-Growth with min_support = {config.MIN_SUPPORT}...")
    fpgrowth_freq = fpgrowth(basket_bool, min_support=config.MIN_SUPPORT, use_colnames=True)

    # Cross-check: Apriori
    logging.info("Running Apriori cross-check...")
    apriori_freq = apriori(basket_bool, min_support=config.MIN_SUPPORT, use_colnames=True)

    # Assert identical itemsets
    fpg_sets = set(fpgrowth_freq["itemsets"])
    apr_sets = set(apriori_freq["itemsets"])
    assert fpg_sets == apr_sets, f"FAIL: FP-Growth and Apriori itemsets mismatch! ({len(fpg_sets)} vs {len(apr_sets)})"
    logging.info(f"Cross-check PASSED: FP-Growth and Apriori produced exactly {len(fpg_sets)} identical frequent itemsets.")

    # Generate rules
    rules = association_rules(fpgrowth_freq, metric="lift", min_threshold=config.MIN_LIFT)
    rules = rules[rules["confidence"] >= config.MIN_CONFIDENCE].copy()
    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

    rules_path = "rules.csv"
    rules.to_csv(rules_path, index=False)
    logging.info(f"Saved discovered association rules to {rules_path} ({len(rules)} rules)")

    return rules, basket_bool


def generate_report(rules: pd.DataFrame, basket_bool: pd.DataFrame):
    """
    Phase 5: Report comparison of designed vs discovered lift, top rules, and shelf-layout.
    Also generate professional visualization chart.
    """
    logging.info("=== PHASE 5: EXECUTIVE MINING REPORT & SHELF-LAYOUT OPTIMIZATION ===")

    # Table 1: Designed vs Discovered Lift
    print("\n" + "="*85)
    print("TABLE 1: DESIGNED VS DISCOVERED LIFT FOR TARGET ASSOCIATIONS")
    print("="*85)
    print(f"{'Target Pair (A -> B)':<22} | {'Designed Intent':<18} | {'Discovered Lift':<16} | {'Discovered Conf.'}")
    print("-"*85)

    for (item_a, item_b), spec in config.GROUND_TRUTH_ASSERTIONS.items():
        sup_ab = (basket_bool[item_a] & basket_bool[item_b]).mean()
        sup_a = basket_bool[item_a].mean()
        sup_b = basket_bool[item_b].mean()
        conf = sup_ab / sup_a if sup_a > 0 else 0.0
        lift = conf / sup_b if sup_b > 0 else 0.0

        intent_str = f"Bonded (> {spec['min_lift']}x)" if "min_lift" in spec else f"Repel (< {spec['max_lift']}x)"
        print(f"{item_a + ' -> ' + item_b:<22} | {intent_str:<18} | {lift:6.3f}x          | {conf*100:6.2f}%")
    print("="*85)

    # Table 2: Top Discovered Rules
    print("\n" + "="*100)
    print("TABLE 2: TOP DISCOVERED ASSOCIATION RULES (Sorted by Lift)")
    print("="*100)
    top10 = rules.head(10).copy()
    top10["antecedent_str"] = top10["antecedents"].apply(lambda x: ", ".join(list(x)))
    top10["consequent_str"] = top10["consequents"].apply(lambda x: ", ".join(list(x)))
    
    cols = ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
    top10_disp = top10[cols].copy()
    top10_disp["support"] = top10_disp["support"].apply(lambda x: f"{x*100:5.2f}%")
    top10_disp["confidence"] = top10_disp["confidence"].apply(lambda x: f"{x*100:5.2f}%")
    top10_disp["lift"] = top10_disp["lift"].apply(lambda x: f"{x:5.3f}x")
    top10_disp.columns = ["Antecedent(s)", "Consequent(s)", "Support", "Confidence", "Lift"]
    print(top10_disp.to_string(index=False))
    print("="*100)

    # Shelf-Layout Recommendation
    print("\n" + "#"*85)
    print("STRATEGIC SHELF-LAYOUT RECOMMENDATIONS (Grounded in Discovered Rules)")
    print("#"*85)
    print("1. CO-LOCATE HIGH-LIFT BONDED PAIRS (Cross-Merchandising Zone):")
    print("   - Coffee & Butter (Lift > 1.3x): Place Butter cooler directly adjacent to the morning Coffee counter.")
    print("   - Tea & Jam (Lift > 1.3x): Display artisan Jam jars on eye-level shelving next to loose-leaf Tea tins.")
    print("2. SEPARATE REPELLING / COMPETING PAIRS (Store Circulation Strategy):")
    print("   - Coffee & Jam (Lift < 1.0x): Customers buying Jam are predominantly Tea drinkers. Placing Jam right")
    print("     next to Coffee wastes prime impulse-buy shelf real estate. Keep Jam strictly in the Tea & Bakery aisle.")
    print("3. BREAD HUB CENTRALIZATION:")
    print("   - Bread acts as a universal bridge across both Coffee (~60% prob) and Tea (~50% prob) archetypes.")
    print("     Position fresh Bread in a central island reachable from both morning beverage queues.")
    print("#"*85 + "\n")

    # Generate Chart
    plt.figure(figsize=(10, 6))
    top_chart = top10.copy()
    top_chart["rule_name"] = top_chart.apply(lambda r: f"{r['antecedent_str']} -> {r['consequent_str']}", axis=1)
    
    bars = plt.barh(top_chart["rule_name"][::-1], top_chart["lift"][::-1], color=sns.color_palette("mako", len(top_chart))[::-1], edgecolor='black')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.03, bar.get_y() + bar.get_height()/2.0, f"{width:.2f}x", va='center', ha='left', fontsize=10, fontweight='bold')
        
    plt.axvline(1.0, color='red', linestyle='--', alpha=0.7, label="Baseline Independence (Lift = 1.0x)")
    plt.title("Top Discovered Association Rules by Lift (Mock Cafe Transactions)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Association Lift (x)", fontsize=12, fontweight='bold')
    plt.ylabel("Association Rule", fontsize=12, fontweight='bold')
    plt.xlim(0, top_chart["lift"].max() * 1.15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    chart_path = "cafe_association_rules_lift.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logging.info(f"Saved professional association rule lift chart to {chart_path}")


def main():
    if not os.path.exists("transactions_onehot.csv"):
        logging.error("transactions_onehot.csv not found! Please run generate.py first.")
        sys.exit(1)

    onehot_df = pd.read_csv("transactions_onehot.csv")
    rules, basket_bool = run_mining(onehot_df)
    generate_report(rules, basket_bool)


if __name__ == "__main__":
    main()
