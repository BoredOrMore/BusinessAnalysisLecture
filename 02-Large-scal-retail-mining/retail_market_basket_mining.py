#!/usr/bin/env python3
"""
retail_market_basket_mining.py

Programmatic Market Basket Analysis (MBA) & Financial Conversion Engine
Dataset: UCI Machine Learning Repository - Online Retail (ID: 352)
Business Context: Large-scale online retail transaction mining using mlxtend to optimize
hyperparameters (min_support, lift/confidence thresholds) and translate top association rules
into exact annual profit margin run rate increments.

Author: Antigravity AI Data Science Team
Adheres to: DATA_SCIENCE_PIPELINE.md
"""

import os
import sys
import time
import ssl
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth, association_rules
from ucimlrepo import fetch_ucirepo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set non-glyph plotting style
plt.style.use('default')
sns.set_theme(style="whitegrid", palette="deep")


def load_and_clean_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Ingest UCI Online Retail dataset (ID=352), either from local cache or via ucimlrepo API,
    and perform rigorous cleaning (removing cancellations, missing descriptions, and non-product overhead).
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, "online_retail.csv")

    if os.path.exists(cache_path):
        logging.info(f"Loading cached dataset from {cache_path}...")
        df = pd.read_csv(cache_path)
    else:
        logging.info("Fetching UCI Online Retail dataset (ID=352) via API...")
        # Workaround for SSL certificate issues on macOS
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            online_retail = fetch_ucirepo(id=352)
            if online_retail.data.original is not None:
                df = online_retail.data.original
            else:
                X = online_retail.data.features
                y = online_retail.data.targets
                df = pd.concat([X, y], axis=1) if y is not None else X
            df.to_csv(cache_path, index=False)
            logging.info(f"Saved raw dataset to {cache_path} ({len(df):,} rows)")
        except Exception as e:
            logging.error(f"Failed to fetch dataset via API: {e}")
            raise e

    logging.info(f"Raw dataset shape: {df.shape[0]:,} rows across {df.shape[1]} columns.")

    # Data cleaning
    # 1. Filter positive quantities and positive unit prices
    clean_df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].dropna(subset=["Description"]).copy()
    clean_df["Description"] = clean_df["Description"].astype(str).str.strip().str.upper()

    # 2. Exclude cancellations ('C' prefix in InvoiceNo)
    clean_df["InvoiceNo"] = clean_df["InvoiceNo"].astype(str)
    clean_df = clean_df[~clean_df["InvoiceNo"].str.upper().str.startswith("C")]

    # 3. Filter out non-product overhead and shipping codes
    overhead_keywords = [
        "POSTAGE", "DOTCOM POSTAGE", "CARRIAGE", "CRUK COMMISSION", "BANK CHARGES",
        "DISCOUNT", "MANUAL", "AMAZON FEE", "SAMPLES", "ADJUSTMENT"
    ]
    clean_df = clean_df[~clean_df["Description"].isin(overhead_keywords)]

    logging.info(f"Cleaned product transaction dataset: {len(clean_df):,} rows across {clean_df['InvoiceNo'].nunique():,} unique invoices.")
    return clean_df


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a boolean Transaction x Item matrix (one-hot encoded) required for mlxtend frequent pattern mining.
    """
    logging.info("Building one-hot encoded boolean basket matrix...")
    t0 = time.time()
    # Group by InvoiceNo and Description, sum quantity, unstack
    basket = df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().reset_index().fillna(0)
    basket = basket.set_index("InvoiceNo")
    basket_bool = (basket > 0)
    dt = time.time() - t0
    logging.info(f"Basket matrix constructed: {basket_bool.shape[0]:,} transactions x {basket_bool.shape[1]:,} distinct items in {dt:.2f}s.")
    return basket_bool


def run_hyperparameter_optimization(basket_bool: pd.DataFrame) -> pd.DataFrame:
    """
    Run programmatic hyperparameter grid search across minimum support and confidence thresholds
    to evaluate frequent itemset generation and rule yield.
    """
    logging.info("Running hyperparameter optimization grid search over min_support and confidence...")
    support_grid = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
    conf_grid = [0.20, 0.30, 0.40, 0.50]

    results = []
    for sup in support_grid:
        t0 = time.time()
        freq_items = fpgrowth(basket_bool, min_support=sup, use_colnames=True)
        dt = time.time() - t0
        num_itemsets = len(freq_items)

        if num_itemsets > 0:
            all_rules = association_rules(freq_items, metric="lift", min_threshold=1.1)
            for c in conf_grid:
                valid_rules = all_rules[all_rules["confidence"] >= c]
                results.append({
                    "min_support": sup,
                    "min_confidence": c,
                    "num_itemsets": num_itemsets,
                    "num_rules": len(valid_rules),
                    "exec_time_sec": dt
                })
        else:
            for c in conf_grid:
                results.append({
                    "min_support": sup,
                    "min_confidence": c,
                    "num_itemsets": 0,
                    "num_rules": 0,
                    "exec_time_sec": dt
                })

    grid_df = pd.DataFrame(results)
    logging.info("Hyperparameter optimization grid search completed successfully.")
    return grid_df


def compute_financial_conversion(
    df: pd.DataFrame,
    basket_bool: pd.DataFrame,
    min_support: float = 0.015,
    min_confidence: float = 0.30,
    min_lift: float = 1.50,
    alpha_conversion: float = 0.18,
    gross_margin: float = 0.38
) -> pd.DataFrame:
    """
    Translate association rules into financial metrics:
    - Compute antecedent opportunity baskets: N_opp = N_A - N_AB
    - Model recommendation engine conversion rate alpha (default 18%)
    - Translate units sold into incremental gross profit contribution and annual run rate.
    """
    logging.info(f"Extracting actionable association rules at min_support={min_support}, min_confidence={min_confidence}, min_lift={min_lift}...")
    n_baskets = len(basket_bool)

    # Compute item-level average price and average quantity per transaction
    item_metrics = df.groupby("Description").agg(
        avg_price=("UnitPrice", "mean"),
        avg_qty=("Quantity", lambda x: x.sum() / x.nunique()),
        total_rev=("Quantity", lambda x: (x * df.loc[x.index, "UnitPrice"]).sum())
    )

    # Frequent pattern mining
    freq_items = fpgrowth(basket_bool, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence].copy()

    # Isolate single-item antecedents and single-item consequents for clean cross-sell actionability
    rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
    rules["antecedent"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["consequent"] = rules["consequents"].apply(lambda x: list(x)[0])

    # Financial conversion formula
    rules["n_A"] = rules["antecedent support"] * n_baskets
    rules["n_AB"] = rules["support"] * n_baskets
    rules["n_opp"] = rules["n_A"] - rules["n_AB"]

    rules["cons_price"] = rules["consequent"].map(item_metrics["avg_price"])
    rules["cons_qty"] = rules["consequent"].map(item_metrics["avg_qty"])

    # Calculate dataset duration in days for annualization run rate
    min_date = pd.to_datetime(df["InvoiceDate"]).min()
    max_date = pd.to_datetime(df["InvoiceDate"]).max()
    days_in_dataset = max((max_date - min_date).days, 1.0)

    # Incremental revenue across the dataset period
    rules["inc_rev_period"] = rules["n_opp"] * alpha_conversion * rules["cons_qty"] * rules["cons_price"]
    # Annualized profit margin run rate increment
    rules["inc_profit_annual"] = rules["inc_rev_period"] * gross_margin * (365.0 / days_in_dataset)

    # Sort by financial impact
    rules = rules.sort_values(by="inc_profit_annual", ascending=False).reset_index(drop=True)
    logging.info(f"Financial conversion modeled across {len(rules)} actionable single-item rules over {days_in_dataset} days.")
    return rules


def generate_visualizations(grid_df: pd.DataFrame, rules_df: pd.DataFrame, output_dir: str = "."):
    """
    Generate professional executive-ready charts at 300 DPI without glyph dependencies.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Hyperparameter Optimization Surface
    plt.figure(figsize=(10, 6))
    for conf in sorted(grid_df["min_confidence"].unique()):
        sub = grid_df[grid_df["min_confidence"] == conf]
        plt.plot(sub["min_support"] * 100, sub["num_rules"], marker='o', linewidth=2.5, label=f"Min Confidence = {int(conf*100)}%")

    plt.title("Market Basket Analysis: Hyperparameter Optimization Surface", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Minimum Support Threshold (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Actionable Association Rules Generated", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Confidence Rule", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "hyperparameter_optimization_surface.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    logging.info(f"Saved hyperparameter optimization chart to {chart1_path}")

    # Chart 2: Top Association Rules by Annual Profit Run Rate Increment
    top_rules = rules_df.head(15).copy()
    top_rules["rule_label"] = top_rules.apply(
        lambda r: f"{r['antecedent'][:22]} -> {r['consequent'][:22]} (Lift: {r['lift']:.1f}x)", axis=1
    )

    plt.figure(figsize=(12, 8))
    norm = plt.Normalize(top_rules["lift"].min(), top_rules["lift"].max())
    colors = plt.cm.viridis(norm(top_rules["lift"]))

    bars = plt.barh(top_rules["rule_label"][::-1], top_rules["inc_profit_annual"][::-1], color=colors[::-1], edgecolor='black', alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 2000,
            bar.get_y() + bar.get_height() / 2.0,
            f"${width:,.0f}/yr",
            va='center',
            ha='left',
            fontsize=9.5,
            fontweight='bold'
        )

    plt.title("Top 15 Strategic Cross-Sell Rules by Annual Profit Margin Run Rate Lift", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Annualized Incremental Gross Profit Contribution ($ / year)", fontsize=12, fontweight='bold')
    plt.ylabel("Association Rule (Antecedent -> Consequent)", fontsize=12, fontweight='bold')
    plt.xlim(0, top_rules["inc_profit_annual"].max() * 1.18)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Colorbar for Lift
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
    cbar.set_label("Association Rule Lift (x)", fontsize=11, fontweight='bold')

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "top_association_rules_profit_impact.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    logging.info(f"Saved financial impact chart to {chart2_path}")


def main():
    parser = argparse.ArgumentParser(description="Retail Market Basket Mining & Financial Conversion Engine")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store cached UCI dataset")
    parser.add_argument("--min_support", type=float, default=0.015, help="Optimal minimum support threshold")
    parser.add_argument("--min_confidence", type=float, default=0.30, help="Optimal minimum confidence threshold")
    parser.add_argument("--alpha", type=float, default=0.18, help="Recommendation engine conversion rate (default: 18 percent)")
    parser.add_argument("--margin", type=float, default=0.38, help="Gross contribution margin (default: 38 percent)")
    args = parser.parse_args()

    logging.info("=== STARTING RETAIL MARKET BASKET MINING PIPELINE ===")
    
    # 1. Ingest and clean data
    clean_df = load_and_clean_data(data_dir=args.data_dir)

    # 2. Build basket matrix
    basket_bool = build_basket_matrix(clean_df)

    # 3. Hyperparameter optimization
    grid_df = run_hyperparameter_optimization(basket_bool)

    # 4. Financial conversion modeling
    rules_df = compute_financial_conversion(
        clean_df,
        basket_bool,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        alpha_conversion=args.alpha,
        gross_margin=args.margin
    )

    # 5. Generate executive charts
    generate_visualizations(grid_df, rules_df, output_dir=".")

    # 6. Executive Summary Table
    print("\n" + "="*110)
    print("EXECUTIVE FINANCIAL CONVERSION SUMMARY: TOP 10 STRATEGIC CROSS-SELL RULES")
    print("="*110)
    cols_to_show = ["antecedent", "consequent", "support", "confidence", "lift", "n_opp", "cons_price", "inc_profit_annual"]
    summary_df = rules_df[cols_to_show].head(10).copy()
    summary_df["support"] = summary_df["support"].apply(lambda x: f"{x*100:5.2f}%")
    summary_df["confidence"] = summary_df["confidence"].apply(lambda x: f"{x*100:5.1f}%")
    summary_df["lift"] = summary_df["lift"].apply(lambda x: f"{x:5.2f}x")
    summary_df["n_opp"] = summary_df["n_opp"].apply(lambda x: f"{x:6.0f}")
    summary_df["cons_price"] = summary_df["cons_price"].apply(lambda x: f"${x:6.2f}")
    summary_df["inc_profit_annual"] = summary_df["inc_profit_annual"].apply(lambda x: f"${x:12,.2f}")
    
    summary_df.columns = ["Antecedent SKU", "Consequent SKU", "Support", "Conf.", "Lift", "Opp. Baskets", "Cons. Price", "Annual Profit Lift ($)"]
    print(summary_df.to_string(index=False))
    print("="*110)

    total_portfolio_profit = rules_df["inc_profit_annual"].head(10).sum()
    print(f"\n---> TOTAL ANNUALIZED PROFIT RUN RATE INCREMENT (Top 10 Rules Portfolio): ${total_portfolio_profit:,.2f} / year <---")
    print("="*110 + "\n")
    logging.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
