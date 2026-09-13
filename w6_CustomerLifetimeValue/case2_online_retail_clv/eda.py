"""Exploratory Data Analysis (EDA) module for Case Study 2: Enterprise Online Retail.

Analyzes transaction dynamics, data hygiene loss, revenue concentration (Pareto),
order velocity, geographic footprint, and the empirical latent defection cliff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    OUTPUTS_DIR,
    RAW_DATA_PATH,
)


def set_plotting_theme() -> None:
    """Apply clean publication-ready plotting style."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300


def run_eda(raw_path: Path = RAW_DATA_PATH) -> dict:
    """Execute complete EDA workflow and generate diagnostic figures and tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    set_plotting_theme()

    print("Loading raw transactional data...")
    raw_df = pd.read_csv(raw_path, encoding="latin-1")
    total_raw_rows = len(raw_df)

    # 1. Hygiene & Data Integrity Diagnostics
    null_cust_mask = raw_df["CustomerID"].isna()
    cancel_mask = raw_df["Quantity"] <= 0
    zero_price_mask = (raw_df["UnitPrice"] <= 0) & (~cancel_mask) & (~null_cust_mask)

    valid_mask = (~null_cust_mask) & (raw_df["Quantity"] > 0) & (raw_df["UnitPrice"] > 0)
    df = raw_df[valid_mask].copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    # 2. Monthly Run-Rate Dynamics
    monthly = df.groupby("YearMonth").agg(
        revenue=("TotalSpend", "sum"),
        transactions=("InvoiceNo", "nunique"),
        active_customers=("CustomerID", "nunique"),
        items_sold=("Quantity", "sum"),
    ).reset_index()
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)
    monthly["aov"] = monthly["revenue"] / monthly["transactions"]
    monthly.to_csv(OUTPUTS_DIR / "eda_monthly_revenue.csv", index=False)

    # 3. Country / Geographic Footprint
    country_summary = df.groupby("Country").agg(
        total_revenue=("TotalSpend", "sum"),
        order_count=("InvoiceNo", "nunique"),
        customer_count=("CustomerID", "nunique"),
        items_sold=("Quantity", "sum"),
    ).reset_index()
    country_summary["revenue_share_pct"] = (country_summary["total_revenue"] / df["TotalSpend"].sum()) * 100.0
    country_summary["aov"] = country_summary["total_revenue"] / country_summary["order_count"]
    country_summary = country_summary.sort_values("total_revenue", ascending=False)
    country_summary.to_csv(OUTPUTS_DIR / "eda_country_breakdown.csv", index=False)

    # 4. Customer-Level Aggregations & RFM Metrics
    customer_orders = df.groupby(["CustomerID", df["InvoiceDate"].dt.floor("D")]).agg(
        order_spend=("TotalSpend", "sum"),
        items_count=("Quantity", "sum"),
    ).reset_index()

    obs_end = customer_orders["InvoiceDate"].max()

    cust_summary = customer_orders.groupby("CustomerID").agg(
        first_order=("InvoiceDate", "min"),
        last_order=("InvoiceDate", "max"),
        order_count=("InvoiceDate", "count"),
        total_spend=("order_spend", "sum"),
        avg_order_value=("order_spend", "mean"),
    ).reset_index()

    cust_summary["recency_days"] = (obs_end - cust_summary["last_order"]).dt.days
    cust_summary["tenure_days"] = (obs_end - cust_summary["first_order"]).dt.days
    cust_summary["repeat_frequency"] = cust_summary["order_count"] - 1

    # 5. Pareto Revenue Concentration Analysis
    cust_summary = cust_summary.sort_values("total_spend", ascending=False).reset_index(drop=True)
    cust_summary["cumulative_spend"] = cust_summary["total_spend"].cumsum()
    total_spend_all = cust_summary["total_spend"].sum()
    cust_summary["cumulative_spend_pct"] = (cust_summary["cumulative_spend"] / total_spend_all) * 100.0
    cust_summary["customer_rank_pct"] = ((cust_summary.index + 1) / len(cust_summary)) * 100.0

    pareto_1pct = float(cust_summary[cust_summary["customer_rank_pct"] <= 1.0]["cumulative_spend_pct"].max())
    pareto_5pct = float(cust_summary[cust_summary["customer_rank_pct"] <= 5.0]["cumulative_spend_pct"].max())
    pareto_10pct = float(cust_summary[cust_summary["customer_rank_pct"] <= 10.0]["cumulative_spend_pct"].max())
    pareto_20pct = float(cust_summary[cust_summary["customer_rank_pct"] <= 20.0]["cumulative_spend_pct"].max())

    pareto_table = pd.DataFrame([
        {"Tier": "Top 1% Customers", "Cumulative_Revenue_Share_Pct": pareto_1pct},
        {"Tier": "Top 5% Customers", "Cumulative_Revenue_Share_Pct": pareto_5pct},
        {"Tier": "Top 10% Customers", "Cumulative_Revenue_Share_Pct": pareto_10pct},
        {"Tier": "Top 20% Customers", "Cumulative_Revenue_Share_Pct": pareto_20pct},
    ])
    pareto_table.to_csv(OUTPUTS_DIR / "eda_pareto_distribution.csv", index=False)

    # 6. Empirical Latent Defection Cliff Analysis (Slide 26)
    # Group customers by inactivity bucket and compute repeat purchase probability
    bins = [0, 30, 60, 90, 120, 180, 270, 375]
    labels = ["0-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-270d", "271d+"]
    cust_summary["recency_bin"] = pd.cut(cust_summary["recency_days"], bins=bins, labels=labels, right=True)

    defection_cohorts = cust_summary.groupby("recency_bin", observed=False).agg(
        customer_count=("CustomerID", "count"),
        total_revenue=("total_spend", "sum"),
        mean_orders=("order_count", "mean"),
        repeat_buyer_count=("repeat_frequency", lambda x: (x > 0).sum()),
    ).reset_index()
    defection_cohorts["repeat_buyer_rate_pct"] = (defection_cohorts["repeat_buyer_count"] / defection_cohorts["customer_count"]) * 100.0
    defection_cohorts["revenue_share_pct"] = (defection_cohorts["total_revenue"] / total_spend_all) * 100.0
    defection_cohorts.to_csv(OUTPUTS_DIR / "eda_defection_cohorts.csv", index=False)

    # 7. Summary Metrics Dictionary
    eda_metrics = {
        "raw_records": total_raw_rows,
        "clean_records": len(df),
        "clean_record_retention_pct": float((len(df) / total_raw_rows) * 100.0),
        "anonymous_checkout_records": int(null_cust_mask.sum()),
        "cancellations_returns_records": int(cancel_mask.sum()),
        "unique_verified_customers": int(df["CustomerID"].nunique()),
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "gross_merchandise_value": float(total_spend_all),
        "mean_customer_lifetime_spend": float(cust_summary["total_spend"].mean()),
        "median_customer_lifetime_spend": float(cust_summary["total_spend"].median()),
        "mean_order_value": float(df["TotalSpend"].sum() / df["InvoiceNo"].nunique()),
        "repeat_buyer_count": int((cust_summary["repeat_frequency"] > 0).sum()),
        "repeat_buyer_pct": float((cust_summary["repeat_frequency"] > 0).mean() * 100.0),
        "one_time_buyer_count": int((cust_summary["repeat_frequency"] == 0).sum()),
        "one_time_buyer_pct": float((cust_summary["repeat_frequency"] == 0).mean() * 100.0),
        "pareto_metrics": {
            "top_1pct_revenue_share": pareto_1pct,
            "top_5pct_revenue_share": pareto_5pct,
            "top_10pct_revenue_share": pareto_10pct,
            "top_20pct_revenue_share": pareto_20pct,
        },
        "uk_revenue_share_pct": float(country_summary.loc[country_summary["Country"] == "United Kingdom", "revenue_share_pct"].iloc[0]),
    }

    with open(OUTPUTS_DIR / "eda_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_metrics, f, indent=2)

    # =========================================================================
    # Visualizations
    # =========================================================================

    # Figure 1: Monthly Revenue & Active Customers Trend
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    x_coords = np.arange(len(monthly))
    bars = ax1.bar(x_coords, monthly["revenue"] / 1000.0, width=0.55, color="#4C72B0", alpha=0.85, label="Monthly Revenue ($k)")
    line = ax2.plot(x_coords, monthly["active_customers"], color="#C44E52", marker="o", linewidth=2.5, label="Active Customers")
    
    ax1.set_xticks(x_coords)
    ax1.set_xticklabels(monthly["YearMonth_str"], rotation=30, ha="right", fontsize=9.5)
    ax1.set_ylabel("Gross Revenue ($k)", fontsize=11, color="#4C72B0", fontweight="bold")
    ax2.set_ylabel("Active Unique Customers", fontsize=11, color="#C44E52", fontweight="bold")
    ax1.set_title("Enterprise Monthly Revenue Trajectory & Active Customer Count", fontsize=13, fontweight="bold")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "eda_monthly_revenue_trend.png", dpi=300)
    plt.close()

    # Figure 2: Pareto Cumulative Revenue Curve
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(cust_summary["customer_rank_pct"], cust_summary["cumulative_spend_pct"], color="#2ca02c", linewidth=2.5, label="Actual Revenue Curve")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", alpha=0.7, label="Equal Distribution Line")
    
    # Highlight 20/80 point
    ax.scatter([20.0], [pareto_20pct], color="#d62728", s=80, zorder=5)
    ax.axvline(20.0, color="#d62728", linestyle=":", alpha=0.7)
    ax.axhline(pareto_20pct, color="#d62728", linestyle=":", alpha=0.7)
    ax.annotate(
        f"Top 20% Customers generate\n{pareto_20pct:.1f}% of Total Revenue",
        (22.0, pareto_20pct - 8),
        fontsize=10,
        fontweight="bold",
        color="#d62728",
    )
    
    ax.set_title("Pareto Principle Verification: Cumulative Customer Revenue Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cumulative Customer Percentage (%)", fontsize=11)
    ax.set_ylabel("Cumulative Revenue Share (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "eda_pareto_revenue_curve.png", dpi=300)
    plt.close()

    # Figure 3: RFM Multi-Panel Distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # Panel A: Recency
    sns.histplot(cust_summary["recency_days"], bins=30, color="#4C72B0", kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("A. Days Since Last Purchase (Recency)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Recency (Days)")
    
    # Panel B: Frequency (capped display)
    sns.histplot(cust_summary["repeat_frequency"].clip(upper=25), bins=26, color="#55A868", discrete=True, ax=axes[0, 1])
    axes[0, 1].set_title("B. Repeat Purchase Frequency (Capped at 25)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Repeat Purchases (Orders - 1)")
    
    # Panel C: Customer Tenure
    sns.histplot(cust_summary["tenure_days"], bins=30, color="#8172B3", kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("C. Customer Lifespan / Tenure", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Tenure from First Order (Days)")
    
    # Panel D: Total Spend (Log scale)
    sns.histplot(cust_summary["total_spend"], bins=40, color="#C44E52", log_scale=True, ax=axes[1, 1])
    axes[1, 1].set_title("D. Customer Total Spend Distribution (Log Scale)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Total Spend ($)")
    
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "eda_rfm_distributions.png", dpi=300)
    plt.close()

    # Figure 4: Empirical Latent Defection Cliff
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax2 = ax1.twinx()
    
    x_pos = np.arange(len(defection_cohorts))
    bars = ax1.bar(x_pos, defection_cohorts["customer_count"], width=0.5, color="#1f77b4", alpha=0.75, label="Customer Count")
    line = ax2.plot(x_pos, defection_cohorts["repeat_buyer_rate_pct"], color="#d62728", marker="s", linewidth=2.5, label="Repeat Buyer %")
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(defection_cohorts["recency_bin"], fontsize=10)
    ax1.set_xlabel("Inactivity Window (Recency Days)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Customers", fontsize=11, color="#1f77b4", fontweight="bold")
    ax2.set_ylabel("Repeat Buyer Proportion (%)", fontsize=11, color="#d62728", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax1.set_title("Empirical Latent Defection Cliff (Slide 26)", fontsize=13, fontweight="bold")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "eda_defection_cliff_analysis.png", dpi=300)
    plt.close()

    # Figure 5: Top 10 Countries by Revenue
    fig, ax = plt.subplots(figsize=(10, 5.5))
    top10_countries = country_summary.head(10).copy()
    # Exclude UK to see international spread clearly or include in inset
    bars = ax.barh(top10_countries["Country"][::-1], top10_countries["total_revenue"][::-1] / 1000.0, color="#4C72B0", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 30, bar.get_y() + 0.18, f"${w:,.1f}k", fontsize=9, fontweight="semibold")
    ax.set_title("Top 10 Geographic Markets by Gross Revenue ($k)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Revenue ($k)", fontsize=11)
    ax.set_xlim(0, max(top10_countries["total_revenue"] / 1000.0) * 1.18)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "eda_top_countries_breakdown.png", dpi=300)
    plt.close()

    return eda_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on Online Retail dataset.")
    parser.add_argument("--data_path", type=str, default=str(RAW_DATA_PATH), help="Path to raw dataset")
    args = parser.parse_args()

    metrics = run_eda(raw_path=Path(args.data_path))

    print("=" * 95)
    print("      CASE STUDY 2: EXPLORATORY DATA ANALYSIS (EDA) & DIAGNOSTIC REPORT      ")
    print("=" * 95)
    print(f"Total Raw Log Rows:              {metrics['raw_records']:,}")
    print(f"Clean Transaction Rows:          {metrics['clean_records']:,} ({metrics['clean_record_retention_pct']:.1f}%)")
    print(f"Anonymous Checkouts Filtered:    {metrics['anonymous_checkout_records']:,}")
    print(f"Returns & Cancellations Filtered:{metrics['cancellations_returns_records']:,}")
    print(f"Unique Verified Customer Base:   {metrics['unique_verified_customers']:,}")
    print(f"Unique Products (SKUs):          {metrics['unique_products']:,}")
    print(f"Gross Transaction Volume:        ${metrics['gross_merchandise_value']:,.2f}")
    print(f"Mean Customer Lifetime Spend:    ${metrics['mean_customer_lifetime_spend']:,.2f}")
    print(f"Median Customer Lifetime Spend:  ${metrics['median_customer_lifetime_spend']:,.2f}")
    print(f"Mean Order Basket Value (AOV):   ${metrics['mean_order_value']:,.2f}")
    print(f"Repeat Buyers vs One-Time:       {metrics['repeat_buyer_pct']:.1f}% Repeat ({metrics['repeat_buyer_count']:,}) vs {metrics['one_time_buyer_pct']:.1f}% One-Time ({metrics['one_time_buyer_count']:,})")
    print(f"UK Revenue Contribution:         {metrics['uk_revenue_share_pct']:.1f}%")
    print("-" * 95)
    print("PARETO CONCENTRATION LAW (Slide 14):")
    print(f"  • Top 1% of Customers generate:  {metrics['pareto_metrics']['top_1pct_revenue_share']:.1f}% of Revenue")
    print(f"  • Top 5% of Customers generate:  {metrics['pareto_metrics']['top_5pct_revenue_share']:.1f}% of Revenue")
    print(f"  • Top 10% of Customers generate: {metrics['pareto_metrics']['top_10pct_revenue_share']:.1f}% of Revenue")
    print(f"  • Top 20% of Customers generate: {metrics['pareto_metrics']['top_20pct_revenue_share']:.1f}% of Revenue (Classic 80/20 Rule)")
    print("=" * 95)
    print(f"EDA visual artifacts saved to {FIGURES_DIR}")
    print(f"EDA tabular summaries saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
