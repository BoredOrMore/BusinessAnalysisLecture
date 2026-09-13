"""Financial Run-Rate Analysis & Distortion Diagnostics module for Case Study 2.

Evaluates:
1. Annualized Run-Rate Trajectories (Monthly & Rolling QTR vs Actual GMV).
2. Seasonality Distortions (Q4 Peak vs Q1 Trough over/under-estimation).
3. One-Time Anomaly & Whale Distortions (Impact of single bulk orders).
4. Ignored Churn Distortions (Naive Linear Run-Rate vs Probabilistic BG/NBD CLV).
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
    SCORED_CUSTOMERS_PATH,
)


def set_plotting_theme() -> None:
    """Apply clean publication-ready plotting style."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300


def analyze_financial_run_rate(
    clean_path: Path = CLEAN_DATA_PATH,
    scored_path: Path = SCORED_CUSTOMERS_PATH,
) -> dict:
    """Run empirical financial run-rate analysis and generate diagnostic charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    set_plotting_theme()

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset missing at {clean_path}. Run ingest.py first.")

    df = pd.read_csv(clean_path, dtype={"CustomerID": str})
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["YearMonth_str"] = df["YearMonth"].astype(str)

    actual_annual_gmv = float(df["TotalSpend"].sum())

    # =========================================================================
    # 1. Seasonality & Monthly Annualized Run-Rates
    # =========================================================================
    monthly = df.groupby("YearMonth_str").agg(
        monthly_gmv=("TotalSpend", "sum"),
        invoice_count=("InvoiceNo", "nunique"),
        active_customers=("CustomerID", "nunique"),
    ).reset_index()

    # Annualized Run Rate = Monthly GMV * 12
    monthly["annualized_run_rate"] = monthly["monthly_gmv"] * 12.0
    monthly["distortion_vs_actual_pct"] = (
        (monthly["annualized_run_rate"] - actual_annual_gmv) / actual_annual_gmv
    ) * 100.0

    peak_month = monthly.loc[monthly["monthly_gmv"].idxmax()]
    trough_month = monthly.loc[monthly["monthly_gmv"].idxmin()]

    monthly.to_csv(OUTPUTS_DIR / "financial_run_rate_monthly_comparison.csv", index=False)

    # =========================================================================
    # 2. One-Time Anomaly & Whale Bulk Order Distortion
    # =========================================================================
    invoice_totals = df.groupby(["InvoiceNo", "CustomerID", "Country"]).agg(
        invoice_total=("TotalSpend", "sum"),
        items_count=("Quantity", "sum"),
        invoice_date=("InvoiceDate", "min"),
    ).reset_index().sort_values("invoice_total", ascending=False)

    top_10_invoices_total = float(invoice_totals.head(10)["invoice_total"].sum())
    top_1_invoice_total = float(invoice_totals.iloc[0]["invoice_total"])
    top_1_invoice_cust = str(invoice_totals.iloc[0]["CustomerID"])

    # Single-order anomaly customers (customers with only 1 huge transaction)
    cust_orders = df.groupby("CustomerID").agg(
        orders_count=("InvoiceNo", "nunique"),
        total_spend=("TotalSpend", "sum"),
    ).reset_index()
    one_order_whales = cust_orders[(cust_orders["orders_count"] == 1) & (cust_orders["total_spend"] > 5000.0)]
    one_order_whales_total = float(one_order_whales["total_spend"].sum())

    anomaly_impact = {
        "top_1_invoice_spend": top_1_invoice_total,
        "top_1_invoice_annualized_distortion": top_1_invoice_total * 12.0,
        "top_10_invoices_spend": top_10_invoices_total,
        "top_10_invoices_share_pct": (top_10_invoices_total / actual_annual_gmv) * 100.0,
        "one_time_whale_accounts_count": len(one_order_whales),
        "one_time_whale_total_spend": one_order_whales_total,
    }

    invoice_totals.head(20).to_csv(OUTPUTS_DIR / "run_rate_top_invoices_anomalies.csv", index=False)

    # =========================================================================
    # 3. Ignored Churn Distortion: Naive Run-Rate vs Probabilistic BG/NBD CLV
    # =========================================================================
    scored_df = pd.read_csv(scored_path)
    # Naive annualized margin = Historical spend * 30% margin * (365 / Tenure)
    scored_df["tenure_safe"] = np.maximum(scored_df["T"], 30.0)
    scored_df["naive_annualized_margin"] = (scored_df["total_historical_spend"] / scored_df["tenure_safe"]) * 365.0 * 0.30
    scored_df["probabilistic_clv_12m"] = scored_df["pred_clv_12m"]

    # Filter out extreme new-user tenure explosion for fair decile comparison
    scored_df["naive_annualized_margin_capped"] = np.minimum(scored_df["naive_annualized_margin"], scored_df["total_historical_spend"] * 2.0 * 0.30)

    # Bin into P(Alive) buckets to show churn distortion
    scored_df["p_alive_bucket"] = pd.cut(
        scored_df["p_alive"] * 100.0,
        bins=[0, 50, 75, 90, 98, 100.1],
        labels=["Critical Churn (<50%)", "High Churn (50-75%)", "Moderate (75-90%)", "Healthy (90-98%)", "Active (>98%)"],
    )

    churn_comparison = scored_df.groupby("p_alive_bucket", observed=False).agg(
        customer_count=("CustomerID", "count"),
        naive_margin_sum=("naive_annualized_margin_capped", "sum"),
        probabilistic_clv_sum=("probabilistic_clv_12m", "sum"),
        mean_p_alive=("p_alive", "mean"),
    ).reset_index()
    churn_comparison["churn_distortion_gap"] = churn_comparison["naive_margin_sum"] - churn_comparison["probabilistic_clv_sum"]
    churn_comparison.to_csv(OUTPUTS_DIR / "run_rate_churn_distortion.csv", index=False)

    # Master metrics dictionary
    run_rate_metrics = {
        "actual_annual_gmv": actual_annual_gmv,
        "peak_month": {
            "month": str(peak_month["YearMonth_str"]),
            "monthly_gmv": float(peak_month["monthly_gmv"]),
            "annualized_run_rate": float(peak_month["annualized_run_rate"]),
            "overestimation_vs_actual_pct": float(peak_month["distortion_vs_actual_pct"]),
        },
        "trough_month": {
            "month": str(trough_month["YearMonth_str"]),
            "monthly_gmv": float(trough_month["monthly_gmv"]),
            "annualized_run_rate": float(trough_month["annualized_run_rate"]),
            "underestimation_vs_actual_pct": float(trough_month["distortion_vs_actual_pct"]),
        },
        "anomaly_diagnostics": anomaly_impact,
        "total_naive_extrapolated_margin": float(scored_df["naive_annualized_margin_capped"].sum()),
        "total_probabilistic_12m_clv": float(scored_df["probabilistic_clv_12m"].sum()),
        "overall_churn_phantom_gap": float(scored_df["naive_annualized_margin_capped"].sum() - scored_df["probabilistic_clv_12m"].sum()),
    }

    with open(OUTPUTS_DIR / "run_rate_distortion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(run_rate_metrics, f, indent=2)

    # =========================================================================
    # Visualizations
    # =========================================================================

    # Figure 1: Seasonality Run-Rate Distortion
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_coords = np.arange(len(monthly))
    
    # Bars for annualized run rate
    bars = ax.bar(x_coords, monthly["annualized_run_rate"] / 1e6, width=0.55, color="#4C72B0", alpha=0.85, label="Annualized Monthly Run-Rate ($M)")
    
    # Reference lines for Actual Annual GMV
    ax.axhline(actual_annual_gmv / 1e6, color="#2ca02c", linestyle="--", linewidth=2.2, label=f"Actual Full-Year GMV (${actual_annual_gmv/1e6:.2f}M)")
    
    # Annotations on Peak and Trough
    peak_idx = int(monthly["monthly_gmv"].argmax())
    trough_idx = int(monthly["monthly_gmv"].argmin())
    
    ax.annotate(
        f"Peak Q4 Run-Rate: ${monthly.loc[peak_idx, 'annualized_run_rate']/1e6:.2f}M\n(+{monthly.loc[peak_idx, 'distortion_vs_actual_pct']:.1f}% Overestimation)",
        (peak_idx, monthly.loc[peak_idx, "annualized_run_rate"] / 1e6),
        xytext=(peak_idx - 3.2, (monthly.loc[peak_idx, "annualized_run_rate"] / 1e6) + 0.8),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d62728"),
    )
    
    ax.annotate(
        f"Q1 Trough Run-Rate: ${monthly.loc[trough_idx, 'annualized_run_rate']/1e6:.2f}M\n({monthly.loc[trough_idx, 'distortion_vs_actual_pct']:.1f}% Underestimation)",
        (trough_idx, monthly.loc[trough_idx, "annualized_run_rate"] / 1e6),
        xytext=(trough_idx + 0.4, (monthly.loc[trough_idx, "annualized_run_rate"] / 1e6) - 2.5),
        arrowprops=dict(facecolor="#7f7f7f", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#424242",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#9e9e9e"),
    )

    ax.set_xticks(x_coords)
    ax.set_xticklabels(monthly["YearMonth_str"], rotation=25, ha="right", fontsize=9.5)
    ax.set_ylabel("Projected Annual Revenue ($M)", fontsize=11, fontweight="bold")
    ax.set_title("Financial Run-Rate Seasonality Distortion: Monthly Projections vs Actual Annual GMV", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.set_ylim(0, max(monthly["annualized_run_rate"] / 1e6) * 1.25)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "financial_run_rate_seasonality_distortion.png", dpi=300)
    plt.close()

    # Figure 2: One-Time Whale & Outlier Distortion
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.2))
    
    # Left: Top 10 Invoices Share
    invoices_top10 = invoice_totals.head(10).copy()
    invoices_top10["inv_label"] = [f"Inv #{row['InvoiceNo']}\n(Cust {str(row['CustomerID']) if pd.notna(row['CustomerID']) else 'N/A'})" for _, row in invoices_top10.iterrows()]
    
    ax_left.barh(invoices_top10["inv_label"][::-1], invoices_top10["invoice_total"][::-1] / 1000.0, color="#C44E52", height=0.6)
    for i, val in enumerate(invoices_top10["invoice_total"][::-1] / 1000.0):
        ax_left.text(val + 2, i - 0.15, f"${val:,.1f}k", fontsize=8.5, fontweight="semibold")
    ax_left.set_title("Top 10 Outlier Single Transactions ($k)", fontsize=11.5, fontweight="bold")
    ax_left.set_xlabel("Single Transaction GMV ($k)")
    ax_left.set_xlim(0, max(invoices_top10["invoice_total"] / 1000.0) * 1.2)

    # Right: Monthly Revenue With vs Without Outliers
    top10_ids = invoice_totals.head(10)["InvoiceNo"].tolist()
    df_no_outliers = df[~df["InvoiceNo"].isin(top10_ids)]
    monthly_no_out = df_no_outliers.groupby("YearMonth_str")["TotalSpend"].sum().reset_index()

    w = 0.38
    x_c = np.arange(len(monthly))
    ax_right.bar(x_c - w/2, monthly["monthly_gmv"] / 1000.0, width=w, label="Full Transaction Base", color="#4C72B0", alpha=0.9)
    ax_right.bar(x_c + w/2, monthly_no_out["TotalSpend"] / 1000.0, width=w, label="Excluding Top 10 Outliers", color="#55A868", alpha=0.9)
    ax_right.set_xticks(x_c)
    ax_right.set_xticklabels(monthly["YearMonth_str"], rotation=30, ha="right", fontsize=9)
    ax_right.set_ylabel("Monthly Revenue ($k)", fontsize=10.5)
    ax_right.set_title("Monthly Revenue: Base vs Outlier-Sanitized", fontsize=11.5, fontweight="bold")
    ax_right.legend(frameon=True, fontsize=9)

    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "run_rate_outlier_whale_impact.png", dpi=300)
    plt.close()

    # Figure 3: Churn Distortion (Naive Run-Rate vs Probabilistic CLV)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Compare by deciles of customer spend
    scored_df["clv_decile"] = pd.qcut(scored_df["total_historical_spend"], 10, labels=[f"D{i+1}" for i in range(10)])
    decile_comp = scored_df.groupby("clv_decile", observed=False).agg(
        naive_margin=("naive_annualized_margin_capped", "sum"),
        probabilistic_clv=("probabilistic_clv_12m", "sum"),
    ).reset_index()

    x_d = np.arange(len(decile_comp))
    ax.bar(x_d - w/2, decile_comp["naive_margin"] / 1000.0, width=w, label="Naive Static Run-Rate Margin (Ignored Churn)", color="#d62728", alpha=0.85)
    ax.bar(x_d + w/2, decile_comp["probabilistic_clv"] / 1000.0, width=w, label="BG/NBD Probabilistic 12M CLV (Churn-Adjusted)", color="#2ca02c", alpha=0.85)
    
    ax.set_xticks(x_d)
    ax.set_xticklabels(decile_comp["clv_decile"], fontsize=10)
    ax.set_xlabel("Historical Customer Spend Decile (D1=Lowest, D10=Highest/VIP)", fontsize=11, fontweight="bold")
    ax.set_ylabel("12-Month Projected Net Margin ($k)", fontsize=11, fontweight="bold")
    ax.set_title("Ignored Churn Distortion: Naive Run-Rate Overestimation across Deciles", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "run_rate_vs_churn_discounted_comparison.png", dpi=300)
    plt.close()

    return run_rate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze financial run rate distortions and impacts.")
    parser.parse_args()

    metrics = analyze_financial_run_rate()
    print("=" * 95)
    print("            FINANCIAL RUN-RATE DISTORTION & IMPACT DIAGNOSTIC REPORT            ")
    print("=" * 95)
    print(f"Actual Full-Year Enterprise GMV:        ${metrics['actual_annual_gmv']:,.2f}")
    print("-" * 95)
    print("1. SEASONALITY DISTORTION (Peak vs Trough):")
    print(f"  • Peak Month ({metrics['peak_month']['month']}):       ${metrics['peak_month']['monthly_gmv']:,.2f} -> Run-Rate: ${metrics['peak_month']['annualized_run_rate']:,.2f} (+{metrics['peak_month']['overestimation_vs_actual_pct']:.1f}% Overestimation)")
    print(f"  • Trough Month ({metrics['trough_month']['month']}):     ${metrics['trough_month']['monthly_gmv']:,.2f} -> Run-Rate: ${metrics['trough_month']['annualized_run_rate']:,.2f} ({metrics['trough_month']['underestimation_vs_actual_pct']:.1f}% Underestimation)")
    print("-" * 95)
    print("2. ONE-TIME ANOMALY / WHALE ORDER DISTORTION:")
    print(f"  • Largest Single Invoice:             ${metrics['anomaly_diagnostics']['top_1_invoice_spend']:,.2f} (Annualizing this single order alone = ${metrics['anomaly_diagnostics']['top_1_invoice_annualized_distortion']:,.2f})")
    print(f"  • Top 10 Bulk Invoices Total:         ${metrics['anomaly_diagnostics']['top_10_invoices_spend']:,.2f} ({metrics['anomaly_diagnostics']['top_10_invoices_share_pct']:.1f}% of entire annual GMV)")
    print(f"  • Single-Purchase Whale Accounts:     {metrics['anomaly_diagnostics']['one_time_whale_accounts_count']} accounts totaling ${metrics['anomaly_diagnostics']['one_time_whale_total_spend']:,.2f}")
    print("-" * 95)
    print("3. IGNORED CHURN DISTORTION (Static Run-Rate vs Probabilistic CLV):")
    print(f"  • Naive Static Extrapolated Margin:   ${metrics['total_naive_extrapolated_margin']:,.2f}")
    print(f"  • BG/NBD Probabilistic Discounted CLV:${metrics['total_probabilistic_12m_clv']:,.2f}")
    print(f"  • Phantom Margin Run-Rate Gap:        ${metrics['overall_churn_phantom_gap']:,.2f} (Phantom profit eliminated by BG/NBD)")
    print("=" * 95)
    print(f"Saved diagnostic charts to: {FIGURES_DIR}")
    print(f"Saved tabular summaries to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
