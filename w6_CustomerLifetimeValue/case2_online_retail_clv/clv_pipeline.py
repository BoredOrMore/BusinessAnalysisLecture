"""Production CLV Machine Learning Pipeline for Case Study 2 (Online Retail).

Fits BG/NBD and Gamma-Gamma probabilistic models, infers latent defection risks,
computes 12-month net discounted margin valuations, assigns strategic action tiers,
and generates executive figures and financial summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import gammaln

from build_features import build_rfmt_summary
from config import (
    ANNUAL_DISCOUNT_RATE,
    BGNBD_PENALIZER,
    CLEAN_DATA_PATH,
    FIGURES_DIR,
    GG_PENALIZER,
    GROSS_MARGIN_PCT,
    HORIZON_DAYS,
    HYGIENE_REPORT_PATH,
    METRICS_PATH,
    MONTHLY_DISCOUNT_RATE,
    OUTPUTS_DIR,
    RANDOM_SEED,
    RFMT_SUMMARY_PATH,
    SCORED_CUSTOMERS_PATH,
    SEGMENT_IMPACT_PATH,
)
from ingest import apply_data_hygiene, ensure_dataset


def set_plotting_theme() -> None:
    """Configure clean scientific visualization styling."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300


# ==============================================================================
# BG/NBD Latent Defection Model (Fader, Hardie, Lee 2005)
# ==============================================================================

def bgnbd_negative_log_likelihood(
    log_params: np.ndarray,
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    penalizer: float = BGNBD_PENALIZER,
) -> float:
    """Compute negative log-likelihood for BG/NBD parameter optimization."""
    r, alpha, a, b = np.exp(log_params)
    x = frequency
    t_x = recency

    log_beta_ab = gammaln(a) + gammaln(b) - gammaln(a + b)
    log_beta_a_bx = gammaln(a) + gammaln(b + x) - gammaln(a + b + x)

    ln_a1 = r * np.log(alpha) + gammaln(r + x) - gammaln(r) - (r + x) * np.log(alpha + T) + (log_beta_a_bx - log_beta_ab)
    log_diff = (r + x) * (np.log(alpha + T) - np.log(alpha + t_x + 1e-12))
    log_ratio = np.log(a) - np.log(b + x - 1.0 + 1e-12) + log_diff
    term2 = np.where(x > 0, np.exp(np.clip(log_ratio, -50.0, 50.0)), 0.0)

    ll = ln_a1 + np.log(1.0 + term2)
    return float(-np.sum(ll) + penalizer * np.sum(np.exp(log_params)))


def fit_bgnbd(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    penalizer: float = BGNBD_PENALIZER,
) -> tuple[float, float, float, float]:
    """Fit BG/NBD model parameters (r, alpha, a, b) via MLE."""
    init_params = np.log([0.5, 10.0, 0.5, 1.0])
    bounds = [(-5.0, 5.0), (-5.0, 7.0), (-5.0, 5.0), (-5.0, 7.0)]
    res = minimize(
        bgnbd_negative_log_likelihood,
        init_params,
        args=(frequency, recency, T, penalizer),
        method="L-BFGS-B",
        bounds=bounds,
    )
    r, alpha, a, b = np.exp(res.x)
    return float(r), float(alpha), float(a), float(b)


def compute_p_alive_vector(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    r: float,
    alpha: float,
    a: float,
    b: float,
) -> np.ndarray:
    """Compute customer-level Probability Active P(Alive)."""
    term = (a / (b + frequency)) * ((alpha + T) / (alpha + recency + 1e-12)) ** (r + frequency)
    return np.where(frequency == 0, 1.0, 1.0 / (1.0 + term))


def compute_expected_transactions_vector(
    frequency: np.ndarray,
    recency: np.ndarray,
    T: np.ndarray,
    p_alive: np.ndarray,
    t_days: float,
    r: float,
    alpha: float,
) -> np.ndarray:
    """Compute expected transactions in future window t_days."""
    rate = (r + frequency) / (alpha + T)
    return rate * t_days * p_alive


# ==============================================================================
# Gamma-Gamma Monetary Spend Model
# ==============================================================================

def gg_negative_log_likelihood(
    log_params: np.ndarray,
    frequency: np.ndarray,
    monetary_value: np.ndarray,
    penalizer: float = GG_PENALIZER,
) -> float:
    """Compute negative log-likelihood for Gamma-Gamma spend model."""
    p, q, v = np.exp(log_params)
    x = frequency
    m = monetary_value

    term1 = gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
    term2 = q * np.log(v) + (p * x - 1.0) * np.log(m) + p * x * np.log(x)
    term3 = (p * x + q) * np.log(x * m + v)
    ll = term1 + term2 - term3
    return float(-np.sum(ll) + penalizer * np.sum(np.exp(log_params)))


def fit_gamma_gamma(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
    penalizer: float = GG_PENALIZER,
) -> tuple[float, float, float]:
    """Fit Gamma-Gamma spend parameters (p, q, v)."""
    init_params = np.log([2.0, 3.0, 100.0])
    bounds = [(-3.0, 4.0), (-3.0, 4.0), (-3.0, 7.0)]
    res = minimize(
        gg_negative_log_likelihood,
        init_params,
        args=(frequency, monetary_value, penalizer),
        method="L-BFGS-B",
        bounds=bounds,
    )
    p, q, v = np.exp(res.x)
    return float(p), float(q), float(v)


def compute_expected_spend_vector(
    frequency: np.ndarray,
    monetary_value: np.ndarray,
    p: float,
    q: float,
    v: float,
    global_mean: float,
) -> np.ndarray:
    """Infer individual expected monetary spend with Bayesian shrinkage."""
    denom = p * frequency + q - 1.0
    prior_mean = p * v / (q - 1.0 + 1e-12) if q > 1.0 else global_mean
    prior_weight = (q - 1.0) / (denom + 1e-12)
    obs_weight = (p * frequency) / (denom + 1e-12)

    exp_spend = prior_weight * prior_mean + obs_weight * monetary_value
    # For one-time buyers, shrink heavily towards empirical prior
    return np.where(frequency > 0, exp_spend, prior_mean)


# ==============================================================================
# End-to-End Pipeline Execution
# ==============================================================================

def run_pipeline(
    use_smoke: bool = False,
    gross_margin: float = GROSS_MARGIN_PCT,
    discount_rate: float = MONTHLY_DISCOUNT_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute complete end-to-end CLV mining and valuation workflow."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion & Hygiene
    raw_df = ensure_dataset(use_smoke=use_smoke)
    clean_df, hygiene_report = apply_data_hygiene(raw_df)

    # 2. RFM-T Feature Extraction
    rfmt = build_rfmt_summary(clean_df)
    rfmt["CustomerID"] = rfmt["CustomerID"].astype(str)

    # 3. Fit BG/NBD
    freq_arr = rfmt["frequency"].values
    rec_arr = rfmt["recency"].values
    T_arr = rfmt["T"].values

    r_fit, alpha_fit, a_fit, b_fit = fit_bgnbd(freq_arr, rec_arr, T_arr)

    # Compute P(Alive) and Expected Purchases
    rfmt["p_alive"] = compute_p_alive_vector(freq_arr, rec_arr, T_arr, r_fit, alpha_fit, a_fit, b_fit)
    rfmt["exp_12m_orders"] = compute_expected_transactions_vector(
        freq_arr, rec_arr, T_arr, rfmt["p_alive"].values, float(HORIZON_DAYS), r_fit, alpha_fit
    )

    # 4. Fit Gamma-Gamma on repeat buyers
    repeat_mask = (rfmt["frequency"] > 0) & (rfmt["monetary_value"] > 0)
    p_fit, q_fit, v_fit = fit_gamma_gamma(
        rfmt.loc[repeat_mask, "frequency"].values,
        rfmt.loc[repeat_mask, "monetary_value"].values,
    )

    global_mean_spend = float(rfmt.loc[repeat_mask, "monetary_value"].mean())
    rfmt["exp_monetary_value"] = compute_expected_spend_vector(
        rfmt["frequency"].values,
        rfmt["monetary_value"].values,
        p_fit,
        q_fit,
        v_fit,
        global_mean_spend,
    )

    # 5. Discounted Cash Flow Valuation (Slide 39)
    dcf_factor = 1.0 / (1.0 + discount_rate * 12.0)
    rfmt["pred_gross_revenue_12m"] = rfmt["exp_12m_orders"] * rfmt["exp_monetary_value"]
    rfmt["pred_clv_12m"] = rfmt["pred_gross_revenue_12m"] * gross_margin * dcf_factor

    # 6. Strategic Action Segmentation
    clv_p75 = float(rfmt["pred_clv_12m"].quantile(0.75))
    clv_p25 = float(rfmt["pred_clv_12m"].quantile(0.25))

    def assign_segment(row: pd.Series) -> str:
        clv = row["pred_clv_12m"]
        pa = row["p_alive"]
        if clv >= clv_p75 and pa >= 0.65:
            return "1. VIP Champions"
        elif clv >= clv_p75 and pa < 0.65:
            return "2. High-LTV At-Risk (Defection Cliff)"
        elif clv < clv_p75 and pa >= 0.65:
            return "3. Emerging Loyalists"
        else:
            return "4. Dormant / Low-Value"

    rfmt["Strategic_Segment"] = rfmt.apply(assign_segment, axis=1)

    # Segment Financial Impact Aggregations
    segment_summary = rfmt.groupby("Strategic_Segment").agg(
        customer_count=("CustomerID", "count"),
        total_predicted_clv=("pred_clv_12m", "sum"),
        mean_clv=("pred_clv_12m", "mean"),
        mean_p_alive=("p_alive", "mean"),
        mean_expected_orders=("exp_12m_orders", "mean"),
        mean_order_spend=("exp_monetary_value", "mean"),
    ).reset_index()

    total_clv = float(rfmt["pred_clv_12m"].sum())
    segment_summary["clv_share_pct"] = (segment_summary["total_predicted_clv"] / total_clv) * 100.0
    segment_summary["customer_share_pct"] = (segment_summary["customer_count"] / len(rfmt)) * 100.0

    # Save data artifacts
    rfmt.to_csv(SCORED_CUSTOMERS_PATH, index=False)
    segment_summary.to_csv(SEGMENT_IMPACT_PATH, index=False)

    # Metrics JSON
    metrics = {
        "dataset_mode": "smoke" if use_smoke else "full_production",
        "total_unique_customers": int(len(rfmt)),
        "total_historical_spend": float(rfmt["total_historical_spend"].sum()),
        "total_predicted_12m_net_clv": total_clv,
        "mean_customer_12m_clv": float(rfmt["pred_clv_12m"].mean()),
        "median_customer_12m_clv": float(rfmt["pred_clv_12m"].median()),
        "overall_mean_p_alive": float(rfmt["p_alive"].mean()),
        "at_risk_high_ltv_count": int((rfmt["Strategic_Segment"] == "2. High-LTV At-Risk (Defection Cliff)").sum()),
        "at_risk_high_ltv_clv": float(
            rfmt.loc[rfmt["Strategic_Segment"] == "2. High-LTV At-Risk (Defection Cliff)", "pred_clv_12m"].sum()
        ),
        "fitted_bgnbd": {"r": r_fit, "alpha": alpha_fit, "a": a_fit, "b": b_fit},
        "fitted_gamma_gamma": {"p": p_fit, "q": q_fit, "v": v_fit},
        "assumptions": {
            "gross_margin_pct": gross_margin,
            "annual_discount_rate": ANNUAL_DISCOUNT_RATE,
            "monthly_discount_rate": discount_rate,
            "horizon_days": HORIZON_DAYS,
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 7. Generate Visualizations
    set_plotting_theme()

    # Figure 1: Data Hygiene Funnel
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages = ["1. Raw Invoices", "2. Registered Only\n(Rule 1)", "3. Positive Qty\n(Rule 2)", "4. Valid Price\n(Rule 3)"]
    row_counts = [
        hygiene_report["raw_row_count"],
        hygiene_report["raw_row_count"] - hygiene_report["rule_01_null_customer_dropped"],
        hygiene_report["raw_row_count"] - hygiene_report["rule_01_null_customer_dropped"] - hygiene_report["rule_02_negative_qty_dropped"],
        hygiene_report["clean_row_count"],
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    bars = ax.barh(stages[::-1], row_counts[::-1], color=colors[::-1], height=0.55)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5000, bar.get_y() + 0.18, f"{int(w):,} rows ({w/hygiene_report['raw_row_count']:.1%})", fontsize=10, fontweight="semibold")
    ax.set_title("Enterprise Data Hygiene Filtration Funnel (Slides 33-34)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Transaction Row Count", fontsize=11)
    ax.set_xlim(0, max(row_counts) * 1.35)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "data_hygiene_funnel.png", dpi=300)
    plt.close()

    # Figure 2: Recency vs Frequency Churn Cliff
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        rfmt["recency"],
        rfmt["frequency"],
        c=rfmt["p_alive"] * 100.0,
        cmap="Spectral",
        s=35,
        alpha=0.8,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Probability Active P(Alive) (%)", fontsize=11, fontweight="bold")
    ax.set_title("Latent Defection Cliff: Recency vs Repeat Frequency", fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer Recency (Days between First & Last Order)", fontsize=11)
    ax.set_ylabel("Repeat Frequency (Order Count - 1)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recency_frequency_churn_cliff.png", dpi=300)
    plt.close()

    # Figure 3: Expected Transactions Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(rfmt["exp_12m_orders"], bins=40, kde=True, color="#4C72B0", ax=ax)
    ax.set_title("Distribution of BG/NBD Expected 12-Month Orders E[X]", fontsize=13, fontweight="bold")
    ax.set_xlabel("Expected Orders in Next 365 Days", fontsize=11)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.set_xlim(0, float(rfmt["exp_12m_orders"].quantile(0.99)))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "expected_transactions_distribution.png", dpi=300)
    plt.close()

    # Figure 4: CLV Valuation by Strategic Segment
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        segment_summary["Strategic_Segment"],
        segment_summary["total_predicted_clv"] / 1000.0,
        color=["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"],
        width=0.55,
        edgecolor="black",
    )
    for bar, pct in zip(bars, segment_summary["clv_share_pct"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f"${h:,.1f}k\n({pct:.1f}%)", ha="center", fontsize=9.5, fontweight="semibold")
    ax.set_title("12-Month Net Discounted CLV by Strategic Action Tier", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Predicted Forward Margin ($k)", fontsize=11)
    ax.set_xticks(range(len(segment_summary)))
    ax.set_xticklabels(segment_summary["Strategic_Segment"], rotation=15, ha="right", fontsize=9.5)
    ax.set_ylim(0, max(segment_summary["total_predicted_clv"] / 1000.0) * 1.25)
    plt.tight_layout(pad=1.5)
    plt.savefig(FIGURES_DIR / "clv_valuation_by_segment.png", dpi=300)
    plt.close()

    # Figure 5: Risk-Value Dashboard (P(Alive) vs CLV)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for seg_name, color in zip(
        ["1. VIP Champions", "2. High-LTV At-Risk (Defection Cliff)", "3. Emerging Loyalists", "4. Dormant / Low-Value"],
        ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"],
    ):
        seg_data = rfmt[rfmt["Strategic_Segment"] == seg_name]
        ax.scatter(
            seg_data["p_alive"] * 100.0,
            seg_data["pred_clv_12m"],
            s=40,
            alpha=0.65,
            label=f"{seg_name} (n={len(seg_data):,})",
            color=color,
            edgecolors="none",
        )
    ax.axvline(65.0, color="black", linestyle="--", alpha=0.6, label="Churn Boundary (P(Alive)=65%)")
    ax.axhline(clv_p75, color="black", linestyle=":", alpha=0.6, label=f"High-LTV 75th %ile (${clv_p75:,.0f})")
    ax.set_title("Enterprise Customer Risk-Value Matrix: P(Alive) vs 12-Month CLV", fontsize=13, fontweight="bold")
    ax.set_xlabel("Probability Active P(Alive) (%)", fontsize=11)
    ax.set_ylabel("12-Month Net Discounted Margin CLV ($)", fontsize=11)
    ax.set_ylim(0, float(rfmt["pred_clv_12m"].quantile(0.995)) * 1.1)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "risk_value_matrix_dashboard.png", dpi=300)
    plt.close()

    return rfmt, segment_summary, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute production CLV ML pipeline for Online Retail.")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic smoke dataset for fast testing")
    parser.add_argument("--gross_margin", type=float, default=GROSS_MARGIN_PCT, help="Gross profit margin percentage (default: 0.30)")
    parser.add_argument("--discount_rate", type=float, default=MONTHLY_DISCOUNT_RATE, help="Monthly discount rate (default: 0.0083)")
    args = parser.parse_args()

    rfmt, segment_summary, metrics = run_pipeline(
        use_smoke=args.smoke,
        gross_margin=args.gross_margin,
        discount_rate=args.discount_rate,
    )

    print("=" * 95)
    print("      ENTERPRISE ONLINE RETAIL CLV PIPELINE & FINANCIAL IMPACT REPORT      ")
    print("=" * 95)
    print(f"Total Customer Accounts Scored: {metrics['total_unique_customers']:,}")
    print(f"Total Historical Gross Revenue: ${metrics['total_historical_spend']:,.2f}")
    print(f"Total Predicted 12M Forward Net CLV: ${metrics['total_predicted_12m_net_clv']:,.2f}")
    print(f"Mean Customer Forward 12M CLV:  ${metrics['mean_customer_12m_clv']:,.2f}")
    print(f"Median Customer Forward 12M CLV:${metrics['median_customer_12m_clv']:,.2f}")
    print(f"Population Mean P(Alive):       {metrics['overall_mean_p_alive'] * 100.0:.1f}%")
    print("-" * 95)
    print("FITTED PARAMETERS:")
    print(f"  • BG/NBD:       r={metrics['fitted_bgnbd']['r']:.4f}, alpha={metrics['fitted_bgnbd']['alpha']:.4f}, a={metrics['fitted_bgnbd']['a']:.4f}, b={metrics['fitted_bgnbd']['b']:.4f}")
    print(f"  • Gamma-Gamma:  p={metrics['fitted_gamma_gamma']['p']:.4f}, q={metrics['fitted_gamma_gamma']['q']:.4f}, v={metrics['fitted_gamma_gamma']['v']:.4f}")
    print("=" * 95)
    print("STRATEGIC SEGMENT FINANCIAL IMPACT BREAKDOWN:")
    print("=" * 95)
    print(segment_summary.to_string(index=False))
    print("=" * 95)
    print(f"High-LTV At-Risk Capital under Defection Threat: ${metrics['at_risk_high_ltv_clv']:,.2f} ({metrics['at_risk_high_ltv_count']:,} accounts)")
    print(f"Saved pipeline outputs to {OUTPUTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
