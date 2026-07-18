#!/usr/bin/env python3
"""
Telco Customer Churn Statistical & Financial Impact Analysis
=============================================================
Prepared for: Chief Marketing Officer (CMO)
Business Context: Reduce overall customer churn from 26% to under 15% using targeted incentives.

Outputs:
1. Cross-tabulations of Churn vs. Contract, PaymentMethod, and PaperlessBilling.
2. Chi-Square tests of independence (p-values, degrees of freedom, test statistics).
3. Financial impact matrix showing EBITDA uplift of converting Month-to-Month customers to 1-Year contracts.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

def main():
    # Locate dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(csv_path):
        csv_path = "/Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')

    total_customers = len(df)
    overall_churners = (df['Churn'] == 'Yes').sum()
    overall_churn_rate = (overall_churners / total_customers) * 100

    print("=" * 80)
    print(f"TELCO CUSTOMER CHURN ANALYSIS REPORT (N = {total_customers:,})")
    print("=" * 80)
    print(f"Current Overall Churn Rate: {overall_churners:,} / {total_customers:,} ({overall_churn_rate:.2f}%)")
    print(f"CMO Target Churn Rate:      < 15.00% (Requires retaining >= {int(overall_churners - total_customers*0.15):,} additional customers)\n")

    # 1. Cross-Tabulations and Chi-Square Tests
    features = ['Contract', 'PaymentMethod', 'PaperlessBilling']

    for col in features:
        print("-" * 80)
        print(f"CROSS-TABULATION & CHI-SQUARE TEST: {col.upper()} vs. CHURN")
        print("-" * 80)

        ct_counts = pd.crosstab(df[col], df['Churn'], margins=True)
        ct_row_pct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100

        summary_table = pd.DataFrame({
            'Retained (No)': ct_counts['No'],
            'Churned (Yes)': ct_counts['Yes'],
            'Total Base': ct_counts['All'],
            'Churn Rate (%)': ct_row_pct['Yes'].round(2),
            'Retained Rate (%)': ct_row_pct['No'].round(2)
        })
        print(summary_table)
        print()

        # Chi-Square Test (excluding margins)
        ct_test = pd.crosstab(df[col], df['Churn'])
        chi2, p, dof, _ = stats.chi2_contingency(ct_test)
        print(f"Chi-Square Test Results:")
        print(f"  Chi-Square Statistic : {chi2:.4f}")
        print(f"  Degrees of Freedom   : {dof}")
        print(f"  p-value              : {p:.4e}")
        print(f"  Statistical Decision : {'Statistically Significant (p < 0.001)' if p < 0.001 else 'Not Significant'}")
        print()

    # 2. Financial Impact Matrix: Month-to-Month to 1-Year Contract Conversion
    print("=" * 80)
    print("FINANCIAL IMPACT MATRIX: EBITDA UPLIFT FROM M2M TO 1-YEAR CONVERSION")
    print("=" * 80)

    m2m = df[df['Contract'] == 'Month-to-month']
    one_yr = df[df['Contract'] == 'One year']

    m2m_count = len(m2m)
    m2m_churn_rate = (m2m['Churn'] == 'Yes').mean()
    m2m_avg_monthly = m2m['MonthlyCharges'].mean()
    m2m_arpu_annual = m2m_avg_monthly * 12

    one_yr_churn_rate = (one_yr['Churn'] == 'Yes').mean()
    churn_drop = m2m_churn_rate - one_yr_churn_rate

    print(f"Baseline Month-to-Month Base : {m2m_count:,} customers")
    print(f"M2M Churn Rate               : {m2m_churn_rate*100:.2f}% | Average Annual Revenue per User: ${m2m_arpu_annual:.2f}")
    print(f"1-Year Contract Churn Rate   : {one_yr_churn_rate*100:.2f}% | Expected Churn Reduction: {churn_drop*100:.2f} percentage points\n")

    conversion_tiers = [0.10, 0.20, 0.30, 0.40, 0.50, 0.667, 0.75, 1.00]
    ebitda_margins = [0.30, 0.35, 0.40, 0.45]
    discount_scenarios = [
        ("Scenario A: Organic / Bundled Features (0% Price Discount)", 0.00),
        ("Scenario B: 5% Monthly Incentive Discount ($3.32/mo)", 0.05),
        ("Scenario C: 10% Monthly Incentive Discount ($6.64/mo)", 0.10)
    ]

    for scenario_name, disc in discount_scenarios:
        print("\n" + scenario_name)
        header = f"{'Conv %':<8} | {'Converted #':<12} | {'New Churn %':<12} | {'Saved #':<10} | {'Retained Rev ($)':<18} | {'Incentive Cost ($)':<19}"
        for m in ebitda_margins:
            header += f" | {f'EBITDA @ {int(m*100)}%':<14}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for cr in conversion_tiers:
            converted_users = int(m2m_count * cr)
            saved_users = converted_users * churn_drop
            new_total_churners = overall_churners - saved_users
            new_overall_churn_rate = (new_total_churners / total_customers) * 100

            retained_annual_rev = saved_users * m2m_arpu_annual
            incentive_annual_cost = converted_users * (m2m_avg_monthly * disc * 12)

            row_str = f"{cr*100:6.1f}%  | {converted_users:<12,} | {new_overall_churn_rate:11.2f}% | {saved_users:<10.1f} | ${retained_annual_rev:<17,.2f} | ${incentive_annual_cost:<18,.2f}"
            for m in ebitda_margins:
                ebitda_uplift = (retained_annual_rev * m) - incentive_annual_cost
                row_str += f" | ${ebitda_uplift:<13,.2f}"
            print(row_str)
        print()

if __name__ == "__main__":
    main()
