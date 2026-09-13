# Week 06: Customer Lifetime Value (CLV) & Enterprise Valuation Architecture

## Overview & Curriculum Alignment
This folder contains the complete pedagogical and production deliverables for **Week 06: Customer Lifetime Value (CLV) Enterprise Upgrade** based on the lecture materials in `Slide/Week_06_CLV_Enterprise_Upgrade_20260801.pdf`.

## Repository Structure & Deliverables

```
w6_CustomerLifetimeValue/
├── plan.md                                    # Master architectural and pedagogical plan
├── WEEK6_CLV_ENTERPRISE_REPORT.md             # Master executive report synthesizing Case 1 & Case 2
│
├── case1_cafe_clv/                            # Case Study 1: Artisanal Cafe Ticket Log Simulation (Slide 48)
│   ├── config.py                              # Business constants, discount rates, margins, paths
│   ├── plan.md                                # Case 1 analytical plan & guardrails
│   ├── generate.py                            # Data validation & RFM-T feature derivation
│   ├── analyze.py                             # Complete reproducible analysis & plotting script
│   ├── CAFE_CLV_REPORT.md                     # Executive summary report & top-3 customer memo
│   ├── data/
│   │   └── CLV_Retail_Transaction_example.csv # 10-customer baseline dataset
│   ├── figures/                               # Generated high-resolution .png visualization artifacts
│   │   ├── customer_clv_comparison.png
│   │   ├── expected_transactions_vs_historic.png
│   │   ├── retention_priority_matrix.png
│   │   └── rfm_distribution.png
│   └── outputs/                               # Model predictions, derived matrices, and metrics
│       ├── case1_metrics.json
│       ├── clv_model_predictions.csv
│       ├── derived_rfmt_matrix.csv
│       └── top3_customer_profiles.csv
│
└── case2_online_retail_clv/                   # Case Study 2: Enterprise Online Retail Mining Project (Slide 49)
    ├── config.py                              # Paths, URLs, and financial constants
    ├── plan.md                                # Case 2 CRISP-DM plan & hygiene contract
    ├── generate_smoke_data.py                 # Fast synthetic smoke dataset generator for CI
    ├── ingest.py                              # Dataset downloader & 3-stage data hygiene enforcement
    ├── build_features.py                      # Transaction aggregation to RFM-T grid
    ├── clv_pipeline.py                        # Production ML pipeline (BG/NBD + Gamma-Gamma + DCF)
    ├── ONLINE_RETAIL_CLV_REPORT.md            # Enterprise C-Suite executive report
    ├── data/
    │   ├── Online_Retail.csv                  # 541k-row enterprise transactional dataset
    │   └── Online_Retail_smoke.csv            # 1.5k-row smoke test dataset
    ├── figures/                               # 5 publication-ready diagnostic charts
    │   ├── clv_valuation_by_segment.png
    │   ├── data_hygiene_funnel.png
    │   ├── expected_transactions_distribution.png
    │   ├── recency_frequency_churn_cliff.png
    │   └── risk_value_matrix_dashboard.png
    └── outputs/                               # Production output tables and JSON reports
        ├── clean_transactions.csv
        ├── clv_scored_customers.csv
        ├── data_hygiene_report.json
        ├── model_metrics.json
        ├── rfmt_summary.csv
        └── segment_financial_impact.csv
```

## Running the Complete Case Studies

```bash
# Run Case Study 1 (Artisanal Cafe)
cd w6_CustomerLifetimeValue/case1_cafe_clv
python3 generate.py && python3 analyze.py

# Run Case Study 2 (Enterprise Online Retail Mining)
cd ../case2_online_retail_clv
python3 ingest.py && python3 build_features.py && python3 clv_pipeline.py
```
