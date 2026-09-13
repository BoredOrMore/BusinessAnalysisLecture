# Analytical Plan: Case Study 1 — Artisanal Cafe Ticket Log Simulation

## 1. Business Context & Objective
- **Stakeholders**: CEO, CMO, and Chief Revenue Officer of Artisanal Cafe.
- **Problem Statement**: Traditional historical CLV heuristics overvalue dormant accounts and fail to capture customer drop-off dynamics. We replace naive extrapolations with probabilistic BG/NBD + Gamma-Gamma valuation models.
- **Core Tasks**:
  1. Compute baseline RFM-T matrices from ticket logs.
  2. Compare Historic CLV vs BG/NBD + Gamma-Gamma 12-Month Net Discounted Margin CLV ($).
  3. Formulate targeted retention strategy recommendations for the top 3 customer profiles (`CUST_1003`, `CUST_1005`, `CUST_1009`) using Type 1 vs Type 2 decision frameworks.

## 2. Mathematical Formulations & Data Contract
- **Derived RFM-T Features**:
  - Repeat Frequency: $x = \text{Frequency\_Orders} - 1$
  - Recency: $x_{rec} = \text{Tenure\_Days} - \text{Recency\_Days}$ (time from first to last purchase)
  - Tenure: $T = \text{Tenure\_Days}$
  - Monetary Value: $m_x = \text{Avg\_Order\_Value}$
- **Probabilistic Valuation**:
  - Probability Active: $P(\text{Alive} \mid x, x_{rec}, T)$
  - Expected Future Purchases: $E[X(365) \mid x, x_{rec}, T]$
  - 12-Month Discounted CLV:
    $$\text{CLV}_{12M} = E[X(365)] \times E[M] \times \text{Gross Margin \%} \times \frac{1}{(1 + d)}$$
    where $\text{Gross Margin} = 30\%$, $d = 0.83\%/\text{month}$ ($10\%/\text{year}$).

## 3. Anti-Hallucination Guardrails
- Fixed seed `RANDOM_SEED = 42`.
- Zero hardcoded report numbers: all values populated directly from script output.
- Exact reproducibility assertions on derived vectors and bounds.
