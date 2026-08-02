# Case Study 1: Cafe RFM Decisions

## Purpose

Create a new dataset for the slide's Manual Cafe Transaction Log. The existing 200-customer Mall Customers file is not an input to this case.

## Locked decisions

- Generate exactly 150 loyalty customers (`CUST-001` through `CUST-150`) across the four slide archetypes.
- Use calendar year 2025 with a generator reference date of 2026-01-01 and `RANDOM_SEED = 42`.
- Store one row per cafe visit. A visit has a unique transaction ID, customer ID, timestamp, item basket, and basket spend.
- Derive spend only from the fixed menu in `config.py`; never sample or hardcode reported totals.
- Keep source archetypes in a separate validation CSV. The analysis cannot read that file until after clustering is complete.
- Calculate recency from the day after the latest observed visit, frequency as distinct transactions, and monetary value as total basket spend.
- Evaluate `k=2..8`. A candidate needs at least eight customers per cluster and mean stability ARI of 0.80. Select the eligible solution with the best silhouette score.
- Use log-transformed RFM plus `StandardScaler`; use a fixed seed and multiple K-Means initializations.

## Fail-loud guardrails

Generation stops for incorrect customer counts, duplicate transaction IDs, nulls, invalid timestamps, nonpositive spend, unknown menu items, price mismatches, or archetype frequency/recency outside its configured range. Analysis stops for schema drift, duplicate IDs, invalid values, missing customers, ineligible cluster solutions, or disagreement between transaction totals and derived RFM totals.

## Deliverables

- `data/cafe_transactions_example.csv`: analysis input with no archetype label.
- `data/cafe_customer_archetypes_validation.csv`: withheld simulation ground truth.
- `outputs/customer_rfm_segments.csv`, candidate metrics, profiles, validation cross-tab, and run metrics.
- `figures/`: business dashboard, cluster selection, RFM segment scatter, and standardized segment profile charts.
- `CAFE_RFM_REPORT.md`: executive interpretation written only after a successful run.
