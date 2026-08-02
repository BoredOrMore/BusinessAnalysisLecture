# Case Study 2: Locked Processing Decisions

## Scope

Segment purchasers from the Kaggle multi-category ecommerce event log. October 2019 passed the first production gate, so the final analysis combines October and November 2019. The unrelated 200-row Mall Customers file is not an input.

## Resource guardrails

- DuckDB: 10 GB memory, four threads, 35 GB maximum temporary storage.
- Warn at 12 GB process RSS or 60 GiB free disk; abort at 14 GB RSS or 50 GiB free disk.
- Warn below 4 GB system-available RAM and abort below 2 GB.
- Process one month at a time. Never materialize a plain multi-gigabyte CSV.
- Convert ZIP to gzip by streaming, then gzip to Zstandard Parquet with 250,000-row groups.

## Data and feature contract

Require the documented nine fields and reconcile all input/output rows. Fail on invalid timestamps, month leakage, missing core IDs/prices, negative prices, unknown event types, or purchase events without sessions. Report optional-field nulls and exact duplicates; never deduplicate purchase lines without an approved quantity rule.

Create one row per purchaser. Frequency is distinct purchase sessions, monetary is summed purchase-line price, and recency is measured from the day after the observation window. Add browsing, cart, activity, session, category-diversity, and cart-conversion features. Treat recorded price as an unspecified value unit rather than asserting a currency.

## Modeling contract

Fit scaling statistics across all purchasers in 100,000-row batches. Evaluate `k=3..8` with full K-Means on a seeded sample of at most 100,000 customers; cap silhouette at 25,000. Require stability ARI of at least 0.80 and a minimum sample cluster of 0.5% or ten customers. Fit the selected MiniBatchKMeans solution and assign every purchaser in bounded batches.

Business names must come from measured profiles, not cluster IDs. A moderate silhouette is reported as overlapping tendencies, not proof of natural customer classes. Recommendations require controlled business experiments.
