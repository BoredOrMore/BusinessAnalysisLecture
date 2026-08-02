# Week 5 Customer Segmentation Plan

## Objective and locked scope

Build and review two reproducible case studies without making a 24 GB MacBook Pro unresponsive:

1. **Manual Cafe Transaction Log:** generate a small synthetic transaction log for exactly 150 loyalty customers, calculate RFM, and explain the clustering workflow.
2. **Programmatic E-commerce Segmentation:** derive behavioral RFM and funnel features from the [Kaggle multi-category ecommerce events](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store), then cluster customers with an out-of-core pipeline.

The current `data/customer_segmentation_example.csv` has 200 rows and Mall Customers fields (`Age`, `Annual Income`, and `Spending Score`). It matches neither slide specification nor the Kaggle event schema. Keep it only as a fast clustering smoke test; do not present it as either case-study source.

## Data-size assessment

The Kaggle description covers **285 million events across October 2019-April 2020**. The seven advertised gzip archives total about **16.45 GB compressed**. October and November alone are approximately 42 million and 67 million events and expand to roughly 5.6 GB and 9.5 GB of CSV. This is larger than safe in-memory pandas usage on this machine.

Work in three gates:

- **Gate A:** validate the code with the 200-row smoke-test file and a deterministic Cafe sample.
- **Gate B:** process October only; approve quality, runtime, peak memory, and disk use.
- **Gate C:** add November. Do not process all seven months until the two-month artifacts meet every acceptance check.

## Machine resource envelope

The current disk has about 137 GiB free. Use one pipeline process and one month at a time with these DuckDB settings:

```sql
SET memory_limit = '10GB';
SET threads = 4;
SET temp_directory = 'w5_CustomerSegmentation/.tmp/duckdb';
SET max_temp_directory_size = '35GB';
SET preserve_insertion_order = false;
SET enable_progress_bar = true;
```

Run long jobs at reduced scheduling priority (`nice -n 10 ...`). Do not run months concurrently. Abort if process RSS exceeds 14 GB, free disk falls below 50 GiB, the temp directory reaches 35 GB, or macOS memory pressure becomes red. Warn and pause new stages at 12 GB RSS or 60 GiB free disk. Record wall time, peak RSS, rows read/written, and temp-disk peak for every run.

DuckDB can scan gzip CSV directly, and its memory setting is not a complete process-wide hard limit. Therefore, the implementation must add an external RSS/disk watchdog and interrupt the active query at the hard thresholds.

## Case Study 1: Cafe RFM

1. Use `case1_cafe_rfm/generate.py` with `RANDOM_SEED = 42`. Generate the four slide archetypes, exactly 150 unique `CUST-001`-`CUST-150` customers, valid menu prices, timestamps, transaction IDs, and visit spend.
2. Keep the archetype label in a separate validation file; never include it as a clustering feature.
3. Define recency as days from `max(transaction_time) + 1 day`, frequency as distinct transactions, and monetary as summed line spend. Print the reference date and definitions.
4. Validate schema, zero nulls, nonnegative spend, unique transaction keys, timestamp range, and customer count before analysis.
5. Compare scaled RFM solutions, but do not claim success merely because four source archetypes exist. Report silhouette, cluster stability across seeds, cluster sizes, and recovery against the withheld labels.

## Case Study 2: Out-of-core ecommerce pipeline

1. Download immutable `.csv.gz` files into `data/raw/`; store source URL, byte size, SHA-256, and download date in a small manifest.
2. Profile the header and first rows, then require the nine documented fields: `event_time`, `event_type`, product/category fields, `brand`, `price`, `user_id`, and `user_session`.
3. Stream each gzip file through DuckDB with explicit types. Write Zstandard level-1 Parquet, one month at a time, with approximately 250,000-row groups. Retain only one canonical processed copy; never expand and keep a full CSV.
4. Reconcile input/output row counts and report timestamp parse failures, null rates, invalid prices, event-type counts, and exact-duplicate rates. Do **not** silently delete duplicate purchase rows: multiple items can occur in one session. Deduplication requires a documented sensitivity check and an approved business rule.
5. Aggregate before Python modeling. Purchase RFM uses distinct purchase sessions for frequency and purchase-line price for monetary value. Add views, cart events, purchase items, active days, session count, category diversity, cart-to-purchase rate, and average order value. State the observation cutoff and use no events after it.
6. Store one row per customer in Parquet. Fit transformations and `MiniBatchKMeans` incrementally; never load raw events into pandas. Evaluate `k=3..8` on a fixed, seeded sample of at most 250,000 customers. Cap silhouette calculation at 25,000 rows because it is pairwise-expensive. Assign all customer labels in batches.

## Required outputs and acceptance checks

- `quality_report.json`: schema, counts, nulls, duplicates, ranges, and reconciliation by month.
- `run_metrics.json`: configuration, versions, elapsed time, peak RSS/temp space, and row counts.
- Case 1 outputs under `case1_cafe_rfm/`; Case 2 `customer_features.parquet`, cluster profiles, figures, and an executive report that derives every number from a recorded run.
- Re-running from the same manifest and seed must reproduce feature counts and materially identical cluster profiles.
- No raw/processed data, temporary files, credentials, caches, or database files may be committed. Only code, manifests without secrets, compact summary artifacts, figures, and reports belong in Git.

## Implementation order

Implement and review in this order: preflight/watchdog -> Cafe generator and validation -> October ingestion -> quality report -> monthly Parquet -> customer features -> sampled model selection -> batched full assignment -> November expansion -> executive review. Stop at any failed gate; never weaken a threshold to force completion.

## Technical references

- [Kaggle dataset and schema](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store/data)
- [DuckDB configuration](https://duckdb.org/docs/stable/configuration/overview)
- [DuckDB compressed CSV import](https://duckdb.org/docs/current/data/overview)
- [DuckDB Parquet guidance](https://duckdb.org/docs/stable/data/parquet/overview)
- [DuckDB out-of-memory guidance](https://duckdb.org/docs/current/guides/performance/oom)
