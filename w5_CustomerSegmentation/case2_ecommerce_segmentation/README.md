# Case Study 2: Programmatic E-commerce Segmentation

This case processes Kaggle's multi-category ecommerce event log out of core with DuckDB and Polars. It does not use `../data/customer_segmentation_example.csv`, which is an unrelated 200-row Mall Customers sample. `data/raw/archive.zip` is the complete Kaggle archive containing both October and November.

## Resource profile

- DuckDB memory limit: 10 GB
- DuckDB threads: 4
- Process RSS warning/abort: 12/14 GB
- Free-disk warning/abort: 60/50 GiB
- DuckDB temporary storage: maximum 35 GB
- Processing unit: one month at a time; combine only the compact monthly Parquet files

## Environment

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use the synthetic smoke input only to prove the code path:

```bash
.venv/bin/python generate_smoke_data.py
```

Run the full October-November workflow from the supplied archive:

```bash
.venv/bin/python prepare_download.py --archive data/raw/archive.zip --member 2019-Oct.csv --output data/raw/2019-Oct.csv.gz
.venv/bin/python prepare_download.py --archive data/raw/archive.zip --member 2019-Nov.csv --output data/raw/2019-Nov.csv.gz
.venv/bin/python ingest.py --input data/raw/2019-Oct.csv.gz --month 2019-10 --output data/processed/2019-Oct.parquet
.venv/bin/python ingest.py --input data/raw/2019-Nov.csv.gz --month 2019-11 --output data/processed/2019-Nov.parquet
.venv/bin/python build_features.py \
  --input data/processed/2019-Oct.parquet \
  --input data/processed/2019-Nov.parquet \
  --output data/derived/2019-Oct-Nov_customer_features.parquet
.venv/bin/python cluster.py \
  --input data/derived/2019-Oct-Nov_customer_features.parquet \
  --output data/derived/2019-Oct-Nov_clustered_customers.parquet \
  --prefix 2019-Oct-Nov
```

The workflow validates each month, writes compressed Parquet, aggregates purchaser features, and runs sampled model selection followed by batched full assignment. DuckDB performs bounded out-of-core scans; Polars handles compact analytical tables and plot inputs. Large and derived data paths are ignored by the Week 5 `.gitignore`.

The source dataset is <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>. A final report must not be written from smoke data.
