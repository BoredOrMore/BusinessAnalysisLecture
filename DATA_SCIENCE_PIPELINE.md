# Modern Data Science Pipeline — A Practical Guide (2025–2026)

A hands-on reference for running an efficient, reproducible data science project using a
current-generation Python stack, organized around the **CRISP-DM** methodology.

The running example throughout is the **Telco Customer Churn** dataset already in this repo:

```
BusinessAnalysisLecture/w3_Churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Target column: `Churn` (Yes/No). Task: binary classification (predict who will churn).

---

## 1. How to use this guide

- **Sections 2–3** teach the *what* and *why*: the modern stack and how each CRISP-DM phase maps
  onto it.
- **Section 4** is a copy-pasteable end-to-end pipeline on the churn data.
- **Sections 5–6** cover project structure, reproducibility, and a production-readiness checklist.

You do **not** need every tool below. Start with `uv`, `polars`/`pandas`, `scikit-learn`, and
`mlflow`; add the rest as the project grows. The guide is pragmatic — it flags where the classic
tool (pandas, Jupyter, Airflow) is still the right call.

---

## 2. The modern stack at a glance

| Job | 2025–26 recommendation | Replaces / why |
|---|---|---|
| Environment & packaging | **`uv`** (Astral) | Replaces pip + venv + pip-tools + poetry. 10–100× faster, single lockfile, manages Python versions. `pixi` if you need conda/C-libs. |
| DataFrames (in-memory) | **Polars** | Multithreaded, lazy execution, Arrow-native — far faster & lower memory than pandas. |
| DataFrames (SQL / bigger-than-RAM) | **DuckDB** | Query CSV/Parquet/Arrow with SQL, out-of-core, zero server. |
| pandas compatibility | **pandas 2.x** w/ PyArrow backend | Keep for legacy code & the huge ecosystem; enable `dtype_backend="pyarrow"`. |
| Write-once, run-on-any-df | **Narwhals** | Library-agnostic API so code runs on Polars *and* pandas. |
| Notebooks | **marimo** (reactive, pure-`.py`, git-friendly) + **Jupyter** | marimo removes hidden-state bugs and diffs cleanly; Jupyter still fine for exploration. |
| EDA (auto) | **ydata-profiling**, **skrub** | One-line dataset reports & smart tabular preprocessing. |
| Visualization | **Altair**/**Plotly** (interactive), **seaborn** (quick static) | Declarative, publishable charts. |
| Data validation / contracts | **Pandera**, **Pydantic v2** | Enforce schema, ranges, nullability at pipeline boundaries. |
| Classical modeling | **scikit-learn** Pipelines | Still the backbone for tabular ML. |
| Gradient boosting | **LightGBM** / **XGBoost** / **CatBoost** | State-of-the-art on tabular data; CatBoost handles categoricals natively. |
| Deep learning (only if needed) | **PyTorch** (+ Lightning) | For text/image/sequence; overkill for most tabular business problems. |
| Hyperparameter tuning | **Optuna** | Efficient Bayesian/pruning search, integrates with sklearn & MLflow. |
| Explainability | **SHAP** | Feature attributions for trust & regulatory needs. |
| Experiment tracking | **MLflow** (or **Weights & Biases**) | Log params/metrics/artifacts, compare runs, model registry. |
| Data & model versioning | **DVC** / **lakeFS** | Git-like versioning for datasets and model artifacts. |
| Orchestration | **Prefect** / **Dagster** | Pythonic, observable pipelines. Airflow = legacy/heavy scheduling. |
| Serving | **FastAPI** + **BentoML**; **ONNX** for portability | Low-latency REST APIs, packaged model services. |
| Monitoring / drift | **Evidently** | Data & prediction drift, performance monitoring in prod. |
| Config | **pydantic-settings** / **Hydra** | Typed, environment-aware configuration. |
| Code quality | **ruff** (lint + format), **pytest**, **pre-commit**, **ty/mypy** | ruff replaces black+isort+flake8; test & type your pipeline. |
| AI-assisted DS | LLM copilots for EDA/feature ideas | Accelerates boilerplate; always validate outputs against data. |

---

## 3. CRISP-DM mapped to the modern stack

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is the most widely used framework
for data/analytics projects. Six phases:

```
        ┌─────────────────────────────────────────────┐
        │                                             ▼
1. Business ──▶ 2. Data ──▶ 3. Data ──▶ 4. Modeling ──▶ 5. Evaluation ──▶ 6. Deployment
  Understanding  Understanding Preparation     ▲              │
        ▲              │            ▲___________│              │
        └──────────────┴───────────────────────┴──────────────┘
   (iterative — you loop back whenever findings demand it; deployment feeds new business questions)
```

> **Key idea:** CRISP-DM is *not* linear. Data Preparation ⇄ Modeling is a tight loop, Evaluation
> can send you back to Business Understanding, and a deployed model generates new questions. Plan
> for iteration, not a one-way waterfall.

### Phase 1 — Business Understanding
- **Goal:** define the business objective and translate it into a data-science problem + success
  metric.
- **Activities:** frame the question ("reduce monthly churn by X%"), identify who acts on the
  prediction, define the target, choose a metric that reflects business cost, list constraints.
- **Churn example:** Objective = retain high-value customers. DS task = rank customers by churn
  probability. Metric = **PR-AUC / recall at a fixed contact budget** (not raw accuracy — churn is
  imbalanced). Decide the cost of a false negative (lost customer) vs false positive (wasted offer).
- **Tools:** a written project brief / model card; a decision on primary metric. No code yet —
  this phase prevents building the wrong thing.

### Phase 2 — Data Understanding
- **Goal:** acquire the data and get a feel for quality, distributions, and relationships.
- **Activities:** load data, profile it, check missingness, spot leakage, sanity-check target
  balance.
- **Tools:** **Polars**/**DuckDB** to load, **ydata-profiling** for an auto report, **Altair** for
  targeted plots.

```python
import polars as pl

CSV = "BusinessAnalysisLecture/w3_Churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pl.read_csv(CSV)

print(df.shape)                       # rows, cols
print(df["Churn"].value_counts())     # class balance
print(df.null_count())                # missingness per column

# Auto EDA report (optional, uses pandas):
# from ydata_profiling import ProfileReport
# ProfileReport(df.to_pandas(), title="Churn EDA").to_file("churn_eda.html")
```
> Note the classic Telco gotcha: `TotalCharges` is read as text and has blank strings for new
> customers — a data-understanding finding that drives the next phase.

### Phase 3 — Data Preparation
- **Goal:** produce a clean, validated, model-ready dataset. Usually the most time-consuming phase.
- **Activities:** fix types, handle missing values, encode categoricals, engineer features, and
  **validate against a schema** so bad data fails loudly.
- **Tools:** **Polars** for transforms, **Pandera** for the data contract, **scikit-learn**
  `ColumnTransformer` for fit-on-train preprocessing (avoids leakage).

```python
import polars as pl
import pandera.polars as pa

df = (
    pl.read_csv(CSV)
    .with_columns(
        pl.col("TotalCharges").str.strip_chars().replace("", None).cast(pl.Float64),
        (pl.col("Churn") == "Yes").cast(pl.Int8).alias("target"),
    )
    .drop("customerID")
)

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.gt(0)),
    "target": pa.Column(pa.Int8, pa.Check.isin([0, 1])),
})
df = schema.validate(df)   # raises on contract violation
```

### Phase 4 — Modeling
- **Goal:** train and tune candidate models inside a leak-proof pipeline.
- **Activities:** split data, build a `Pipeline` (preprocess + estimator), tune with cross-
  validation, track every experiment.
- **Tools:** **scikit-learn** `Pipeline`, **LightGBM**, **Optuna** for tuning, **MLflow** for
  tracking. Fit preprocessing *inside* the pipeline so it learns only from training folds.

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
import mlflow

pdf = df.to_pandas()
X, y = pdf.drop(columns="target"), pdf["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

cat = X.select_dtypes("object").columns.tolist()
num = X.select_dtypes("number").columns.tolist()
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num),
])
model = Pipeline([("pre", pre),
                  ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                         class_weight="balanced", random_state=42))])

mlflow.set_experiment("telco-churn")
with mlflow.start_run():
    model.fit(X_tr, y_tr)
    mlflow.sklearn.log_model(model, "model")
```

### Phase 5 — Evaluation
- **Goal:** judge the model against the *business* metric from Phase 1, not just ML metrics.
- **Activities:** compute PR-AUC/ROC-AUC, calibration, confusion at the operating threshold,
  segment performance, explainability, and a fairness check.
- **Tools:** **scikit-learn** metrics, **SHAP** for explanations, **Evidently** for a report.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import shap

proba = model.predict_proba(X_te)[:, 1]
print("ROC-AUC :", roc_auc_score(y_te, proba))
print("PR-AUC  :", average_precision_score(y_te, proba))     # better for imbalanced churn
print(classification_report(y_te, (proba > 0.5).astype(int)))

# Explainability
explainer = shap.TreeExplainer(model.named_steps["clf"])
# shap_values = explainer.shap_values(model.named_steps["pre"].transform(X_te))
```
> Decision gate: does the model beat the current baseline **and** clear the business threshold? If
> not, loop back to Phase 3/4. Also check performance across customer segments to avoid bias.

### Phase 6 — Deployment
- **Goal:** put the model where it creates value, and keep it healthy.
- **Activities:** package the pipeline, expose a scoring API, schedule batch/real-time inference,
  and monitor for drift and performance decay.
- **Tools:** **MLflow Model Registry** (versioning/staging), **FastAPI** + **BentoML** (serving),
  **Prefect**/**Dagster** (scheduled scoring), **Evidently** (drift monitoring), **Docker** (ship it).

```python
# serve.py — minimal FastAPI scoring stub
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI()
model = mlflow.sklearn.load_model("models:/telco-churn/Production")

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    # ... remaining features

@app.post("/predict")
def predict(c: Customer):
    import pandas as pd
    p = float(model.predict_proba(pd.DataFrame([c.model_dump()]))[:, 1][0])
    return {"churn_probability": p, "action": "offer_retention" if p > 0.5 else "none"}
```
> Deployment is not the end: monitor input drift and realized churn, and retrain on a schedule or
> when Evidently flags degradation — closing the CRISP-DM loop back to Business Understanding.

---

## 4. End-to-end reference pipeline (churn CSV)

A concrete, ordered runbook you can execute top to bottom.

```bash
# 0. Project + environment (uv)
uv init telco-churn && cd telco-churn
uv add polars duckdb pandas pyarrow scikit-learn lightgbm optuna \
       mlflow shap pandera ydata-profiling evidently fastapi uvicorn
uv add --dev ruff pytest pre-commit

# 1. Explore & profile
uv run python -c "import polars as pl; print(pl.read_csv('<path>/WA_Fn-UseC_-Telco-Customer-Churn.csv').describe())"

# 2. Track experiments
uv run mlflow ui           # open http://127.0.0.1:5000

# 3. Train (script from Phase 3–4 above)
uv run python train.py

# 4. Serve
uv run uvicorn serve:app --reload
```

**Logical flow:** `uv` env → load with Polars/DuckDB → clean + `Pandera` contract → sklearn
`Pipeline` + `ColumnTransformer` → LightGBM (tuned with Optuna) → log to MLflow → evaluate
(PR-AUC, calibration, SHAP, fairness) → register best model → FastAPI/BentoML serving → Evidently
drift monitoring.

---

## 5. Project layout & reproducibility

```
telco-churn/
├── pyproject.toml        # deps + tool config (managed by uv)
├── uv.lock               # exact, reproducible dependency versions
├── .python-version       # pinned interpreter
├── data/
│   ├── raw/              # immutable source data (never edit; version with DVC)
│   └── processed/        # generated, reproducible artifacts
├── notebooks/            # marimo/Jupyter exploration (not the source of truth)
├── src/telco_churn/
│   ├── data.py           # load + Pandera validation
│   ├── features.py       # feature engineering
│   ├── train.py          # pipeline + MLflow logging
│   └── evaluate.py       # metrics + SHAP
├── serve.py              # FastAPI app
├── tests/                # pytest
└── .pre-commit-config.yaml
```

**Reproducibility rules of thumb**
- Pin everything: commit `uv.lock` and `.python-version`.
- Set random seeds (`random_state=42`) everywhere; log them to MLflow.
- Treat `data/raw/` as read-only; version data with **DVC**, models via the **MLflow Registry**.
- Keep config out of code with **pydantic-settings** / **Hydra**.
- No hidden notebook state in the pipeline — production logic lives in `src/`, tested with pytest.

---

## 6. Modern practices checklist

- [ ] **Reproducible env** — `uv` project with committed lockfile.
- [ ] **Data contract** — Pandera/Pydantic schema validated at ingestion.
- [ ] **Leak-proof pipeline** — all preprocessing inside `Pipeline`/`ColumnTransformer`, fit on
      train only.
- [ ] **Right metric** — chosen from the business cost, not defaulting to accuracy.
- [ ] **Experiment tracking** — every run logged to MLflow (params, metrics, artifacts, model).
- [ ] **Explainability** — SHAP (or equivalent) for global + local feature attribution.
- [ ] **Fairness / segment check** — performance validated across key customer segments.
- [ ] **Tests + linting** — pytest on data/feature logic, ruff format+lint, pre-commit hooks.
- [ ] **Versioning** — code (git), data (DVC), models (MLflow registry).
- [ ] **CI** — lint + test on every push (GitHub Actions).
- [ ] **Monitoring** — Evidently drift/performance reports post-deploy; retraining trigger defined.
- [ ] **Model card** — documented purpose, data, metrics, limitations, and intended use.

---

## 7. Further reading

- **CRISP-DM** — the original 6-phase reference model (widely available as the CRISP-DM 1.0 guide).
- **uv** — https://docs.astral.sh/uv/
- **Polars** — https://docs.pola.rs/ · **DuckDB** — https://duckdb.org/docs/
- **scikit-learn** pipelines — https://scikit-learn.org/stable/modules/compose.html
- **LightGBM** — https://lightgbm.readthedocs.io/ · **Optuna** — https://optuna.org/
- **MLflow** — https://mlflow.org/docs/latest/ · **SHAP** — https://shap.readthedocs.io/
- **Pandera** — https://pandera.readthedocs.io/ · **Evidently** — https://docs.evidentlyai.com/
- **marimo** — https://marimo.io/ · **ruff** — https://docs.astral.sh/ruff/

---

*This document is tool-recommendation focused but pragmatic: pandas, Jupyter, and Airflow remain
valid choices where their ecosystems or your team's familiarity outweigh the newer options. Adopt
incrementally.*
