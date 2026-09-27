"""Run the credit-card retention study in Customer_Churn_Practice.pdf, pp. 2-24.

Select models on training CV, choose cost thresholds on validation, and evaluate
frozen decisions on test. Historical churn prediction does not establish uplift.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyze import CASE_DIR, LABELS, TARGET, plt, summarize


NUMERIC = [
    "Customer_Age", "Dependent_count", "Months_on_book",
    "Total_Relationship_Count", "Months_Inactive_12_mon", "Contacts_Count_12_mon",
    "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
]
CATEGORICAL = ["Gender", "Education_Level", "Marital_Status", "Income_Category",
               "Card_Category"]
DERIVED = ["Average_Ticket", "Contact_Ratio"]
GRID = {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [2, 8],
        "model__max_features": ["sqrt", 0.6]}


def features(data):
    """Use an explicit predictor allowlist and row-local safe ratios."""
    missing = set(NUMERIC + CATEGORICAL) - set(data.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    x = data[NUMERIC + CATEGORICAL].copy().replace(r"^\s*$", np.nan, regex=True)
    for column in NUMERIC:
        x[column] = pd.to_numeric(x[column], errors="raise")
        values = x[column].dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Nonfinite or negative values in {column}")
    if (x["Avg_Utilization_Ratio"].dropna() > 1).any():
        raise ValueError("Avg_Utilization_Ratio must be between 0 and 1")
    if (x["Months_Inactive_12_mon"].dropna() > 12).any():
        raise ValueError("Months_Inactive_12_mon exceeds 12")
    x["Average_Ticket"] = x["Total_Trans_Amt"] / x["Total_Trans_Ct"].clip(lower=1)
    x["Contact_Ratio"] = (
        x["Contacts_Count_12_mon"] / x["Total_Relationship_Count"].clip(lower=1)
    )
    return x


def split_indices(y, seed):
    """Match practice cohort sizes; seed is our reproducibility choice."""
    train_val, test = train_test_split(
        np.arange(len(y)), test_size=math.ceil(len(y) * 0.2),
        stratify=y, random_state=seed,
    )
    train, validation = train_test_split(
        train_val, test_size=math.floor(len(y) * 0.2),
        stratify=y.iloc[train_val], random_state=seed,
    )
    assert not (set(train) & set(validation) or set(train) & set(test)
                or set(validation) & set(test))
    assert len(set(train) | set(validation) | set(test)) == len(y)
    return {"train": train, "validation": validation, "test": test}


def make_pipeline(kind, seed):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    branches = [("numeric", numeric, NUMERIC + DERIVED)]
    if kind == "weighted_full":
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        branches.append(("categorical", categorical, CATEGORICAL))
    steps = [("preprocess", ColumnTransformer(branches))]
    if kind == "adasyn_numeric":
        steps.append(("resample", ADASYN(random_state=seed, n_neighbors=5)))
    steps.append(("model", RandomForestClassifier(
        n_estimators=150, random_state=seed, n_jobs=1,
        class_weight="balanced" if kind == "weighted_full" else None,
    )))
    return Pipeline(steps)


def threshold_costs(y, probabilities, fn_cost, fp_cost):
    """Evaluate every distinct >= threshold in O(n log n), including no flags."""
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    p = np.asarray(probabilities)[order]
    labels = np.asarray(y)[order]
    ends = np.r_[p[:-1] != p[1:], True]
    tp = np.r_[0, labels.cumsum()[ends]]
    flagged = np.r_[0, np.arange(1, len(p) + 1)[ends]]
    fp, fn = flagged - tp, labels.sum() - tp
    return pd.DataFrame({
        "threshold": np.r_[np.nextafter(1.0, 2.0), p[ends]],
        "false_negatives": fn, "false_positives": fp, "flagged": flagged,
        "cost": fn_cost * fn + fp_cost * fp,
    })


def evaluate(y, p, threshold, fn_cost, fp_cost):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    return {
        "n": len(y), "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": float((tp + tn) / len(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": float(fn_cost * fn + fp_cost * fp),
        "flagged": int(tp + fp), "threshold": float(threshold),
    }


def save_figure(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def training_eda(data, y, outputs, figures):
    """Derive bins and associations from training records only."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    tables = []
    columns = ["Total_Trans_Ct", "Months_Inactive_12_mon", "Total_Relationship_Count"]
    for ax, column in zip(axes, columns):
        group = (pd.qcut(data[column], 7, duplicates="drop")
                 if column == "Total_Trans_Ct" else data[column])
        table = pd.DataFrame({"group": group, "churn": y}).groupby(
            "group", observed=True, dropna=False
        )["churn"].agg(["count", "sum", "mean"]).reset_index()
        table.columns = ["group", "customers", "churners", "churn_rate"]
        table.insert(0, "feature", column)
        tables.append(table)
        bars = ax.bar(table["group"].astype(str), 100 * table["churn_rate"], color="#25638a")
        ax.bar_label(bars, labels=[f"n={n}" for n in table["customers"]], fontsize=8)
        ax.set(title=column, ylabel="Observed churn (%)", ylim=(0, 65))
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Training cohort only: activity, inactivity, relationships")
    save_figure(fig, figures, "training_eda")
    pd.concat(tables).to_csv(outputs / "training_eda.csv", index=False)


def explanation(model, x, ids, probabilities, seed, outputs, figures):
    """Explain a deterministic test sample plus its highest-risk customer."""
    import shap

    rng = np.random.default_rng(seed)
    positions = np.unique(np.r_[rng.choice(len(x), min(250, len(x)), replace=False),
                                np.argmax(probabilities)])
    sample = x.iloc[positions]
    pre = model.named_steps["preprocess"]
    z = pre.transform(sample)
    forest = model.named_steps["model"]
    engine = shap.TreeExplainer(forest, model_output="raw",
                               feature_perturbation="tree_path_dependent")
    raw = engine.shap_values(z)
    class_index = list(forest.classes_).index(1)
    phi = raw[class_index] if isinstance(raw, list) else raw[:, :, class_index]
    base = float(engine.expected_value[class_index])
    np.testing.assert_allclose(base + phi.sum(axis=1), probabilities[positions],
                               atol=1e-5, rtol=0)
    names = NUMERIC + DERIVED
    grouped = [phi[:, i] for i in range(len(names))]
    if "categorical" in pre.named_transformers_:
        encoder = pre.named_transformers_["categorical"].named_steps["encode"]
        offset = len(names)
        for column, categories in zip(CATEGORICAL, encoder.categories_):
            grouped.append(phi[:, offset:offset + len(categories)].sum(axis=1))
            offset += len(categories)
        names = names + CATEGORICAL
    values = np.column_stack(grouped)
    np.testing.assert_allclose(values.sum(axis=1), phi.sum(axis=1), atol=1e-10)
    importance = pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)
    importance.rename("mean_absolute_shap").to_csv(outputs / "shap_importance.csv", index_label="feature")
    contributions = pd.DataFrame(values, columns=names)
    contributions.insert(0, "CLIENTNUM", ids.iloc[positions].to_numpy())
    contributions.to_csv(outputs / "shap_contributions.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = list(importance.head(8).index)[::-1]
    for i, column in enumerate(top):
        j = names.index(column)
        vertical = i + rng.uniform(-0.22, 0.22, len(sample))
        if column in CATEGORICAL:
            ax.scatter(values[:, j], vertical, color="#79529a", s=9, alpha=0.6)
        else:
            ax.scatter(values[:, j], vertical, c=sample[column].to_numpy(),
                       cmap="coolwarm", s=9, alpha=0.6)
    ax.set(yticks=range(len(top)), yticklabels=top, xlabel="Contribution to churn probability",
           title="Grouped TreeSHAP: test sample (blue low, red high; purple categorical)")
    ax.axvline(0, color="grey", lw=0.7)
    save_figure(fig, figures, "shap_beeswarm")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sample["Total_Trans_Ct"], values[:, names.index("Total_Trans_Ct")],
               s=12, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(xlabel="Transaction count", ylabel="SHAP contribution", title="Transaction-count attribution")
    save_figure(fig, figures, "shap_dependence")
    local = int(np.flatnonzero(positions == np.argmax(probabilities))[0])
    display = np.array([np.nan if c in CATEGORICAL else sample.iloc[local][c] for c in names])
    shap.plots.waterfall(shap.Explanation(values=values[local], base_values=base,
                                        data=display, feature_names=names),
                         max_display=6, show=False)
    save_figure(plt.gcf(), figures, "shap_waterfall")
    return {"sample_size": len(sample), "reference_probability": base,
            "top_feature": str(importance.index[0]),
            "top_mean_absolute_shap": float(importance.iloc[0]),
            "local_customer_id": str(ids.iloc[positions[local]]),
            "local_probability": float(probabilities[positions[local]]),
            "max_additivity_error": float(np.abs(base + values.sum(1) - probabilities[positions]).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CASE_DIR / "data/BankChurners.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fn-cost", type=float, default=5, help="Teaching cost per missed churn")
    parser.add_argument("--fp-cost", type=float, default=1, help="Teaching cost per false alarm")
    parser.add_argument("--illustrative-hazard", type=float, default=0.035,
                        help="Hypothetical hazard per unspecified time unit; not fitted")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not all(np.isfinite(c) and c > 0 for c in [args.fn_cost, args.fp_cost]):
        parser.error("Costs must be finite and positive")
    if not np.isfinite(args.illustrative_hazard) or args.illustrative_hazard < 0:
        parser.error("Illustrative hazard must be finite and nonnegative")
    outputs, figures = CASE_DIR / "outputs", CASE_DIR / "figures"
    outputs.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    data = pd.read_csv(args.data)
    audit = summarize(data)
    x = features(data)
    if x.assign(target=data[TARGET]).duplicated().any():
        raise ValueError("Repeated profiles require review/grouping before random splitting")
    y = data[TARGET].map(LABELS).astype(int)
    splits = split_indices(y, args.seed)
    manifest = data[["CLIENTNUM"]].copy()
    for name, indices in splits.items():
        manifest.loc[indices, "split"] = name
    manifest.to_csv(outputs / "split_manifest.csv", index=False)
    train, validation, test = (splits[k] for k in ["train", "validation", "test"])
    print(f"Split: {len(train)} train / {len(validation)} validation / {len(test)} test", flush=True)
    training_eda(data.iloc[train], y.iloc[train], outputs, figures)
    cv = list(StratifiedKFold(3, shuffle=True, random_state=args.seed).split(x.iloc[train], y.iloc[train]))
    candidates, fitted = [], {}
    for kind in ["weighted_full", "numeric_control", "adasyn_numeric"]:
        print(f"Tuning {kind}: 12 candidates x 3 folds", flush=True)
        search = GridSearchCV(make_pipeline(kind, args.seed), GRID, cv=cv,
                              scoring="average_precision", n_jobs=args.jobs,
                              error_score="raise", return_train_score=False)
        search.fit(x.iloc[train], y.iloc[train])
        fitted[kind] = search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(outputs / f"cv_{kind}.csv", index=False)
        candidates.append({"model": kind, "cv_ap": float(search.best_score_),
                           "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                           "parameters": search.best_params_})
        print(f"{kind}: CV AP {search.best_score_:.4f}", flush=True)
    winner = max(candidates, key=lambda c: c["cv_ap"])["model"]
    model = fitted[winner]
    thresholds = {}
    for candidate in candidates:
        kind = candidate["model"]
        probabilities = fitted[kind].predict_proba(x.iloc[validation])[:, 1]
        costs = threshold_costs(y.iloc[validation], probabilities, args.fn_cost, args.fp_cost)
        costs.to_csv(outputs / f"threshold_{kind}.csv", index=False)
        thresholds[kind] = float(costs.loc[costs["cost"].idxmin(), "threshold"])
        candidate["validation"] = evaluate(y.iloc[validation], probabilities, thresholds[kind],
                                            args.fn_cost, args.fp_cost)
    print(f"Frozen selection: {winner}; threshold {thresholds[winner]:.6f}", flush=True)
    predictions = data.iloc[test][["CLIENTNUM", TARGET]].copy()
    test_metrics = {}
    for kind, estimator in fitted.items():
        p = estimator.predict_proba(x.iloc[test])[:, 1]
        predictions[kind] = p
        test_metrics[kind] = evaluate(y.iloc[test], p, thresholds[kind], args.fn_cost, args.fp_cost)
    p = predictions[winner].to_numpy()
    predictions["selected_flag"] = (p >= thresholds[winner]).astype(int)
    predictions.to_csv(outputs / "test_predictions.csv", index=False)
    final = test_metrics[winner]
    test_y = y.iloc[test].to_numpy()
    rng = np.random.default_rng(args.seed)
    boot = [average_precision_score(test_y[idx], p[idx]) for idx in
            rng.integers(0, len(test), size=(500, len(test)))]
    final["ap_bootstrap_95_interval"] = np.quantile(boot, [0.025, 0.975]).tolist()
    subgroup = []
    for column in ["Gender", "Income_Category"]:
        for group in data.iloc[test][column].unique():
            mask = data.iloc[test][column].eq(group).to_numpy()
            result = evaluate(test_y[mask], p[mask], thresholds[winner], args.fn_cost, args.fp_cost)
            subgroup.append({"field": column, "group": group, **result})
    pd.DataFrame(subgroup).to_csv(outputs / "subgroup_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for kind in fitted:
        pr, re, _ = precision_recall_curve(test_y, predictions[kind])
        axes[0].plot(re, pr, label=kind)
        fpr, tpr, _ = roc_curve(test_y, predictions[kind])
        axes[1].plot(fpr, tpr, label=kind)
    axes[0].axhline(test_y.mean(), color="grey", ls="--", label="Prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall")
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    observed, predicted = calibration_curve(test_y, p, n_bins=10, strategy="quantile")
    pd.DataFrame({"predicted": predicted, "observed": observed}).to_csv(outputs / "calibration.csv", index=False)
    axes[2].plot(predicted, observed, "o-")
    axes[2].plot([0, 1], [0, 1], color="grey", ls="--")
    axes[2].set(xlabel="Predicted risk", ylabel="Observed churn", title=f"Calibration: {winner}")
    axes[0].legend(fontsize=7)
    save_figure(fig, figures, "test_evaluation")
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array([[final["tn"], final["fp"]], [final["fn"], final["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stay", "Churn"],
           yticklabels=["Stay", "Churn"], xlabel="Predicted", ylabel="Actual",
           title="Frozen test confusion matrix")
    save_figure(fig, figures, "test_confusion_matrix")
    print("Computing exact TreeSHAP and checking probability reconstruction", flush=True)
    explanations = explanation(model, x.iloc[test], data.iloc[test]["CLIENTNUM"],
                               p, args.seed, outputs, figures)
    versions = {name: importlib.metadata.version(name) for name in
                ["pandas", "numpy", "scikit-learn", "imbalanced-learn", "shap", "matplotlib", "joblib"]}
    artifact = {"model": model, "threshold": thresholds[winner], "feature_columns": list(x.columns),
                "feature_recipe": "churn_pipeline.features", "versions": versions}
    joblib.dump(artifact, outputs / "churn_model.joblib", compress=3)
    loaded = joblib.load(outputs / "churn_model.joblib")
    np.testing.assert_allclose(loaded["model"].predict_proba(x.iloc[test])[:, 1], p)
    metrics = {"data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
               "seed": args.seed, "fn_cost": args.fn_cost, "fp_cost": args.fp_cost,
               "illustrative_hazard": args.illustrative_hazard,
               "versions": versions, "audit": audit,
               "splits": {name: {"n": len(idx), "churners": int(y.iloc[idx].sum())}
                          for name, idx in splits.items()},
               "candidates": candidates, "selected_model": winner, "test": test_metrics,
               "shap": explanations,
               "always_stay_test_cost": float(args.fn_cost * test_y.sum()),
               "always_flag_test_cost": float(args.fp_cost * (1 - test_y).sum())}
    (outputs / "pipeline_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    write_report(metrics, outputs, figures)
    print(json.dumps({"selected": winner, **final}, indent=2), flush=True)
    print("Saved PIPELINE_REPORT.md, figures, CSVs, metrics, and reload-verified model.", flush=True)


def write_report(m, outputs, figures):
    """Render computed results alongside practice coverage and remaining limits."""
    time = np.linspace(0, 30, 121)
    survival = np.exp(-m["illustrative_hazard"] * time)
    assert survival[0] == 1 and (np.diff(survival) <= 0).all()
    pd.DataFrame({"hypothetical_time": time, "hypothetical_survival": survival}).to_csv(
        outputs / "hypothetical_survival.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, survival)
    ax.set(xlabel="Hypothetical time units (not months)", ylabel="Hypothetical S(t)",
           ylim=(0, 1.05), title=f"Illustration only: constant hazard {m['illustrative_hazard']:g}")
    save_figure(fig, figures, "hypothetical_survival")
    winner = m["selected_model"]
    result, s = m["test"][winner], m["shap"]
    rows = "\n".join(
        f"| {c['model']} | {c['cv_ap']:.4f} | {m['test'][c['model']]['average_precision']:.4f} | "
        f"{m['test'][c['model']]['recall']:.2%} | {m['test'][c['model']]['precision']:.2%} | "
        f"{m['test'][c['model']]['cost']:.0f} |" for c in m["candidates"]
    )
    ci = result["ap_bootstrap_95_interval"]
    eda = pd.read_csv(outputs / "training_eda.csv")
    eda_findings = []
    for feature, groups in eda.groupby("feature", sort=False):
        high = groups.loc[groups["churn_rate"].idxmax()]
        low = groups.loc[groups["churn_rate"].idxmin()]
        eda_findings.append(
            f"- **{feature}:** highest observed rate {high['churn_rate']:.2%} in "
            f"group `{high['group']}` (n={int(high['customers']):,}); lowest "
            f"{low['churn_rate']:.2%} in `{low['group']}` (n={int(low['customers']):,})."
        )
    eda_text = "\n".join(eda_findings)
    selected_params = next(c["parameters"] for c in m["candidates"] if c["model"] == winner)
    report = f"""# Credit-card Churn: Full Pipeline Report

Procedure: [Customer_Churn_Practice.pdf](Customer_Churn_Practice.pdf), pages 2-24.
This is the completed credit-card case. The separate e-commerce case is not included.
All numerical findings below are generated by `churn_pipeline.py` from the local data.

## Decision and measured performance

Selected **{winner}** by mean training cross-validation average precision (AP).
Validation selected threshold **{result['threshold']:.6f}** by minimizing
`{m['fn_cost']:g} * false negatives + {m['fp_cost']:g} * false positives`.
Equal-cost thresholds prefer fewer flags. Costs are teaching units, not verified money.
The earlier top-10% contact scenario is superseded by this practice cost rule.
Selected parameters: `{json.dumps(selected_params, sort_keys=True)}`.

| Model | Training CV AP | Test AP | Test recall | Test precision | Test cost |
|---|---:|---:|---:|---:|---:|
{rows}

Model selection was frozen before validation threshold selection and test scoring.
Test comparisons are descriptive and do not trigger reselection. CV estimates are
selection scores, not unbiased generalization estimates.

- Selected test ROC-AUC: **{result['roc_auc']:.4f}**; Brier score: **{result['brier']:.4f}**.
- Test F1: **{result['f1']:.4f}**; accuracy: **{result['accuracy']:.2%}**.
- Test prevalence / constant-score AP baseline: **{result['prevalence']:.2%}**.
- AP bootstrap 95% interval: **[{ci[0]:.4f}, {ci[1]:.4f}]**, using 500 customer resamples.
  This conditions on the fitted model and split; it does not capture retraining or temporal uncertainty.
- Confusion matrix: TN **{result['tn']}**, FP **{result['fp']}**, FN **{result['fn']}**, TP **{result['tp']}**.
- Customers flagged: **{result['flagged']} / {result['n']}**.
- Always-stay test cost: **{m['always_stay_test_cost']:.0f}**; always-flag cost: **{m['always_flag_test_cost']:.0f}**.
  Classification cost reduction is not measured campaign profit or causal retention lift.

![Evaluation](figures/test_evaluation.png)
![Confusion matrix](figures/test_confusion_matrix.png)

## Practice steps and problems

| Practice pages | Implementation / finding | Status |
|---|---|---|
| 2-4: architecture and integrity | {m['audit']['rows']:,} rows, {m['audit']['missing_cells']} missing cells; Unknown remains explicit; IDs unique | Complete |
| 5-7: activity, inactivity, relationships | Seven training-derived activity quantile bins; group rates and counts exported | Complete; sparse groups are descriptive |
| 8: preprocessing | Drop ID and both target-derived outputs; numeric median/scale; categorical mode/one-hot; fit inside CV | Complete |
| 9: features | Average ticket and contact ratio use denominator floor 1 | Complete; timing remains unverified |
| 10-11: model and search | {m['splits']['train']['n']:,} train / {m['splits']['validation']['n']:,} validation / {m['splits']['test']['n']:,} test; three folds; 12 configurations; 150 trees | Complete |
| 12-14: ranking, costs, calibration | AP/ROC, validation-only threshold, held-out confusion matrix and reliability diagram | Complete; no test-fitted recalibration |
| 15-22: TreeSHAP | Frozen selected forest; grouped dummy contributions; beeswarm, dependence, local waterfall | Complete; associations are not interventions |
| 23: survival | Hypothetical illustration reproduced; event origin, event time, censoring, and feature availability dates absent | Empirical survival blocked by missing data |
| 24: retention action | Experiment and launch gates below | Proposal only; treatment evidence absent |

Split seed **{m['seed']}** is a local reproducibility choice. We match the practice
procedure, not its reported fitted outputs: its complete software environment and
split assignment are not provided. Source SHA-256: `{m['data_sha256']}`.
Split membership is saved before EDA. Validation and test rows are never resampled.

![Training EDA](figures/training_eda.png)

{eda_text}

These training associations are hypotheses for service review. In particular,
small inactivity groups should not define an intervention trigger without further
evidence. Bins are fitted locally on training data and need not match slide bins.

## ADASYN comparison and its boundary

`weighted_full` follows the practice with balanced class weights and all predictors.
`numeric_control` and `adasyn_numeric` share the same numeric features, tuning grid,
unweighted forest, and CV folds; the latter adds fold-local ADASYN after imputation
and scaling. This paired comparison separates oversampling from dropping categories.
Full-feature versus numeric-only results cannot isolate the effect of ADASYN alone.

No categorical interpolation or SMOTE is used. Numeric interpolation still creates
fractional counts and may break algebraic relationships between ratios and parent
features; these are synthetic model-space points, not real customer profiles.
ADASYN can emphasize noisy boundaries. Calibration and held-out performance are
therefore reported; selection is empirical rather than assuming oversampling helps.
See [imbalanced-learn oversampling guidance](https://imbalanced-learn.org/stable/over_sampling.html).

## Explainability

Exact path-dependent TreeSHAP explains churn probability for **{s['sample_size']}**
test customers: a seeded random sample plus the highest-risk test customer, chosen
without its true label. Largest mean absolute grouped attribution:
**{s['top_feature']}**, **{100*s['top_mean_absolute_shap']:.2f} percentage points**.

The model reference is **{s['reference_probability']:.2%}**, reflecting stored
training-path weights, including any class weighting or resampling. It is not
automatically the population churn rate. Local example `{s['local_customer_id']}`
has predicted risk **{s['local_probability']:.2%}**. Baseline plus contributions
reconstructs predicted probabilities; maximum error **{s['max_additivity_error']:.2e}**.
Grouping one-hot columns preserves the sum but does not define a new coalition game.
Correlated inputs can share credit. Colors compare values within each numeric feature;
categorical values are purple. Findings describe model reliance, not causal drivers.
See [TreeExplainer documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html).

![SHAP beeswarm](figures/shap_beeswarm.png)
![SHAP dependence](figures/shap_dependence.png)
![Local explanation](figures/shap_waterfall.png)

## Survival illustration only

To reproduce page 23's teaching concept, plot `S(t) = exp(-hazard * t)` using an
assumed constant hazard of **{m['illustrative_hazard']:g}** per hypothetical time unit.
This parameter is supplied through `--illustrative-hazard`, not estimated from
customers. The curve is not empirical survival, a monthly retention forecast, or
an input to CLV. Event and censoring dates are still required for a real study.

![Hypothetical survival only](figures/hypothetical_survival.png)

## Retention proposal and unresolved gates

1. Verify feature cutoff times and the definition/date of churn; validate the frozen
   pipeline on an independent future cohort before treating scores as prospective risk.
2. Review eligible active accounts for disengagement and service friction. Historical
   test predictions are evaluation records, not an authorized live contact list.
3. Randomize eligible customers to a defined service/contact action or control.
   Pre-register horizon, minimum meaningful lift, sample size, exclusions, and stop rule.
4. Measure incremental retained contribution after handling and incentive costs.
   Transaction spend is not bank margin; no CLV, survival curve, or ROI is inferred here.
5. Monitor missingness, feature drift, calibration, recall, contact capacity, and
   subgroup outcomes. `subgroup_metrics.csv` reports Gender and Income_Category
   performance with denominators; these descriptive checks do not certify fairness.
   Pause rollout if timing, future-validation, or positive incremental-value gates fail.

## Reproduce and inspect

```bash
w9/.venv/bin/python w9/analyze.py
w9/.venv/bin/python w9/test_analyze.py
w9/.venv/bin/python w9/test_pipeline.py
w9/.venv/bin/python w9/churn_pipeline.py --seed {m['seed']} --fn-cost {m['fn_cost']:g} --fp-cost {m['fp_cost']:g} --illustrative-hazard {m['illustrative_hazard']:g}
```

`requirements.txt` records package versions used. `outputs/pipeline_metrics.json`
records configuration, versions, audit, model choices, evaluation, and SHAP checks.
Other outputs include split assignments, all CV trials, validation threshold curves,
test predictions, subgroup metrics, calibration bins, and grouped SHAP contributions.
`outputs/churn_model.joblib` contains the fitted pipeline and threshold and was
reloaded to verify identical predictions. For scoring, apply `features(raw_rows)`
before its pipeline; do not refit preprocessing or use the target as an input.
Only load model artifacts you trust. The local model and virtual environment are ignored by Git.
"""
    (CASE_DIR / "PIPELINE_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
