"""Check leakage boundaries, ratio safety, and exact cost threshold selection."""

import numpy as np
import pandas as pd

from churn_pipeline import (
    CATEGORICAL, DERIVED, NUMERIC, features, make_pipeline,
    split_indices, threshold_costs,
)


def main():
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(rng.uniform(1, 10, size=(300, len(NUMERIC))), columns=NUMERIC)
    raw["Avg_Utilization_Ratio"] /= 10
    for column in CATEGORICAL:
        raw[column] = "known"
    raw["CLIENTNUM"] = np.arange(len(raw))
    raw["Attrition_Flag"] = "Existing Customer"
    raw["Naive_Bayes_Classifier_Attrition_Flag_1"] = 0.99
    raw.loc[0, ["Total_Trans_Ct", "Total_Relationship_Count"]] = 0
    x = features(raw)
    assert set(x.columns) == set(NUMERIC + CATEGORICAL + DERIVED)
    assert x.loc[0, "Average_Ticket"] == raw.loc[0, "Total_Trans_Amt"]
    assert x.loc[0, "Contact_Ratio"] == raw.loc[0, "Contacts_Count_12_mon"]
    invalid = raw.copy()
    invalid.loc[0, "Avg_Utilization_Ratio"] = 1.1
    try:
        features(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid utilization accepted")
    y = pd.Series((np.arange(len(raw)) % 5 == 0).astype(int))
    sizes = split_indices(pd.Series(np.arange(10127) % 5 == 0), 42)
    assert [len(sizes[k]) for k in ["train", "validation", "test"]] == [6076, 2025, 2026]
    assert all(np.array_equal(sizes[k], split_indices(pd.Series(np.arange(10127) % 5 == 0), 42)[k])
               for k in sizes)

    for kind in ["weighted_full", "numeric_control", "adasyn_numeric"]:
        model = make_pipeline(kind, 42).set_params(model__n_estimators=3)
        model.fit(x.iloc[:250], y.iloc[:250])
        holdout = x.iloc[250:].copy()
        holdout["Gender"] = "unseen"
        holdout["Customer_Age"] = 1000
        scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_[0], x.iloc[:250]["Customer_Age"].mean())
        assert model.predict_proba(holdout).shape == (len(holdout), 2)
        if kind == "adasyn_numeric":
            z = model.named_steps["preprocess"].transform(x.iloc[:250])
            assert z.shape[1] == len(NUMERIC + DERIVED)
            resampled, labels = model.named_steps["resample"].fit_resample(z, y.iloc[:250])
            assert len(resampled) > len(z) and labels.sum() > y.iloc[:250].sum()

    labels = np.array([1, 0, 1, 0, 0])
    probabilities = np.array([0.9, 0.9, 0.2, 0.1, 0.0])
    costs = threshold_costs(labels, probabilities, 5, 1)
    assert len(costs) == len(np.unique(probabilities)) + 1
    for row in costs.itertuples():
        flags = probabilities >= row.threshold
        brute = 5 * ((labels == 1) & ~flags).sum() + ((labels == 0) & flags).sum()
        assert brute == row.cost
    assert costs.loc[costs["cost"].idxmin(), "threshold"] == 0.2
    print("PASS: leakage allowlist, safe ratios, split, train-only transforms, ADASYN, cost thresholds")


if __name__ == "__main__":
    main()
