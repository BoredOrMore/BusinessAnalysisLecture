"""Check that baseline auditing separates missingness, Unknown, and leakage."""

import pandas as pd

from analyze import summarize


def main():
    data = pd.DataFrame({
        "CLIENTNUM": ["a", "b", "c", "d"],
        "Attrition_Flag": ["Existing Customer", "Attrited Customer",
                           "Existing Customer", "Existing Customer"],
        "Education_Level": ["Unknown", " ", None, "Unknown"],
        "Naive_Bayes_Classifier_Attrition_Flag_1": [0.1, 0.9, 0.2, 0.3],
    })
    metrics = summarize(data)
    assert metrics["missing_cells"] == 2
    assert metrics["unknown_by_column"] == {"Education_Level": 2}
    assert metrics["duplicate_profiles"] == 1
    assert metrics["churn_share"] == 0.25
    assert metrics["always_stay_accuracy"] == 0.75
    assert metrics["always_stay_recall"] == 0.0
    assert metrics["excluded_predictor_columns"] == [
        "CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_1"
    ]
    for column, value in [("CLIENTNUM", "b"), ("Attrition_Flag", "unexpected")]:
        invalid = data.copy()
        invalid.loc[0, column] = value
        try:
            summarize(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid {column} was accepted")
    print("PASS: missingness, Unknown, duplicates, baseline, and invalid inputs")


if __name__ == "__main__":
    main()
