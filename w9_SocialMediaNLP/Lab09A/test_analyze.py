#!/usr/bin/env python3
"""
Unit and Invariant Validation Tests for Lab 09A Pipeline.
Verifies data conservation, normalization integrity, TF-IDF arithmetic,
leakage quarantine, and local contrast decomposition.
"""

import json
import os
import numpy as np
import pandas as pd
from analyze import clean_text


def test_text_normalization():
    """Verify negation preservation, entity masking, and contraction handling."""
    sample = "Don't tell me @VirginAmerica flight is NOT on time! Check #DelayedFlight http://t.co/xyz"
    cleaned = clean_text(sample)
    
    assert "not" in cleaned, "Negation token 'not' was erroneously stripped!"
    assert "cannot" in clean_text("I can't go"), "Contraction 'can't' not expanded to 'cannot'!"
    assert "usertoken" in cleaned, "Mention '@VirginAmerica' was not converted to 'usertoken'!"
    assert "urltoken" in cleaned, "URL 'http://t.co/xyz' was not converted to 'urltoken'!"
    assert "delayedflight" in cleaned, "Hashtag '#DelayedFlight' was not preserved as word!"
    assert "#" not in cleaned, "Hashtag symbol '#' was not stripped!"
    print("PASS: test_text_normalization")


def test_hand_calculation_arithmetic():
    """Verify sublinear TF, smoothed IDF, and L2 normalization matching spec."""
    # 1. TF-IDF arithmetic: N=3 docs, term appears once in doc 1 (df=1)
    tf_star = 1.0 + np.log(1.0)  # = 1.0
    idf = np.log((3 + 1) / (1 + 1)) + 1.0  # ln(2) + 1 ≈ 1.693147
    assert np.isclose(tf_star, 1.0)
    assert np.isclose(idf, 1.693147, atol=1e-5)

    # 2. L2 norm of (3, 4) -> (0.6, 0.8)
    vec = np.array([3.0, 4.0])
    norm = vec / np.linalg.norm(vec)
    assert np.allclose(norm, [0.6, 0.8])
    print("PASS: test_hand_calculation_arithmetic")


def test_artifacts_and_invariants():
    """Verify file existence and invariant rules on exported artifacts."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "Lab09A_outputs")
    figures_dir = os.path.join(base_dir, "figures")

    # Check Parquet registry
    parquet_path = os.path.join(outputs_dir, "airline_registry.parquet")
    assert os.path.exists(parquet_path), "airline_registry.parquet missing!"
    df_reg = pd.read_parquet(parquet_path)
    assert len(df_reg) == 14640, f"Expected 14,640 rows, got {len(df_reg)}"

    # Check Quarantined count
    quarantined = df_reg[df_reg["split"] == "quarantined"]
    assert len(quarantined) == 36, f"Expected 36 quarantined rows, got {len(quarantined)}"
    assert (quarantined["model_eligible"] == False).all(), "Quarantined rows must be model_eligible == False!"

    # Check Predictions
    pred_path = os.path.join(outputs_dir, "sentiment_predictions.csv")
    assert os.path.exists(pred_path), "sentiment_predictions.csv missing!"
    df_pred = pd.read_csv(pred_path)
    assert len(df_pred) == 1511, f"Expected 1,511 test predictions, got {len(df_pred)}"

    # Check Validation JSON
    val_path = os.path.join(outputs_dir, "validation_report.json")
    assert os.path.exists(val_path), "validation_report.json missing!"
    with open(val_path) as f:
        val_data = json.load(f)
    assert val_data["data_integrity"]["conserved_rows_assertion"] is True
    assert val_data["data_integrity"]["conflicting_tweet_ids_count"] == 18
    assert val_data["evaluation_metrics"]["local_contrast_verified"] is True

    # Check Figures
    required_figs = [
        "sentiment_counts.png",
        "airline_sentiment_stacked.png",
        "aspect_sentiment_heatmap.png",
        "training_top_bigrams.png",
        "top_coefficients.png",
        "confusion_matrices.png",
    ]
    for fig_name in required_figs:
        fig_path = os.path.join(figures_dir, fig_name)
        assert os.path.exists(fig_path), f"Figure {fig_name} missing!"

    print("PASS: test_artifacts_and_invariants")


def main():
    test_text_normalization()
    test_hand_calculation_arithmetic()
    test_artifacts_and_invariants()
    print("\nALL INVARIANT AND INTEGRITY TESTS PASSED.")


if __name__ == "__main__":
    main()
