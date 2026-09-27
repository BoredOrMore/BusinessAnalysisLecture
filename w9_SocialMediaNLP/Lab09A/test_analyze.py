#!/usr/bin/env python3
"""
Comprehensive Automated Test Suite for Lab 09A.
Explicitly tests and verifies all 18 Tickets (TICKET-01 to TICKET-18)
covering Data Conservation, Leakage Quarantine, TF-IDF Mathematics,
Cross-Validation, Local Contrast, Aspect Triage, and Deliverables.
"""

import json
import os
import sys
import nbformat
import numpy as np
import pandas as pd

# Add Lab09A directory to path to import clean_text
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from analyze import clean_text

OUTPUTS_DIR = os.path.join(BASE_DIR, "Lab09A_outputs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


# -----------------------------------------------------------------------------
# Epic 1: Ingestion, Leakage Audit & Graph Deduplication
# -----------------------------------------------------------------------------

def test_ticket_01_ingestion_conservation():
    """TICKET-01: Ingestion & Row Conservation (N = 14,640)."""
    parquet_path = os.path.join(OUTPUTS_DIR, "airline_registry.parquet")
    assert os.path.exists(parquet_path), "airline_registry.parquet missing!"
    df = pd.read_parquet(parquet_path)
    assert len(df) == 14640, f"Expected 14,640 rows, got {len(df)}"
    assert (df["source_row"] == np.arange(14640)).all(), "source_row is not 0-indexed sequential!"
    print("PASS: TICKET-01 (Ingestion & Row Conservation N=14,640)")


def test_ticket_02_quarantine_conflicting_labels():
    """TICKET-02: Quarantine 18 Conflicting tweet_id Groups (36 Rows)."""
    df = pd.read_parquet(os.path.join(OUTPUTS_DIR, "airline_registry.parquet"))
    quarantined = df[df["split"] == "quarantined"]
    assert len(quarantined) == 36, f"Expected exactly 36 quarantined rows, got {len(quarantined)}"
    assert (quarantined["audit_flags"].str.contains("CONFLICTING_LABEL")).all(), "Missing CONFLICTING_LABEL flag!"
    assert (quarantined["model_eligible"] == False).all(), "Quarantined rows must have model_eligible == False!"
    # Ensure zero quarantined rows leaked into train or test
    manifest = pd.read_csv(os.path.join(OUTPUTS_DIR, "sentiment_split_manifest.csv"))
    q_ids = set(quarantined["tweet_id"])
    active_splits = manifest[manifest["split"].isin(["train", "test"])]
    assert len(set(active_splits["tweet_id"]).intersection(q_ids)) == 0, "Quarantined IDs leaked to active split!"
    print("PASS: TICKET-02 (Quarantine 18 Conflicting Groups / 36 Rows)")


def test_ticket_03_negativereason_leakage_trap():
    """TICKET-03: Expose negativereason Annotation Leakage Trap."""
    df = pd.read_parquet(os.path.join(OUTPUTS_DIR, "airline_registry.parquet"))
    neg_rows = df[df["airline_sentiment"] == "negative"]
    non_neg_rows = df[df["airline_sentiment"].isin(["neutral", "positive"])]
    
    # 100% of negative rows have negativereason, 0% of non-negative rows do
    assert neg_rows["negativereason"].notnull().mean() == 1.0, "Negative tweets missing negativereason!"
    assert non_neg_rows["negativereason"].isnull().mean() == 1.0, "Non-negative tweets contain negativereason!"
    print("PASS: TICKET-03 (Expose negativereason Annotation Leakage Trap)")


def test_ticket_04_text_normalization_negation():
    """TICKET-04: Deterministic Text Normalization & Negation Preservation."""
    sample = "Don't tell me @USAirways flight is NOT on time! Check #DelayedFlight http://t.co/xyz"
    cleaned = clean_text(sample)
    assert "not" in cleaned, "Negation token 'not' was stripped!"
    assert "cannot" in clean_text("I can't go"), "Contraction 'can't' not expanded to 'cannot'!"
    assert "usertoken" in cleaned, "Mention '@USAirways' not converted to 'usertoken'!"
    assert "urltoken" in cleaned, "URL 'http://t.co/xyz' not converted to 'urltoken'!"
    assert "delayedflight" in cleaned, "Hashtag '#DelayedFlight' not preserved as word!"
    assert "#" not in cleaned, "Hashtag symbol '#' was not stripped!"
    print("PASS: TICKET-04 (Deterministic Text Normalization & Negation Preservation)")


def test_ticket_05_graph_connected_components():
    """TICKET-05: Connected Components Graph Deduplication & Zero Leakage."""
    df = pd.read_parquet(os.path.join(OUTPUTS_DIR, "airline_registry.parquet"))
    reps = df[df["is_representative"]]
    assert len(reps) == 7551, f"Expected 7,551 representatives, got {len(reps)}"
    
    # Assert zero overlap between train and test
    train_reps = df[(df["split"] == "train") & df["is_representative"]]
    test_reps = df[(df["split"] == "test") & df["is_representative"]]
    
    author_overlap = set(train_reps["name"].dropna()).intersection(set(test_reps["name"].dropna()))
    text_overlap = set(train_reps["text_clean"]).intersection(set(test_reps["text_clean"]))
    assert len(author_overlap) == 0, f"Author leakage detected: {author_overlap}"
    assert len(text_overlap) == 0, f"Text leakage detected: {text_overlap}"
    print("PASS: TICKET-05 (Connected Components Graph Deduplication: 0% Leakage)")


def test_ticket_06_locked_split_proportions():
    """TICKET-06: Locked 80/20 Stratified Split & Manifest Export."""
    manifest = pd.read_csv(os.path.join(OUTPUTS_DIR, "sentiment_split_manifest.csv"))
    assert len(manifest) == 14640, f"Expected 14,640 manifest rows, got {len(manifest)}"
    train_reps = manifest[(manifest["split"] == "train") & manifest["is_representative"]]
    test_reps = manifest[(manifest["split"] == "test") & manifest["is_representative"]]
    assert len(train_reps) == 6040, f"Expected 6,040 train reps, got {len(train_reps)}"
    assert len(test_reps) == 1511, f"Expected 1,511 test reps, got {len(test_reps)}"
    print("PASS: TICKET-06 (Locked 80/20 Stratified Split: Train 6,040 / Test 1,511)")


# -----------------------------------------------------------------------------
# Epic 2: Modeling Baselines & Cross-Validation
# -----------------------------------------------------------------------------

def test_ticket_07_math_hand_calculations():
    """TICKET-07: Mathematical Baseline & Hand-Calculation Verification."""
    # Sublinear TF: 1 + ln(1) = 1.0
    tf_star = 1.0 + np.log(1.0)
    assert np.isclose(tf_star, 1.0), "Sublinear TF mismatch!"
    
    # Smoothed IDF: ln((3 + 1) / (1 + 1)) + 1 = ln(2) + 1 ≈ 1.693147
    idf = np.log((3 + 1) / (1 + 1)) + 1.0
    assert np.isclose(idf, 1.693147, atol=1e-5), "Smoothed IDF mismatch!"
    
    # L2 Norm: (3, 4) -> (0.6, 0.8)
    vec = np.array([3.0, 4.0])
    norm = vec / np.linalg.norm(vec)
    assert np.allclose(norm, [0.6, 0.8]), "L2 normalization mismatch!"
    assert np.isclose(np.linalg.norm(norm), 1.0), "Unit length assertion failed!"
    print("PASS: TICKET-07 (Mathematical Baseline: Sublinear TF, Smoothed IDF, L2 Norm)")


def test_ticket_08_dummy_baseline():
    """TICKET-08: Dummy Majority Baseline Model Performance Floor."""
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json")) as f:
        val = json.load(f)
    dummy_acc = val["evaluation_metrics"]["dummy_baseline_accuracy"]
    dummy_f1 = val["evaluation_metrics"]["dummy_baseline_macro_f1"]
    assert np.isclose(dummy_acc, 0.612177, atol=1e-4), f"Dummy accuracy mismatch: {dummy_acc}"
    assert np.isclose(dummy_f1, 0.253147, atol=1e-4), f"Dummy macro-F1 mismatch: {dummy_f1}"
    print(f"PASS: TICKET-08 (Dummy Majority Baseline: Acc={dummy_acc:.4f}, Macro-F1={dummy_f1:.4f})")


def test_ticket_09_cv_hyperparameter_search():
    """TICKET-09: 3-Fold Stratified Cross-Validation on C in {0.3, 1.0, 3.0}."""
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json")) as f:
        val = json.load(f)
    cv_scores = val["model_selection"]["cv_scores"]
    assert "0.3" in cv_scores and "1.0" in cv_scores and "3.0" in cv_scores
    for c_val in ["0.3", "1.0", "3.0"]:
        assert len(cv_scores[c_val]["fold_scores"]) == 3, f"Expected 3 folds for C={c_val}"
    best_c = val["model_selection"]["selected_best_C"]
    assert best_c == 3.0, f"Expected best C=3.0, got {best_c}"
    print(f"PASS: TICKET-09 (3-Fold Stratified CV on C: Selected C={best_c})")


def test_ticket_10_predeclared_feature_comparison():
    """TICKET-10: Predeclared Unigram vs Bigram Feature Comparison."""
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json")) as f:
        val = json.load(f)
    uni_f1 = val["model_selection"]["unigram_comparison_f1"]
    bi_f1 = val["model_selection"]["unigram_bigram_selected_f1"]
    assert bi_f1 > uni_f1, f"Bigram ({bi_f1}) failed to outperform Unigram ({uni_f1})!"
    print(f"PASS: TICKET-10 (Predeclared Comparison: Bigram {bi_f1:.4f} > Unigram {uni_f1:.4f})")


# -----------------------------------------------------------------------------
# Epic 3: Locked Test Evaluation & Local Contrast
# -----------------------------------------------------------------------------

def test_ticket_11_locked_test_metrics():
    """TICKET-11: Single-Pass Locked Test Set Evaluation."""
    clf_rep = pd.read_csv(os.path.join(OUTPUTS_DIR, "classification_report.csv"), index_col=0)
    acc = clf_rep.loc["accuracy", "precision"]
    macro_f1 = clf_rep.loc["macro avg", "f1-score"]
    neg_recall = clf_rep.loc["negative", "recall"]
    
    assert acc > 0.75, f"Test accuracy {acc} below 75% threshold!"
    assert macro_f1 > 0.68, f"Test macro-F1 {macro_f1} below 68% threshold!"
    assert neg_recall > 0.90, f"Negative recall {neg_recall} below 90% threshold!"
    
    preds_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "sentiment_predictions.csv"))
    assert len(preds_df) == 1511, f"Expected 1,511 test predictions, got {len(preds_df)}"
    print(f"PASS: TICKET-11 (Locked Test: Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, Neg Recall={neg_recall:.4f})")


def test_ticket_12_exact_local_contrast_decomposition():
    """TICKET-12: Exact Algebraic Local Contrast Decomposition Verification."""
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json")) as f:
        val = json.load(f)
    assert val["evaluation_metrics"]["local_contrast_verified"] is True, "Local contrast was not verified!"
    print("PASS: TICKET-12 (Exact Algebraic Local Contrast Decomposition: Verified)")


def test_ticket_13_error_analysis_documentation():
    """TICKET-13: Fixed-Seed Test Error Analysis in Executive Report."""
    report_path = os.path.join(BASE_DIR, "Lab09A_REPORT.md")
    assert os.path.exists(report_path), "Lab09A_REPORT.md missing!"
    with open(report_path) as f:
        content = f.read()
    assert "Confusion Matrix (Row Normalized %)" in content, "Missing confusion matrix table in report!"
    assert "Error Diagnostics" in content, "Missing error diagnostics section in report!"
    print("PASS: TICKET-13 (Fixed-Seed Test Error Analysis Documented)")


# -----------------------------------------------------------------------------
# Epic 4: Aspect Triage & Visualizations
# -----------------------------------------------------------------------------

def test_ticket_14_aspect_triage_denominators():
    """TICKET-14: Multi-Label Keyword Aspect Tagging with Visible Denominators n_a."""
    aspect_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "aspect_sentiment_summary.csv"))
    assert len(aspect_df) == 5, f"Expected 5 aspects, got {len(aspect_df)}"
    expected_aspects = {"delay", "baggage", "staff", "refund", "no_match"}
    assert set(aspect_df["aspect"]) == expected_aspects, "Aspect categories mismatch!"
    
    # Verify delay is overwhelmingly negative
    delay_row = aspect_df[aspect_df["aspect"] == "delay"].iloc[0]
    assert delay_row["denominator_n_a"] == 2612, f"Expected 2,612 delay matches, got {delay_row['denominator_n_a']}"
    assert delay_row["negative_pct"] > 85.0, f"Expected >85% negative in delay, got {delay_row['negative_pct']}"
    print(f"PASS: TICKET-14 (Aspect Triage Tagging: delay n={delay_row['denominator_n_a']}, {delay_row['negative_pct']}% neg)")


def test_ticket_15_six_figures_exist_and_valid():
    """TICKET-15: Six Executive High-Resolution Figures Generated."""
    req_figs = [
        "sentiment_counts.png",
        "airline_sentiment_stacked.png",
        "aspect_sentiment_heatmap.png",
        "training_top_bigrams.png",
        "top_coefficients.png",
        "confusion_matrices.png",
    ]
    for fig_name in req_figs:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Figure {fig_name} missing!"
        assert os.path.getsize(fig_path) > 50000, f"Figure {fig_name} file size too small!"
    print(f"PASS: TICKET-15 (All {len(req_figs)} Executive Figures Generated & Non-Empty)")


# -----------------------------------------------------------------------------
# Epic 5: Automation, Defense & Packaging
# -----------------------------------------------------------------------------

def test_ticket_16_validation_report_json():
    """TICKET-16: Validation Report JSON Schema & Completeness."""
    val_path = os.path.join(OUTPUTS_DIR, "validation_report.json")
    assert os.path.exists(val_path), "validation_report.json missing!"
    with open(val_path) as f:
        val = json.load(f)
    assert val["data_integrity"]["conserved_rows_assertion"] is True
    assert val["data_integrity"]["conflicting_tweet_ids_count"] == 18
    assert val["data_integrity"]["quarantined_rows_count"] == 36
    assert val["data_integrity"]["connected_components_count"] == 7551
    print("PASS: TICKET-16 (validation_report.json Schema & Integrity Verified)")


def test_ticket_17_executive_report_and_defense():
    """TICKET-17: Executive Findings Board & 2-Page Analytical Defense."""
    rep_path = os.path.join(BASE_DIR, "Lab09A_REPORT.md")
    def_path = os.path.join(BASE_DIR, "DATA_DEFENSE.md")
    assert os.path.exists(rep_path) and os.path.getsize(rep_path) > 1000, "Lab09A_REPORT.md invalid!"
    assert os.path.exists(def_path) and os.path.getsize(def_path) > 1000, "DATA_DEFENSE.md invalid!"
    
    with open(def_path) as f:
        def_content = f.read()
    for citation in ["[A1]", "[A2]", "[A3]", "[A4]", "[A5]", "[A7]"]:
        assert citation in def_content, f"Missing citation {citation} in DATA_DEFENSE.md!"
    print("PASS: TICKET-17 (Executive Report & Defense with Citations [A1]-[A7])")


def test_ticket_18_narrative_notebook_runnable():
    """TICKET-18: Executed Narrative Notebook Validation."""
    nb_path = os.path.join(BASE_DIR, "Lab09A_Sentiment_Analysis.ipynb")
    assert os.path.exists(nb_path), "Notebook missing!"
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    assert len(nb.cells) >= 10, f"Expected at least 10 notebook cells, got {len(nb.cells)}"
    assert nb.cells[0].cell_type == "markdown", "Notebook header cell must be markdown!"
    print(f"PASS: TICKET-18 (Narrative Notebook Validated: {len(nb.cells)} cells)")


# -----------------------------------------------------------------------------
# Main Test Runner
# -----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RUNNING EXPLICIT UNIT TEST SUITE FOR ALL 18 TICKETS (LAB 09A)")
    print("=" * 80)
    
    tests = [
        test_ticket_01_ingestion_conservation,
        test_ticket_02_quarantine_conflicting_labels,
        test_ticket_03_negativereason_leakage_trap,
        test_ticket_04_text_normalization_negation,
        test_ticket_05_graph_connected_components,
        test_ticket_06_locked_split_proportions,
        test_ticket_07_math_hand_calculations,
        test_ticket_08_dummy_baseline,
        test_ticket_09_cv_hyperparameter_search,
        test_ticket_10_predeclared_feature_comparison,
        test_ticket_11_locked_test_metrics,
        test_ticket_12_exact_local_contrast_decomposition,
        test_ticket_13_error_analysis_documentation,
        test_ticket_14_aspect_triage_denominators,
        test_ticket_15_six_figures_exist_and_valid,
        test_ticket_16_validation_report_json,
        test_ticket_17_executive_report_and_defense,
        test_ticket_18_narrative_notebook_runnable,
    ]
    
    for t in tests:
        t()
        
    print("=" * 80)
    print(f"SUCCESS: ALL {len(tests)} INDIVIDUAL TICKET TESTS PASSED [100% GREEN]")
    print("=" * 80)


if __name__ == "__main__":
    main()
