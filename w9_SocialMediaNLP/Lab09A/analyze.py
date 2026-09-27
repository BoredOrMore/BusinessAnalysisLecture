#!/usr/bin/env python3
"""
Business Data Analytics — Course 961731
Week 09A: Social Media Sentiment Analysis & Operational Triage

Stakeholder Context:
    Target Audience: VP of Customer Experience (CX) & Customer Support Operations Leadership.
    Business Problem:
        Social media complaint volumes surge during flight disruptions. Manual triage is slow and expensive.
        Naive sentiment classifiers mistake post topics (e.g. 'baggage') for negative polarity, or fail
        under imbalanced class distributions. Furthermore, uncurated social datasets contain duplicate
        posts and conflicting human annotations that contaminate automated triage systems.
    Analytical Scope:
        1. Ingestion & Audit: Process 14,640 CrowdFlower airline tweets, enforcing row conservation ($N=14,640$)
           and quarantining 18 conflicting-annotation tweet_id groups (36 rows).
        2. Representation & Deduplication: Clean and normalize text while strictly preserving negation tokens.
           Form graph connected components on shared authors/texts to isolate 7,551 independent modeling units.
        3. Supervised Classification: Compare a Dummy majority baseline with a Sparse TF-IDF Multinomial
           Logistic Regression pipeline under 3-fold stratified cross-validation (seed 961731) on C in {0.3, 1.0, 3.0}.
        4. Locked Test Evaluation & Local Contrast: Single-pass out-of-sample evaluation reporting Macro-F1,
           multiclass log loss, confusion matrices, top coefficients, and exact algebraic local contrast verification.
        5. Aspect Triage & Visualization: Multi-label keyword matching across 4 operational dictionaries
           ('delay', 'baggage', 'staff', 'refund') with transparent denominator reporting and 6 executive figures.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# Text Normalization & Cleaning Contract
# -----------------------------------------------------------------------------

CONTRACTIONS = {
    r"\bcan\'t\b": "cannot",
    r"\bwon\'t\b": "will not",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'t\b": " not",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am",
}


def clean_text(text: str) -> str:
    """
    Deterministic normalization:
    - Unicode NFKC normalization
    - URL masking -> 'urltoken'
    - Mention masking -> 'usertoken'
    - Hashtag word preservation (strip '#')
    - Contraction expansion
    - Strict negation preservation ('not', 'no', 'never', 'neither', 'nor', 'without')
    - Lowercasing and whitespace collapse
    """
    if not isinstance(text, str):
        return ""
    # Unicode NFKC
    norm = unicodedata.normalize("NFKC", text)
    # Mask URLs and Mentions
    norm = re.sub(r"https?://\S+|www\.\S+", "urltoken", norm)
    norm = re.sub(r"@[A-Za-z0-9_]+", "usertoken", norm)
    # Strip hashtag symbol, retain word
    norm = re.sub(r"#([A-Za-z0-9_]+)", r"\1", norm)
    # Expand contractions
    for pat, repl in CONTRACTIONS.items():
        norm = re.sub(pat, repl, norm, flags=re.IGNORECASE)
    # Lowercase
    norm = norm.lower()
    # Normalize whitespace
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


# -----------------------------------------------------------------------------
# Aspect Keyword Dictionaries (Non-capturing groups)
# -----------------------------------------------------------------------------

ASPECT_PATTERNS = {
    "delay": r"\b(?:delay|delayed|delays|late|cancel|cancelled|cancelling|cancellation|cancellations|holding|hours? late|on hold)\b",
    "baggage": r"\b(?:bag|bags|baggage|luggage|suitcase|suitcases|lost bag|claim)\b",
    "staff": r"\b(?:staff|crew|agent|agents|attendant|attendants|representative|representatives|flight attendant|gate agent|employee|employees|pilot|pilots)\b",
    "refund": r"\b(?:refund|refunds|refunded|refunding|voucher|vouchers|reimburse|reimbursement|compensation|fee|fees|credit|credits)\b",
}


# -----------------------------------------------------------------------------
# Main Analysis Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Lab 09A Sentiment Classification & Operational Triage Pipeline."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(script_dir, "data", "Tweets.csv"),
        help="Path to CrowdFlower Tweets.csv dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=961731,
        help="Deterministic random seed (course standard: 961731).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(script_dir, "Lab09A_outputs"),
        help="Directory to save parquet, csv, and json deliverables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=os.path.join(script_dir, "figures"),
        help="Directory to save generated .png figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="deep")
    class_order = ["negative", "neutral", "positive"]
    class_palette = {"negative": "#c44e52", "neutral": "#8c8c8c", "positive": "#4c72b0"}

    print("=" * 80)
    print("CMU DATA SCIENCE 961731: LAB 09A OPERATIONAL SENTIMENT & TRIAGE")
    print("=" * 80)
    print(f"Data source       : {args.data}")
    print(f"Random seed       : {args.seed}")
    print(f"Artifacts output  : {args.output_dir}")
    print(f"Figures output    : {args.figures_dir}")
    print("-" * 80)

    # 1. Ingestion & Conservation
    if not os.path.exists(args.data):
        sys.exit(f"Error: dataset file not found at {args.data}")

    raw_df = pd.read_csv(args.data)
    n_source_rows = len(raw_df)
    print(f"\n[1/6] Ingesting dataset: {n_source_rows:,} raw records.")
    assert n_source_rows == 14640, f"Expected 14,640 rows, got {n_source_rows}"

    # Build primary registry
    registry = raw_df.copy()
    registry["source_row"] = registry.index.astype(int)
    registry["tweet_id"] = registry["tweet_id"].astype(str)
    registry["tweet_created_utc"] = pd.to_datetime(registry["tweet_created"], errors="coerce", utc=True)
    registry["text_clean"] = registry["text"].apply(clean_text)

    # Injected / Observed Defect Audit
    conflicts_by_id = registry.groupby("tweet_id")["airline_sentiment"].nunique()
    conflicting_ids = set(conflicts_by_id[conflicts_by_id > 1].index)
    n_conflict_ids = len(conflicting_ids)
    print(f"  Observed conflicting-label tweet_id groups: {n_conflict_ids} groups")

    # Quarantine conflicting rows
    is_conflicted = registry["tweet_id"].isin(conflicting_ids)
    n_quarantined = is_conflicted.sum()
    print(f"  Quarantining all rows in conflicting groups: {n_quarantined} rows (Strict Isolation).")

    # Audit flags
    registry["audit_flags"] = ""
    registry.loc[is_conflicted, "audit_flags"] = "CONFLICTING_LABEL"
    registry.loc[registry["text_clean"] == "", "audit_flags"] = registry["audit_flags"].str.cat(
        ["EMPTY_TEXT"] * len(registry), sep=";"
    ).str.strip(";")

    # Model eligibility before deduplication
    valid_sentiment = registry["airline_sentiment"].isin(class_order)
    valid_time = registry["tweet_created_utc"].notnull()
    valid_text = registry["text_clean"] != ""
    eligible_pool = (~is_conflicted) & valid_sentiment & valid_time & valid_text
    print(f"  Non-quarantined eligible rows before deduplication: {eligible_pool.sum():,}")

    # 2. Component-Based Deduplication (Author & Normalized Text)
    print("\n[2/6] Building Graph Connected Components for Leakage Prevention...")
    G = nx.Graph()
    for idx, r in registry[eligible_pool].iterrows():
        r_node = f"row_{idx}"
        G.add_node(r_node)
        if pd.notnull(r["name"]) and str(r["name"]).strip() != "":
            G.add_edge(r_node, f"author_{str(r['name']).strip()}")
        if r["text_clean"] != "":
            G.add_edge(r_node, f"text_{r['text_clean']}")

    registry["component_id"] = -1
    registry["is_representative"] = False

    representatives = []
    comp_counter = 0
    for comp in nx.connected_components(G):
        row_indices = [int(n.replace("row_", "")) for n in comp if n.startswith("row_")]
        if not row_indices:
            continue
        registry.loc[row_indices, "component_id"] = comp_counter
        # Earliest representative: sort by UTC time, tie-break by tweet_id string, then source_row
        sub = registry.loc[row_indices].sort_values(
            by=["tweet_created_utc", "tweet_id", "source_row"], ascending=[True, True, True]
        )
        rep_idx = sub.index[0]
        representatives.append(rep_idx)
        registry.loc[rep_idx, "is_representative"] = True
        comp_counter += 1

    n_components = comp_counter
    print(f"  Total independent connected components: {n_components:,}")
    print(f"  Selected earliest representatives      : {len(representatives):,} units")

    registry["model_eligible"] = registry["is_representative"] & eligible_pool

    # 3. Stratified Split (80% Train / 20% Locked Test)
    print("\n[3/6] Generating Locked Stratified 80/20 Train/Test Split...")
    rep_df = registry[registry["model_eligible"]].copy()

    train_idx, test_idx = train_test_split(
        rep_df.index,
        test_size=0.20,
        random_state=args.seed,
        stratify=rep_df["airline_sentiment"],
    )

    registry["split"] = "ineligible"
    registry.loc[is_conflicted, "split"] = "quarantined"

    # Assign split to all rows in the component to guarantee absolute isolation
    comp_to_split = {}
    for idx in train_idx:
        comp_to_split[registry.loc[idx, "component_id"]] = "train"
    for idx in test_idx:
        comp_to_split[registry.loc[idx, "component_id"]] = "test"

    for r_idx in registry[eligible_pool].index:
        c_id = registry.loc[r_idx, "component_id"]
        if c_id in comp_to_split:
            registry.loc[r_idx, "split"] = comp_to_split[c_id]

    train_reps = registry[(registry["split"] == "train") & registry["is_representative"]]
    test_reps = registry[(registry["split"] == "test") & registry["is_representative"]]

    print(f"  Training representative units : {len(train_reps):,} ({len(train_reps)/len(rep_df)*100:.1f}%)")
    print(f"  Locked Test representative units: {len(test_reps):,} ({len(test_reps)/len(rep_df)*100:.1f}%)")

    # Verify zero leakage across author and text
    train_authors = set(train_reps["name"].dropna())
    test_authors = set(test_reps["name"].dropna())
    author_overlap = train_authors.intersection(test_authors)
    train_texts = set(train_reps["text_clean"])
    test_texts = set(test_reps["text_clean"])
    text_overlap = train_texts.intersection(test_texts)
    assert len(author_overlap) == 0, f"Author leakage detected: {author_overlap}"
    assert len(text_overlap) == 0, f"Text leakage detected: {text_overlap}"
    print("  Assertion Passed: 0% Author and 0% Normalized-Text overlap between Train and Test.")

    # 4. Supervised Model Training & Cross-Validation
    print("\n[4/6] Cross-Validation & Model Selection (Training Data Only)...")

    # Baseline: Dummy majority prior
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_reps["text_clean"], train_reps["airline_sentiment"])
    dummy_test_preds = dummy.predict(test_reps["text_clean"])
    dummy_acc = accuracy_score(test_reps["airline_sentiment"], dummy_test_preds)
    dummy_macro_f1 = f1_score(test_reps["airline_sentiment"], dummy_test_preds, average="macro", zero_division=0)
    print(f"  Dummy Baseline -> Accuracy: {dummy_acc:.4f} | Macro-F1: {dummy_macro_f1:.4f}")

    # 3-Fold Stratified Cross-Validation for C in {0.3, 1.0, 3.0}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    c_candidates = [0.3, 1.0, 3.0]
    cv_results = {}

    for c in c_candidates:
        fold_scores = []
        for fold, (tr_i, val_i) in enumerate(skf.split(train_reps["text_clean"], train_reps["airline_sentiment"])):
            fold_train = train_reps.iloc[tr_i]
            fold_val = train_reps.iloc[val_i]

            pipe = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        smooth_idf=True,
                        norm="l2",
                        max_features=5000,
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        C=c,
                        max_iter=1000,
                        random_state=args.seed,
                    ),
                ),
            ])
            pipe.fit(fold_train["text_clean"], fold_train["airline_sentiment"])
            val_pred = pipe.predict(fold_val["text_clean"])
            fold_scores.append(f1_score(fold_val["airline_sentiment"], val_pred, average="macro", zero_division=0))

        mean_f1 = float(np.mean(fold_scores))
        cv_results[c] = {"fold_scores": fold_scores, "mean_macro_f1": mean_f1}
        print(f"  Candidate C={c:<3} -> 3-Fold Macro-F1: {mean_f1:.4f} (folds: {[round(s, 4) for s in fold_scores]})")

    # Select best C
    best_c = max(cv_results.keys(), key=lambda k: cv_results[k]["mean_macro_f1"])
    print(f"  Selected Optimal Hyperparameter: C = {best_c} (Mean Macro-F1 = {cv_results[best_c]['mean_macro_f1']:.4f})")

    # Predeclared comparison: Unigram-only vs Unigram+Bigram
    unigram_scores = []
    for fold, (tr_i, val_i) in enumerate(skf.split(train_reps["text_clean"], train_reps["airline_sentiment"])):
        fold_train = train_reps.iloc[tr_i]
        fold_val = train_reps.iloc[val_i]
        pipe_uni = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, smooth_idf=True, norm="l2", max_features=5000, min_df=2)),
            ("clf", LogisticRegression(solver="lbfgs", C=best_c, max_iter=1000, random_state=args.seed)),
        ])
        pipe_uni.fit(fold_train["text_clean"], fold_train["airline_sentiment"])
        v_pred = pipe_uni.predict(fold_val["text_clean"])
        unigram_scores.append(f1_score(fold_val["airline_sentiment"], v_pred, average="macro", zero_division=0))
    mean_uni_f1 = float(np.mean(unigram_scores))
    print(f"  Predeclared Feature Comparison (at C={best_c}):")
    print(f"    - Unigram-only (1, 1)    : Macro-F1 = {mean_uni_f1:.4f}")
    print(f"    - Unigram + Bigram (1, 2): Macro-F1 = {cv_results[best_c]['mean_macro_f1']:.4f} (Selected)")

    # 5. Locked Test Set Evaluation
    print("\n[5/6] Opening Locked Test Set Evaluation (Single Pass)...")
    final_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                smooth_idf=True,
                norm="l2",
                max_features=5000,
                min_df=2,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                C=best_c,
                max_iter=1000,
                random_state=args.seed,
            ),
        ),
    ])
    final_pipeline.fit(train_reps["text_clean"], train_reps["airline_sentiment"])

    test_preds = final_pipeline.predict(test_reps["text_clean"])
    test_probs = final_pipeline.predict_proba(test_reps["text_clean"])

    test_acc = accuracy_score(test_reps["airline_sentiment"], test_preds)
    test_macro_f1 = f1_score(test_reps["airline_sentiment"], test_preds, average="macro", zero_division=0)
    test_loss = log_loss(test_reps["airline_sentiment"], test_probs, labels=final_pipeline.classes_)

    print(f"  Locked Test Metrics:")
    print(f"    - Accuracy          : {test_acc:.4f} (vs Dummy: {dummy_acc:.4f})")
    print(f"    - Macro-averaged F1 : {test_macro_f1:.4f} (vs Dummy: {dummy_macro_f1:.4f})")
    print(f"    - Multiclass Log Loss: {test_loss:.4f}")

    # Confusion Matrices
    cm = confusion_matrix(test_reps["airline_sentiment"], test_preds, labels=class_order)
    cm_norm = confusion_matrix(test_reps["airline_sentiment"], test_preds, labels=class_order, normalize="true")

    print("\n  Confusion Matrix (Row Normalized %):")
    cm_df = pd.DataFrame(cm_norm * 100, index=class_order, columns=[f"Pred_{c}" for c in class_order])
    print(cm_df.round(1).to_string())

    # Classification Report
    clf_rep_dict = classification_report(
        test_reps["airline_sentiment"],
        test_preds,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    clf_rep_df = pd.DataFrame(clf_rep_dict).transpose()

    # Local Contrast Decomposition Verification
    print("\n  Verifying Exact Local Contrast Decomposition on Test Instance...")
    vec = final_pipeline.named_steps["tfidf"]
    clf = final_pipeline.named_steps["clf"]
    classes = list(clf.classes_)

    sample_row_idx = 0
    sample_text = test_reps["text_clean"].iloc[sample_row_idx]
    sample_x = vec.transform([sample_text])
    sample_probs = test_probs[sample_row_idx]
    rank_order = np.argsort(sample_probs)[::-1]
    winner_k = rank_order[0]
    runner_up_j = rank_order[1]

    # Direct Logit Difference: z_k - z_j
    logits = clf.intercept_ + clf.coef_ @ sample_x.toarray().flatten()
    delta_z_direct = logits[winner_k] - logits[runner_up_j]

    # Decomposed Logit Difference: (b_k - b_j) + sum (w_ki - w_ji) * x_i
    delta_b = clf.intercept_[winner_k] - clf.intercept_[runner_up_j]
    word_diffs = (clf.coef_[winner_k] - clf.coef_[runner_up_j]) * sample_x.toarray().flatten()
    delta_z_decomposed = delta_b + np.sum(word_diffs)

    contrast_verified = bool(np.isclose(delta_z_direct, delta_z_decomposed, atol=1e-5))
    assert contrast_verified, "Decomposition mismatch!"
    print(f"    - Sample Text      : '{sample_text[:60]}...'")
    print(f"    - Winner vs Runner : {classes[winner_k]} vs {classes[runner_up_j]}")
    print(f"    - Direct Delta z   : {delta_z_direct:.6f}")
    print(f"    - Decomposed Delta : {delta_z_decomposed:.6f}")
    print(f"    - Algebraic Match  : {contrast_verified} (Assertion Passed)")

    # 6. Aspect Triage & Figures
    print("\n[6/6] Aspect Triage Tagging & Figure Generation...")
    for aspect, pattern in ASPECT_PATTERNS.items():
        registry[f"aspect_{aspect}"] = registry["text"].str.contains(pattern, case=False, regex=True, na=False)

    aspect_cols = [f"aspect_{a}" for a in ASPECT_PATTERNS]
    registry["aspect_matches_count"] = registry[aspect_cols].sum(axis=1)
    registry["aspect_no_match"] = registry["aspect_matches_count"] == 0

    all_aspects = list(ASPECT_PATTERNS.keys()) + ["no_match"]
    aspect_summary_rows = []
    for a in all_aspects:
        col = f"aspect_{a}"
        sub = registry[registry[col]]
        n_a = len(sub)
        counts = sub["airline_sentiment"].value_counts().to_dict()
        neg_c = counts.get("negative", 0)
        neu_c = counts.get("neutral", 0)
        pos_c = counts.get("positive", 0)
        aspect_summary_rows.append({
            "aspect": a,
            "denominator_n_a": n_a,
            "negative_count": neg_c,
            "neutral_count": neu_c,
            "positive_count": pos_c,
            "negative_pct": round(neg_c / n_a * 100, 2) if n_a > 0 else 0.0,
            "neutral_pct": round(neu_c / n_a * 100, 2) if n_a > 0 else 0.0,
            "positive_pct": round(pos_c / n_a * 100, 2) if n_a > 0 else 0.0,
        })
    aspect_summary_df = pd.DataFrame(aspect_summary_rows)

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------

    # Figure 1: Sentiment Counts Bar Chart
    plt.figure(figsize=(8, 5))
    s_counts = registry["airline_sentiment"].value_counts()[class_order]
    s_pcts = (s_counts / len(registry) * 100)
    ax = sns.barplot(x=s_counts.index, y=s_counts.values, hue=s_counts.index, palette=class_palette, legend=False)
    for i, (cnt, pct) in enumerate(zip(s_counts.values, s_pcts.values)):
        ax.text(i, cnt + 150, f"{cnt:,} ({pct:.1f}%)", ha="center", fontweight="bold", fontsize=11)
    plt.title(f"Observed Sentiment Class Balance (Source Corpus: N={len(registry):,})", fontsize=13, fontweight="bold")
    plt.xlabel("Sentiment Class", fontsize=11)
    plt.ylabel("Tweet Count", fontsize=11)
    plt.ylim(0, s_counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "sentiment_counts.png"), dpi=300)
    plt.close()

    # Figure 2: Airline Sentiment 100% Stacked Bar
    plt.figure(figsize=(10, 6))
    airline_sent = pd.crosstab(registry["airline"], registry["airline_sentiment"], normalize="index")[class_order] * 100
    bottom = np.zeros(len(airline_sent))
    for c in class_order:
        vals = airline_sent[c].values
        plt.bar(airline_sent.index, vals, bottom=bottom, label=c.capitalize(), color=class_palette[c], edgecolor="white")
        for idx_pos, (val, b_val) in enumerate(zip(vals, bottom)):
            if val > 5:
                plt.text(idx_pos, b_val + val / 2, f"{val:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        bottom += vals
    plt.title("Sentiment Proportions by Airline (100% Stacked Distribution)", fontsize=13, fontweight="bold")
    plt.ylabel("Proportion of Inbound Tweets (%)", fontsize=11)
    plt.xlabel("Airline Operator", fontsize=11)
    plt.legend(title="Sentiment", loc="upper right")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "airline_sentiment_stacked.png"), dpi=300)
    plt.close()

    # Figure 3: Aspect Sentiment Heatmap
    plt.figure(figsize=(9, 6))
    heatmap_data = aspect_summary_df.set_index("aspect")[["negative_pct", "neutral_pct", "positive_pct"]]
    heatmap_data.columns = ["Negative (%)", "Neutral (%)", "Positive (%)"]
    heatmap_data.index = [f"{row['aspect'].capitalize()} (n={row['denominator_n_a']:,})" for _, row in aspect_summary_df.iterrows()]
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Proportion within Aspect Tag (%)"}, annot_kws={"fontsize": 11, "fontweight": "bold"})
    plt.title("Aspect Triage Heatmap: Sentiment Distribution by Keyword Tag", fontsize=13, fontweight="bold")
    plt.ylabel("Operational Aspect (with Denominator $n_a$)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "aspect_sentiment_heatmap.png"), dpi=300)
    plt.close()

    # Figure 4: Training Top Bigrams Document Frequency
    plt.figure(figsize=(10, 6))
    vec_bigram = TfidfVectorizer(ngram_range=(2, 2), min_df=5, binary=True)
    X_bi = vec_bigram.fit_transform(train_reps["text_clean"])
    bi_df_counts = np.asarray((X_bi > 0).sum(axis=0)).flatten()
    bi_names = np.asarray(vec_bigram.get_feature_names_out())
    top_bi_indices = np.argsort(bi_df_counts)[::-1][:15]
    top_bi_df = pd.DataFrame({"Bigram": bi_names[top_bi_indices], "DocFrequency": bi_df_counts[top_bi_indices]})
    ax_bi = sns.barplot(x="DocFrequency", y="Bigram", data=top_bi_df, color="#4c72b0")
    for i, v in enumerate(top_bi_df["DocFrequency"]):
        ax_bi.text(v + 5, i, f"{v:,}", va="center", fontsize=10, fontweight="bold")
    plt.title("Top 15 Bigrams in Training Corpus (by Document Frequency)", fontsize=13, fontweight="bold")
    plt.xlabel("Document Frequency (Distinct Training Representatives)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "training_top_bigrams.png"), dpi=300)
    plt.close()

    # Figure 5: Top 10 Coefficients per Class
    plt.figure(figsize=(14, 5))
    feature_names = np.asarray(vec.get_feature_names_out())
    for class_idx, class_name in enumerate(class_order):
        c_i = classes.index(class_name)
        class_coefs = clf.coef_[c_i]
        top_coef_idx = np.argsort(class_coefs)[::-1][:10]
        sub_df = pd.DataFrame({
            "Feature": feature_names[top_coef_idx],
            "Coefficient": class_coefs[top_coef_idx],
        })
        plt.subplot(1, 3, class_idx + 1)
        sns.barplot(x="Coefficient", y="Feature", data=sub_df, color=class_palette[class_name])
        plt.title(f"Class: {class_name.upper()}", fontsize=12, fontweight="bold")
        plt.xlabel("Linear Model Weight")
    plt.suptitle("Top 10 Positive Predictive Coefficients by Sentiment Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "top_coefficients.png"), dpi=300)
    plt.close()

    # Figure 6: Confusion Matrices (Raw Count & Row Normalized)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=class_order, yticklabels=class_order)
    axes[0].set_title("Test Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Ground Truth")

    sns.heatmap(cm_norm * 100, annot=True, fmt=".1f", cmap="Blues", ax=axes[1], xticklabels=class_order, yticklabels=class_order)
    axes[1].set_title("Test Confusion Matrix (Row Normalized %)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Ground Truth")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    print(f"  All 6 executive figures saved successfully in '{args.figures_dir}'.")

    # -------------------------------------------------------------------------
    # Deliverables & Artifact Exports
    # -------------------------------------------------------------------------
    print("\nWriting deliverables to Lab09A_outputs/...")

    # 1. airline_registry.parquet
    registry_export = registry.copy()
    registry_export.to_parquet(os.path.join(args.output_dir, "airline_registry.parquet"), index=False)
    print(f"  - Saved airline_registry.parquet ({len(registry_export):,} rows)")

    # 2. sentiment_split_manifest.csv
    split_manifest = registry[["source_row", "tweet_id", "split", "is_representative", "airline_sentiment"]].copy()
    split_manifest.to_csv(os.path.join(args.output_dir, "sentiment_split_manifest.csv"), index=False)
    print(f"  - Saved sentiment_split_manifest.csv ({len(split_manifest):,} rows)")

    # 3. sentiment_predictions.csv
    test_pred_df = test_reps[["source_row", "tweet_id", "airline_sentiment", "text_clean"]].copy()
    test_pred_df = test_pred_df.rename(columns={"airline_sentiment": "true_sentiment"})
    test_pred_df["pred_sentiment"] = test_preds
    for c_i, c_name in enumerate(classes):
        test_pred_df[f"prob_{c_name}"] = test_probs[:, c_i]
    test_pred_df["is_correct"] = test_pred_df["true_sentiment"] == test_pred_df["pred_sentiment"]
    test_pred_df.to_csv(os.path.join(args.output_dir, "sentiment_predictions.csv"), index=False)
    print(f"  - Saved sentiment_predictions.csv ({len(test_pred_df):,} test evaluations)")

    # 4. classification_report.csv
    clf_rep_df.to_csv(os.path.join(args.output_dir, "classification_report.csv"), index=True)
    print("  - Saved classification_report.csv")

    # 5. aspect_sentiment_summary.csv
    aspect_summary_df.to_csv(os.path.join(args.output_dir, "aspect_sentiment_summary.csv"), index=False)
    print("  - Saved aspect_sentiment_summary.csv")

    # 6. validation_report.json
    validation_report = {
        "metadata": {
            "course": "961731 Business Data Analytics",
            "assignment": "Lab 09A — Social Media Sentiment Analysis & Operational Triage",
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": args.seed,
            "system_python": sys.version.split()[0],
        },
        "data_integrity": {
            "total_source_rows": n_source_rows,
            "conserved_rows_assertion": bool(n_source_rows == 14640),
            "conflicting_tweet_ids_count": n_conflict_ids,
            "quarantined_rows_count": int(n_quarantined),
            "connected_components_count": n_components,
            "train_representatives": len(train_reps),
            "test_representatives": len(test_reps),
            "author_overlap_count": len(author_overlap),
            "text_overlap_count": len(text_overlap),
        },
        "model_selection": {
            "cv_strategy": "3-fold StratifiedKFold on Training Set",
            "candidate_C": c_candidates,
            "cv_scores": cv_results,
            "selected_best_C": best_c,
            "unigram_comparison_f1": mean_uni_f1,
            "unigram_bigram_selected_f1": cv_results[best_c]["mean_macro_f1"],
        },
        "evaluation_metrics": {
            "dummy_baseline_accuracy": float(dummy_acc),
            "dummy_baseline_macro_f1": float(dummy_macro_f1),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_multiclass_log_loss": float(test_loss),
            "local_contrast_verified": contrast_verified,
        },
        "aspect_triage": {
            "aspects_analyzed": all_aspects,
            "aspect_counts": {r["aspect"]: r["denominator_n_a"] for _, r in aspect_summary_df.iterrows()},
        },
    }
    with open(os.path.join(args.output_dir, "validation_report.json"), "w") as f:
        json.dump(validation_report, f, indent=2)
    print("  - Saved validation_report.json")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY: ALL INVARIANTS & DELIVERABLES VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
