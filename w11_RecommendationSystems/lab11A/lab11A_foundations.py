#!/usr/bin/env python3
"""
Lab 11A - Recommendation System Foundations (Course 961731, Week 11).

Business context: before the CMO funds a personalized recommender, the analytics
team must prove (1) the rating log is valid and split without future leakage,
(2) the user-item matrix is represented honestly (missing = unknown, not zero),
and (3) a popularity "bestseller list" baseline is scored on a fixed shelf with
explicit denominators. Every later model (Labs 11B, 11C) must beat this control
under the same contract.

Dataset: data/interaction_events.csv (UserID, ProductID, Rating 1-5, Unix Timestamp).
Tasks: 1.1-1.3 inspect/validate/partition, 2.1-2.3 map/represent/diagnose,
3.1-3.3 popularity rank/calculate/qualify.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import reco_contract as rc  # noqa: E402

plt.style.use("default")
sns.set_theme(style="whitegrid", palette="deep")

FIG_DIR = os.path.join(HERE, "figures")
EVIDENCE_DIR = os.path.join(HERE, "evidence")


def banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def problem1(df):
    banner("PROBLEM 1.1  INSPECT RAW INTERACTIONS")
    print(f"Source file : {os.path.relpath(rc.EVENTS_CSV, os.path.dirname(HERE))}")
    print(f"SHA256      : {rc.sha256_of(rc.EVENTS_CSV)}")
    print(f"Shape       : {df.shape[0]} events x {len(rc.REQUIRED_COLUMNS)} source columns")
    print(f"Users       : {df['UserID'].nunique()} | Products: {df['ProductID'].nunique()}")
    print("\nFirst five raw records:")
    print(df[rc.REQUIRED_COLUMNS].head(5).to_string(index=False))
    print("\nColumn dtypes after strict parsing:")
    print(df.dtypes.to_string())
    print("\nRating frequencies:")
    freq = df["Rating"].value_counts().sort_index()
    for rating, n in freq.items():
        print(f"  Rating {int(rating)} : {n:>4d} ({n / len(df):6.2%})")
    ev = df.iloc[0]
    print(f"\nTimestamp conversion for event 0: {ev['Timestamp']} s -> {ev['EventTime'].isoformat()}")
    print(f"Observed time range: {df['EventTime'].min().isoformat()} to {df['EventTime'].max().isoformat()}")
    print("UserID/ProductID are labels: arithmetic on them is meaningless, so they stay strings.")

    banner("PROBLEM 1.2  VALIDATE THE CONTRACT")
    rc.validate_events(df)
    print("PASS: required columns, no blank IDs, integer ratings in [1,5], valid timestamps,")
    print("      zero duplicated (UserID, ProductID) pairs.")
    broken = df.copy()
    broken.loc[0, "Rating"] = 7
    try:
        rc.validate_events(broken)
        raise RuntimeError("Validation should have failed on the corrupted copy")
    except AssertionError as err:
        print(f"Corrupted COPY (row 0 Rating=7) fails as intended -> AssertionError: {err}")
    assert df.loc[0, "Rating"] != 7, "Source frame must remain unchanged"
    print("Source frame unchanged after the test.")

    banner("PROBLEM 1.3  TEMPORAL PARTITION")
    train, valid, test = rc.temporal_split(df)
    print(f"Cutoffs: train < {rc.TRAIN_END.date()} <= valid < {rc.VALID_END.date()} <= test")
    rows = []
    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        rows.append({
            "partition": name, "events": len(part), "users": part["UserID"].nunique(),
            "items": part["ProductID"].nunique(),
            "min_utc": part["EventTime"].min().isoformat(), "max_utc": part["EventTime"].max().isoformat(),
        })
    split = pd.DataFrame(rows)
    print(split.to_string(index=False))
    assert split["events"].tolist() == [121, 38, 41], "Split counts differ from the lab contract"
    print("Training model is frozen in both holdout windows: no refit and no parameter change after")
    print("21 May, so validation and test scores measure what the shop could have known at that date.")
    return train, valid, test, split


def problem2(train, valid, test):
    banner("PROBLEM 2.1  REVERSIBLE ID MAPPINGS (TRAINING ONLY)")
    users, items, u2i, p2i = rc.fit_mappings(train)
    rows, cols, vals, observed = rc.coo_arrays(train, u2i, p2i)
    assert [users[r] for r in rows] == train["UserID"].tolist()
    assert [items[c] for c in cols] == train["ProductID"].tolist()
    print(f"U = {len(users)} users, I = {len(items)} items; every coordinate maps back to its original ID.")
    assert (len(users), len(items)) == (71, 70)

    banner("PROBLEM 2.2  SPARSE COORDINATE REPRESENTATION")
    cells = len(users) * len(items)
    density = len(vals) / cells
    print(f"Stored observations E = {len(vals)} (row, col, rating, observed arrays of equal length)")
    print(f"Conceptual matrix {len(users)} x {len(items)} = {cells:,} cells; density = {density:.6f}; "
          f"sparsity = {1 - density:.6f}")
    assert len(vals) == 121 and cells == 4970
    print("\nFirst five coordinates:")
    print(pd.DataFrame({"row": rows, "col": cols, "rating": vals, "observed": observed}).head().to_string(index=False))

    top_p = rc.popularity_ranking(train)["ProductID"].head(8)
    top_u = (train[train["ProductID"].isin(top_p)].groupby("UserID").size()
             .sort_values(ascending=False, kind="mergesort").head(8).index)
    sub = train[train["UserID"].isin(top_u) & train["ProductID"].isin(top_p)]
    sub = sub.pivot(index="UserID", columns="ProductID", values="Rating").reindex(index=sorted(top_u), columns=sorted(top_p))
    print("\nSmall submatrix (NaN = unknown, NOT a low rating):")
    print(sub.to_string(na_rep="  ?"))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(sub, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=5, mask=sub.isna(),
                cbar_kws={"label": "Observed rating"}, linewidths=0.5, linecolor="lightgray", ax=ax)
    ax.set_facecolor("#d9d9d9")
    ax.grid(False)
    ax.set_title("Training submatrix: grey cells are unknown, not zero")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "masked_submatrix.png"), dpi=150)
    plt.close(fig)

    banner("PROBLEM 2.3  SPARSITY AND COLD-START DIAGNOSIS")
    per_user = train.groupby("UserID").size()
    per_item = train.groupby("ProductID").size()
    print(f"Interactions per user: mean {per_user.mean():.2f}, median {per_user.median():.0f}, max {per_user.max()}")
    print(f"Interactions per item: mean {per_item.mean():.2f}, median {per_item.median():.0f}, max {per_item.max()}")
    diag = {}
    for name, part in [("valid", valid), ("test", test)]:
        cold_u = ~part["UserID"].isin(u2i)
        cold_i = ~part["ProductID"].isin(p2i)
        diag[name] = {
            "events": int(len(part)),
            "events_cold_user": int(cold_u.sum()), "distinct_cold_users": int(part.loc[cold_u, "UserID"].nunique()),
            "events_unseen_item": int(cold_i.sum()), "distinct_unseen_items": int(part.loc[cold_i, "ProductID"].nunique()),
        }
        d = diag[name]
        print(f"{name:5s}: {d['events']} events | cold-user events {d['events_cold_user']} "
              f"({d['distinct_cold_users']} users) | unseen-item events {d['events_unseen_item']} "
              f"({d['distinct_unseen_items']} items)")
    print("Cold user  -> no training row; fallback = global popularity list over the eligible catalog.")
    print("Cold item  -> outside the 70-item catalog; cannot be recommended or scored (excluded, reported).")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    per_user.value_counts().sort_index().plot.bar(ax=axes[0], color=sns.color_palette("deep")[0])
    axes[0].set(title="Training interactions per user", xlabel="Interactions", ylabel="Users")
    per_item.value_counts().sort_index().plot.bar(ax=axes[1], color=sns.color_palette("deep")[1])
    axes[1].set(title="Training interactions per item", xlabel="Interactions", ylabel="Items")
    for a in axes:
        a.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "interaction_counts.png"), dpi=150)
    plt.close(fig)

    pd.DataFrame({"UserID": users, "index": range(len(users))}).to_csv(os.path.join(EVIDENCE_DIR, "user_mapping.csv"), index=False)
    pd.DataFrame({"ProductID": items, "index": range(len(items))}).to_csv(os.path.join(EVIDENCE_DIR, "item_mapping.csv"), index=False)
    return users, items, density, diag


def problem3(train, valid, test, items, k):
    banner("PROBLEM 3.1  POPULARITY BASELINE")
    ranking = rc.popularity_ranking(train)
    ranking.to_csv(os.path.join(EVIDENCE_DIR, "popularity_ranking.csv"), index=False)
    print("Top 10 of the training popularity ranking (count desc, ProductID asc for ties):")
    print(ranking.head(10).to_string(index=False))
    recommend = rc.make_popularity_recommender(train, k)
    consumed = train.groupby("UserID")["ProductID"].apply(sorted)
    warm = train["UserID"].value_counts().index[0]
    cold = sorted(set(test["UserID"]) - set(train["UserID"]))[0]
    print(f"\nWarm user {warm}: consumed in training {consumed[warm]} -> excluded")
    print(f"  Top-{k}: {recommend(warm)}")
    print(f"Cold user {cold}: no training history -> same eligible popularity list")
    print(f"  Top-{k}: {recommend(cold)}")

    banner("PROBLEM 3.2  METRICS: WORKED EXAMPLE AND BASELINE SCORES")
    ex_recs, ex_rel = ["E1", "E2", "E3", "E4", "E5"], {"E1", "E3", "E4", "E6"}
    ex = {"precision@5": rc.precision_at_k(ex_recs, ex_rel), "recall@5": rc.recall_at_k(ex_recs, ex_rel),
          "dcg@5": rc.dcg_at_k(ex_recs, ex_rel), "ndcg@5": rc.ndcg_at_k(ex_recs, ex_rel)}
    print("Worked example: " + ", ".join(f"{m} = {v:.6f}" for m, v in ex.items()))
    assert abs(ex["precision@5"] - 0.6) < 1e-9 and abs(ex["recall@5"] - 0.75) < 1e-9
    assert abs(ex["dcg@5"] - 1.930677) < 1e-6 and abs(ex["ndcg@5"] - 0.753698) < 1e-6

    catalog = set(items)
    results = {}
    for name, part in [("valid", valid), ("test", test)]:
        per_user, summary, excluded = rc.evaluate(recommend, train, part, catalog, k)
        per_user.to_csv(os.path.join(EVIDENCE_DIR, f"popularity_per_user_{name}.csv"), index=False)
        results[name] = summary
        print(f"\n[{name}] users evaluated = {summary['n_users_evaluated']} "
              f"(cold {summary['n_cold_users_evaluated']}), holdout users excluded = {summary['n_holdout_users_excluded']}")
        print(f"  Precision@{k} = {summary[f'precision@{k}']:.4f} | Recall@{k} = {summary[f'recall@{k}']:.4f} "
              f"| NDCG@{k} = {summary[f'ndcg@{k}']:.4f}")
        if name == "test":
            print(per_user[["UserID", "cold_user", "n_relevant", "hits", "precision", "recall", "ndcg"]].to_string(index=False))

    banner("PROBLEM 3.3  WHAT THE RESTRICTED METRIC CANNOT SAY")
    unseen = int((~test["ProductID"].isin(catalog)).sum())
    print(f"Test events: {len(test)}; involving items unseen in training: {unseen}")
    print(f"Evaluable test users: {results['test']['n_users_evaluated']} "
          f"(including {results['test']['n_cold_users_evaluated']} cold users)")
    assert unseen == 17 and results["test"]["n_users_evaluated"] == 9 and results["test"]["n_cold_users_evaluated"] == 2
    print("Scores describe only positives on the 70-item shelf. The 17 unseen-item events, low ratings")
    print("and users without an eligible positive are outside the denominator, so the number is not")
    print("a statement about all 41 test events or about live click-through.")
    return ex, results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--k", type=int, default=rc.K, help="List length K (contract default 5)")
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    df = rc.load_events()
    train, valid, test, split = problem1(df)
    split.to_csv(os.path.join(EVIDENCE_DIR, "split_summary.csv"), index=False)
    users, items, density, diag = problem2(train, valid, test)
    ex, results = problem3(train, valid, test, items, args.k)

    out = {
        "source_sha256": rc.sha256_of(rc.EVENTS_CSV),
        "cutoffs_utc": {"train_end": rc.TRAIN_END.isoformat(), "valid_end": rc.VALID_END.isoformat()},
        "k": args.k, "relevance_rule": f"Rating >= {rc.RELEVANT_MIN_RATING}, eligible catalog, unseen in train",
        "split_events": dict(zip(split["partition"], split["events"].astype(int))),
        "train_users": len(users), "train_items": len(items), "density": density,
        "cold_start": diag, "worked_example": ex, "popularity": results,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    with open(os.path.join(EVIDENCE_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=int)
    banner("OUTPUTS")
    print(f"Evidence: {os.path.relpath(EVIDENCE_DIR, HERE)}/ | Figures: {os.path.relpath(FIG_DIR, HERE)}/")


if __name__ == "__main__":
    main()
