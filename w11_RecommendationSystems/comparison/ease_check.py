#!/usr/bin/env python3
"""
Week 11 comparison - EASE under the Lab 11A contract.

Business context: before recommending a newer method to the CMO, check whether a
strong modern implicit-feedback baseline (EASE, Steck 2019) changes the Lab 11
conclusion on this rating log. EASE has a closed-form solution and one
hyperparameter, so it runs with numpy only and is selected on validation.

Dataset: data/interaction_events.csv. Same split, 70-item shelf, K = 5,
relevance rule and popularity fallback as Labs 11A to 11C.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import reco_contract as rc  # noqa: E402

LAMBDAS = [1.0, 10.0, 100.0, 500.0]
EVIDENCE_DIR = os.path.join(HERE, "evidence")


def banner(title):
    """Print a section banner for the console report."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fit_ease(X, lam):
    """Return the EASE item-item weight matrix B with a zero diagonal (closed form)."""
    G = X.T @ X + lam * np.eye(X.shape[1])
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return B


def make_ease_recommender(train, B, u2i, items, k):
    """Return recommend(user): EASE scores for known users, popularity list for unknown users."""
    X = implicit_matrix(train, u2i, {p: n for n, p in enumerate(items)})
    popularity = rc.make_popularity_recommender(train, k)
    pop_rank = {p: r for r, p in enumerate(rc.popularity_ranking(train)["ProductID"])}

    def recommend(user):
        """Return the top-k list for one user."""
        if user not in u2i:
            return popularity(user)
        row = X[u2i[user]]
        scores = row @ B
        cands = [n for n in range(len(items)) if row[n] == 0]
        cands.sort(key=lambda n: (-scores[n], pop_rank[items[n]], items[n]))
        return [items[n] for n in cands[:k]]

    return recommend


def implicit_matrix(train, u2i, p2i):
    """Return the binary user x item matrix: 1 for every observed training interaction."""
    X = np.zeros((len(u2i), len(p2i)))
    X[train["UserID"].map(u2i), train["ProductID"].map(p2i)] = 1.0
    return X


def main():
    """Select EASE lambda on validation, evaluate once on test, and write evidence/ease_results.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--k", type=int, default=rc.K, help="List length K (contract default 5)")
    args = parser.parse_args()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    df = rc.load_events()
    rc.validate_events(df)
    train, valid, test = rc.temporal_split(df)
    users, items, u2i, p2i = rc.fit_mappings(train)
    X = implicit_matrix(train, u2i, p2i)
    catalog = set(items)

    banner("EASE: SELECT LAMBDA ON VALIDATION")
    record = []
    for lam in LAMBDAS:
        rec = make_ease_recommender(train, fit_ease(X, lam), u2i, items, args.k)
        per_user, s, _ = rc.evaluate(rec, train, valid, catalog, args.k)
        cov = len({i for r in per_user["recs"] for i in r.split()}) / len(items)
        record.append({"lambda": lam, f"val_ndcg@{args.k}": s[f"ndcg@{args.k}"], "val_coverage": cov, "val_users": s["n_users_evaluated"]})
    record = pd.DataFrame(record)
    print(record.round(4).to_string(index=False))
    best = record.sort_values([f"val_ndcg@{args.k}", "lambda"], ascending=[False, False], kind="mergesort").iloc[0]
    lam = float(best["lambda"])
    print(f"Selected lambda = {lam} (highest validation NDCG@{args.k}; ties -> larger lambda)")

    banner("EASE: UNTOUCHED TEST")
    rec = make_ease_recommender(train, fit_ease(X, lam), u2i, items, args.k)
    pop = rc.make_popularity_recommender(train, args.k)
    rows = []
    for name, r in [("popularity", pop), ("ease", rec)]:
        per_user, s, _ = rc.evaluate(r, train, test, catalog, args.k)
        cov = len({i for x in per_user["recs"] for i in x.split()}) / len(items)
        rows.append({"method": name, "users": s["n_users_evaluated"], f"precision@{args.k}": s[f"precision@{args.k}"],
                     f"recall@{args.k}": s[f"recall@{args.k}"], f"ndcg@{args.k}": s[f"ndcg@{args.k}"], "coverage": cov})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    with open(os.path.join(EVIDENCE_DIR, "ease_results.json"), "w") as f:
        json.dump({"lambda_record": record.to_dict(orient="records"), "selected_lambda": lam,
                   "test": table.to_dict(orient="records")}, f, indent=2)


if __name__ == "__main__":
    main()
