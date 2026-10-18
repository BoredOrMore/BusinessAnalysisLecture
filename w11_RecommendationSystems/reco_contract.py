#!/usr/bin/env python3
"""
Week 11 Recommendation Systems: shared data and evaluation contract.

Business context: an online shop (CMO stakeholder) wants personalized product
recommendations. Lab 11A fixes the data contract (schema, validation, temporal
split, ID mappings) and the evaluation contract (eligible catalog, relevance
rule, Precision/Recall/NDCG@K). Labs 11B and 11C import this module unchanged
so every model is scored on the same shelf and the same held-out customers.
"""

import hashlib
import math
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVENTS_CSV = os.path.join(DATA_DIR, "interaction_events.csv")
CATALOG_CSV = os.path.join(DATA_DIR, "synthetic_teaching_catalog.csv")

REQUIRED_COLUMNS = ["UserID", "ProductID", "Rating", "Timestamp"]
RATING_MIN, RATING_MAX = 1, 5
TRAIN_END = pd.Timestamp("2024-05-21", tz="UTC")  # train: t < TRAIN_END
VALID_END = pd.Timestamp("2024-06-13", tz="UTC")  # valid: TRAIN_END <= t < VALID_END; test: t >= VALID_END
K = 5
RELEVANT_MIN_RATING = 4
RANDOM_SEED = 42


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_events(path=EVENTS_CSV):
    """Read IDs as strings and parse numbers strictly (invalid tokens raise)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = raw.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="raise")
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="raise").astype("int64")
    df["EventTime"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    return df


def validate_events(df):
    """Fail loudly on any contract violation; never repair silently."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    for col in ["UserID", "ProductID"]:
        assert df[col].notna().all() and (df[col].str.strip() != "").all(), f"Blank {col} found"
    r = df["Rating"]
    assert r.notna().all() and np.isfinite(r).all(), "Non-finite Rating found"
    assert (r == r.round()).all(), "Non-integer Rating found"
    assert r.between(RATING_MIN, RATING_MAX).all(), (
        f"Rating outside [{RATING_MIN},{RATING_MAX}]: {sorted(r[~r.between(RATING_MIN, RATING_MAX)].unique())}"
    )
    assert df["Timestamp"].notna().all() and (df["Timestamp"] > 0).all(), "Invalid Timestamp found"
    dup = df.duplicated(["UserID", "ProductID"], keep=False)
    assert not dup.any(), f"Duplicated (UserID, ProductID) pairs: {int(dup.sum())} rows"
    return True


def temporal_split(df):
    """Stable sort once, then cut on fixed UTC instants (equal times stay together)."""
    d = df.sort_values(["Timestamp", "UserID", "ProductID"], kind="mergesort").reset_index(drop=True)
    train = d[d["EventTime"] < TRAIN_END].copy()
    valid = d[(d["EventTime"] >= TRAIN_END) & (d["EventTime"] < VALID_END)].copy()
    test = d[d["EventTime"] >= VALID_END].copy()
    assert len(train) + len(valid) + len(test) == len(d), "Split masks are not exhaustive"
    assert train["EventTime"].max() < valid["EventTime"].min() and valid["EventTime"].max() < test["EventTime"].min()
    return train, valid, test


def fit_mappings(train):
    """Sorted, reversible string-to-index maps fitted on training IDs only."""
    users = sorted(train["UserID"].unique())
    items = sorted(train["ProductID"].unique())
    u2i = {u: k for k, u in enumerate(users)}
    p2i = {p: k for k, p in enumerate(items)}
    return users, items, u2i, p2i


def coo_arrays(train, u2i, p2i):
    rows = train["UserID"].map(u2i).to_numpy()
    cols = train["ProductID"].map(p2i).to_numpy()
    vals = train["Rating"].to_numpy(dtype=float)
    observed = np.ones(len(rows), dtype=np.int8)
    return rows, cols, vals, observed


def relevant_holdout(train, holdout, catalog):
    """Positives: Rating >= 4, item in eligible catalog, not consumed in training."""
    seen = set(zip(train["UserID"], train["ProductID"]))
    h = holdout[(holdout["Rating"] >= RELEVANT_MIN_RATING) & holdout["ProductID"].isin(catalog)]
    h = h[[(u, p) not in seen for u, p in zip(h["UserID"], h["ProductID"])]]
    return h.groupby("UserID")["ProductID"].apply(set).to_dict()


def precision_at_k(recs, relevant, k=K):
    return sum(1 for x in recs[:k] if x in relevant) / k  # denominator stays K


def recall_at_k(recs, relevant, k=K):
    return sum(1 for x in recs[:k] if x in relevant) / len(relevant)


def dcg_at_k(recs, relevant, k=K):
    return sum(1.0 / math.log2(rank + 1) for rank, x in enumerate(recs[:k], start=1) if x in relevant)


def ndcg_at_k(recs, relevant, k=K):
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg_at_k(recs, relevant, k) / idcg if idcg > 0 else float("nan")


def evaluate(recommend, train, holdout, catalog, k=K):
    """Macro-average over users with >=1 eligible positive. recommend(user) -> ranked list."""
    rel = relevant_holdout(train, holdout, catalog)
    train_users = set(train["UserID"])
    rows = []
    for user in sorted(rel):
        recs = recommend(user)
        rows.append({
            "UserID": user,
            "cold_user": user not in train_users,
            "n_relevant": len(rel[user]),
            "hits": sum(1 for x in recs[:k] if x in rel[user]),
            "precision": precision_at_k(recs, rel[user], k),
            "recall": recall_at_k(recs, rel[user], k),
            "ndcg": ndcg_at_k(recs, rel[user], k),
            "recs": " ".join(recs[:k]),
        })
    per_user = pd.DataFrame(rows)
    excluded = sorted(set(holdout["UserID"]) - set(rel))
    summary = {
        "n_users_evaluated": int(len(per_user)),
        "n_cold_users_evaluated": int(per_user["cold_user"].sum()) if len(per_user) else 0,
        "n_holdout_users_excluded": len(excluded),
        f"precision@{k}": float(per_user["precision"].mean()),
        f"recall@{k}": float(per_user["recall"].mean()),
        f"ndcg@{k}": float(per_user["ndcg"].mean()),
    }
    return per_user, summary, excluded


def popularity_ranking(train):
    """Descending training count, ties broken by lexicographic ProductID."""
    counts = train.groupby("ProductID").size().rename("count").reset_index()
    counts = counts.sort_values(["count", "ProductID"], ascending=[False, True], kind="mergesort")
    return counts.reset_index(drop=True)


def make_popularity_recommender(train, k=K):
    ranking = popularity_ranking(train)["ProductID"].tolist()
    consumed = train.groupby("UserID")["ProductID"].apply(set).to_dict()

    def recommend(user):
        seen = consumed.get(user, set())  # cold user: empty set, same eligible list
        return [p for p in ranking if p not in seen][:k]

    return recommend
