#!/usr/bin/env python3
"""
Lab 11B - Content and Collaborative Recommendation (Course 961731, Week 11).

Business context: the CMO asks whether "similar description" (content-based) or
"similar customers / similar products" (collaborative filtering) should drive
the shop's recommendation slot. Each mechanism is calculated by hand on a tiny
traceable example, then all four methods are compared on the real rating log
under the unchanged Lab 11A contract (same split, 70-item shelf, K = 5,
relevance = rating >= 4), with every fallback recorded.

Datasets: data/tiny_content_catalog.csv, data/tiny_ratings.csv (mechanics),
data/synthetic_teaching_catalog.csv + data/interaction_events.csv (comparison).
Tasks: 1.1-1.3 content-based, 2.1-2.3 neighbourhood CF, 3.1-3.3 comparison.
"""

import argparse
import json
import math
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
TINY_CATALOG = os.path.join(rc.DATA_DIR, "tiny_content_catalog.csv")
TINY_RATINGS = os.path.join(rc.DATA_DIR, "tiny_ratings.csv")
pd.set_option("display.width", 120)

MIN_OVERLAP = 3
SHRINK = 2  # weight multiplier n / (n + SHRINK)
USER_NEIGHBORS = 2
NEUTRAL_RATING = 3.0  # profile weight = max(rating - NEUTRAL_RATING, 0)


# ---------------------------------------------------------------- content-based
def classroom_tfidf(descriptions):
    """TF = log10(1+count), IDF = log10(N/df), then L2-normalise each item row.

    descriptions: pd.Series indexed by item id (NaN allowed = missing metadata).
    Returns (unit-length tfidf DataFrame, counts, df, idf, norms). Rows whose
    vector is all zero (missing text, or only terms present in every document)
    stay zero instead of dividing by zero.
    """
    tokens = descriptions.fillna("").str.split()
    vocab = sorted({t for toks in tokens for t in toks})
    counts = pd.DataFrame(0, index=descriptions.index, columns=vocab, dtype=float)
    for item, toks in tokens.items():
        for t in toks:
            counts.loc[item, t] += 1
    n_docs = len(descriptions)
    doc_freq = (counts > 0).sum(axis=0)
    idf = np.log10(n_docs / doc_freq)
    tf = np.log10(1 + counts)
    raw = tf * idf
    norms = np.sqrt((raw ** 2).sum(axis=1))
    unit = raw.div(norms.where(norms > 0, 1.0), axis=0)
    return unit, counts, doc_freq, idf, norms


def content_profile(user_ratings, item_vectors, neutral=NEUTRAL_RATING):
    """Weighted sum of item vectors with weight max(rating - 3, 0); unit length.

    Returns (profile vector or None, weights). None means no positive history or
    a zero-length profile: the caller must use its declared fallback.
    """
    weights = {i: max(r - neutral, 0.0) for i, r in user_ratings.items() if i in item_vectors.index}
    weights = {i: w for i, w in weights.items() if w > 0}
    if not weights:
        return None, weights
    raw = sum(w * item_vectors.loc[i].to_numpy() for i, w in weights.items())
    norm = float(np.sqrt((raw ** 2).sum()))
    if norm == 0:
        return None, weights
    return raw / norm, weights


def content_scores(profile, item_vectors, exclude=()):
    """Return cosine scores (dot products of unit vectors) for items not in exclude."""
    scores = item_vectors.to_numpy() @ profile
    s = pd.Series(scores, index=item_vectors.index)
    return s.drop(labels=[i for i in exclude if i in s.index])


# ------------------------------------------------------------ neighbourhood CF
def ratings_dict(df, user_col="UserID", item_col="ProductID", rating_col="Rating"):
    """Return {user: {item: rating}} built from a long ratings frame."""
    out = {}
    for u, i, r in zip(df[user_col], df[item_col], df[rating_col]):
        out.setdefault(u, {})[i] = float(r)
    return out


def user_means(R):
    """Return each user's mean over their full observed training row."""
    return {u: float(np.mean(list(row.values()))) for u, row in R.items()}


def global_mean(R):
    """Return the mean of all observed training ratings (new-user fallback)."""
    vals = [r for row in R.values() for r in row.values()]
    return float(np.mean(vals))


def pearson_pair(a, b):
    """Pearson over co-rated items, centred on the pair's own co-rated means.
    Returns (r or None if undefined, overlap n)."""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return None, 0
    x = np.array([a[i] for i in common])
    y = np.array([b[i] for i in common])
    dx, dy = x - x.mean(), y - y.mean()
    den = math.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if den == 0:
        return None, n  # constant ratings on the overlap: correlation undefined
    return float((dx * dy).sum() / den), n


def adjusted_cosine(R, means, i, j):
    """Cosine of item columns after subtracting each user's full training mean."""
    users = sorted(u for u, row in R.items() if i in row and j in row)
    n = len(users)
    if n == 0:
        return None, 0
    x = np.array([R[u][i] - means[u] for u in users])
    y = np.array([R[u][j] - means[u] for u in users])
    den = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    if den == 0:
        return None, n
    return float((x * y).sum() / den), n


def shrink(sim, n, k=SHRINK):
    """Return the similarity scaled by overlap evidence n / (n + k)."""
    return sim * n / (n + k)


def predict_user_cf(R, u, i, means=None, gmean=None, min_overlap=MIN_OVERLAP, k=USER_NEIGHBORS, trace=False):
    """User-based prediction. Returns (prediction, source, details)."""
    means = means or user_means(R)
    gmean = global_mean(R) if gmean is None else gmean
    if u not in R:
        return gmean, "global_mean(new user)", []
    cands = []
    for v in sorted(R):
        if v == u or i not in R[v]:
            continue
        r, n = pearson_pair(R[u], R[v])
        row = {"neighbor": v, "overlap": n, "pearson": r, "weight": None, "used": False}
        if r is not None and n >= min_overlap:
            row["weight"] = shrink(r, n)
        cands.append(row)
    pos = sorted([c for c in cands if c["weight"] is not None and c["weight"] > 0],
                 key=lambda c: (-c["weight"], c["neighbor"]))[:k]
    if not pos:
        return means[u], "user_mean(no positive neighbor)", cands if trace else []
    for c in pos:
        c["used"] = True
    num = sum(c["weight"] * (R[c["neighbor"]][i] - means[c["neighbor"]]) for c in pos)
    den = sum(c["weight"] for c in pos)
    return means[u] + num / den, "user_cf", cands if trace else []


def predict_item_cf(R, u, i, means=None, gmean=None, min_overlap=MIN_OVERLAP, trace=False):
    """Item-based prediction with adjusted cosine. Returns (prediction, source, details)."""
    means = means or user_means(R)
    gmean = global_mean(R) if gmean is None else gmean
    if u not in R:
        return gmean, "global_mean(new user)", []
    rows = []
    for j in sorted(R[u]):
        if j == i:
            continue
        s, n = adjusted_cosine(R, means, i, j)
        row = {"neighbor_item": j, "support": n, "adj_cosine": s, "weight": None, "used": False}
        if s is not None and n >= min_overlap:
            row["weight"] = shrink(s, n)
            row["used"] = row["weight"] > 0
        rows.append(row)
    pos = [r for r in rows if r["used"]]
    if not pos:
        return means[u], "user_mean(no positive item neighbor)", rows if trace else []
    num = sum(r["weight"] * (R[u][r["neighbor_item"]] - means[u]) for r in pos)
    den = sum(r["weight"] for r in pos)
    return means[u] + num / den, "item_cf", rows if trace else []


# ------------------------------------------------------- top-K recommenders
def rank_with_tiebreak(scores, pop_rank, k):
    """Sort by score desc, then popularity rank, then ProductID (deterministic)."""
    order = sorted(scores, key=lambda i: (-scores[i], pop_rank.get(i, len(pop_rank)), i))
    return order[:k]


def make_content_recommender(train, item_vectors, popularity, k):
    """Return a top-k content recommender; recommend.log records content or popularity fallback per user."""
    R = ratings_dict(train)
    pop_rank = {p: r for r, p in enumerate(popularity)}
    log = {}

    def recommend(user):
        """Return the top-k list for one user."""
        seen = set(R.get(user, {}))
        profile, _ = content_profile(R.get(user, {}), item_vectors)
        if profile is None:
            log[user] = "popularity(no positive profile)"
            return [p for p in popularity if p not in seen][:k]
        s = content_scores(profile, item_vectors, exclude=seen)
        log[user] = "content"
        return rank_with_tiebreak(s.to_dict(), pop_rank, k)

    recommend.log = log
    return recommend


def make_cf_recommender(train, catalog, popularity, k, kind="user"):
    """Return a top-k user- or item-CF recommender ranking predicted ratings; recommend.log records each slot's score source."""
    R = ratings_dict(train)
    means, gmean = user_means(R), global_mean(R)
    pop_rank = {p: r for r, p in enumerate(popularity)}
    predict = predict_user_cf if kind == "user" else predict_item_cf
    log = {}

    def recommend(user):
        """Return the top-k list for one user."""
        seen = set(R.get(user, {}))
        scores, sources = {}, {}
        for item in catalog:
            if item in seen:
                continue
            scores[item], sources[item], _ = predict(R, user, item, means, gmean)
        recs = rank_with_tiebreak(scores, pop_rank, k)
        log[user] = {i: sources[i] for i in recs}
        return recs

    recommend.log = log
    return recommend


def banner(title):
    """Print a section banner for the console report."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def close(a, b, tol=1e-6):
    """Return True when two numbers agree to the lab's six-decimal precision."""
    return abs(a - b) < tol


def load_tiny():
    """Return the tiny catalog, wide rating matrix and its ratings dict."""
    cat = pd.read_csv(TINY_CATALOG, dtype=str).set_index("ItemID")
    wide = pd.read_csv(TINY_RATINGS, dtype={"UserID": str}).set_index("UserID")
    R = {u: {i: float(v) for i, v in row.dropna().items()} for u, row in wide.iterrows()}
    return cat, wide, R


def load_catalog():
    """Return the synthetic teaching catalog indexed by ProductID."""
    cat = pd.read_csv(rc.CATALOG_CSV, dtype=str, keep_default_na=False).set_index("ProductID")
    assert cat.index.is_unique, "Catalog ProductID must be unique"
    return cat


def problem1(tiny_cat, catalog, train):
    """Run Problem 1 (TF-IDF, profile, recommendation); returns vectors, popularity order and warm user."""
    banner("PROBLEM 1.1  CLASSROOM TF-IDF REPRESENTATION")
    print(f"Synthetic catalog: {len(catalog)} items; provenance labels = {sorted(catalog['MetadataStatus'].unique())}")
    print("Descriptions are teaching data, not real product text.")
    unit, counts, dfreq, idf, norms = classroom_tfidf(tiny_cat["Description"])
    print("\nTiny catalog term counts:")
    print(counts.astype(int).to_string())
    print("\nDocument frequency and IDF = log10(N/df), N = 4:")
    print(pd.DataFrame({"df": dfreq, "idf": idf.round(6)}).to_string())
    tf_a = math.log10(1 + 2)
    print(f"\nCount-two cell (A, coffee): TF = log10(1+2) = {tf_a:.6f}; TF-IDF = {tf_a:.6f} x {idf['coffee']:.6f} "
          f"= {tf_a * idf['coffee']:.6f}")
    print(f"Norm of A = {norms['A']:.6f}; normalised A = {unit.loc['A'].round(6).tolist()}")
    assert close(idf["coffee"], 0.124939) and close(idf["tea"], 0.301030)
    assert close(tf_a * idf["coffee"], 0.059611) and close(norms["A"], 0.108468)
    assert np.allclose(unit.loc["A"].to_numpy(), [0.549572, 0, 0.835446, 0], atol=1e-6)
    sk_tf, sk_idf = 2, math.log((1 + 4) / (1 + 3)) + 1
    print(f"Library default (scikit-learn TfidfVectorizer): raw TF {sk_tf}, smoothed IDF ln(5/4)+1 = {sk_idf:.6f}, "
          f"cell before L2 = {sk_tf * sk_idf:.6f} -> different numbers, so the convention must be declared.")

    real_unit, real_counts, real_df, real_idf, real_norms = classroom_tfidf(catalog["Description"])
    print(f"\nReal synthetic catalog: {real_unit.shape[0]} items x {real_unit.shape[1]} terms")
    print(pd.DataFrame({"df": real_df, "idf": real_idf.round(6)}).sort_values("df", ascending=False).to_string())
    print(f"Item norms: min {real_norms.min():.6f}, max {real_norms.max():.6f}, zero vectors {(real_norms == 0).sum()}")
    real_unit.round(6).to_csv(os.path.join(EVIDENCE_DIR, "catalog_tfidf_unit.csv"))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    sns.heatmap(unit, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, cbar=False, ax=ax)
    ax.set_title("Tiny catalog: unit-length classroom TF-IDF")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tiny_tfidf.png"), dpi=150)
    plt.close(fig)

    banner("PROBLEM 1.2  POSITIVE-WEIGHT USER PROFILE")
    hist = {"A": 5.0, "C": 4.0}
    profile, weights = content_profile(hist, unit)
    print(f"History {hist} -> weights max(rating-3,0) = {weights}")
    print(f"Unit profile = {dict(zip(unit.columns, profile.round(6)))}")
    print("Fallbacks: no positive history (all ratings <= 3) or zero-length profile -> popularity list")
    print("over the eligible catalog (the Lab 11A cold-user rule). Low ratings are never turned into")
    print("negative weights here, because a dislike is not the opposite direction of a description.")

    banner("PROBLEM 1.3  COSINE RECOMMENDATION AND EXPLANATION")
    scores = content_scores(profile, unit, exclude=hist)
    print(f"Scores after excluding seen A, C: {scores.round(6).to_dict()}")
    assert close(scores["B"], 0.847392) and close(scores["D"], 0.556954)
    contrib = (unit.loc["B"] * profile).round(6)
    print(f"Why B leads: term contributions profile x B = {contrib[contrib > 0].to_dict()} (shared coffee/roast)")

    R = ratings_dict(train)
    popularity = rc.popularity_ranking(train)["ProductID"].tolist()
    rec = make_content_recommender(train, real_unit, popularity, rc.K)
    warm = pick_warm_user(train)
    p_real, w_real = content_profile(R[warm], real_unit)
    top = rec(warm)
    print(f"\nReal warm user {warm}: training ratings {R[warm]} -> positive weights {w_real}")
    print(f"Top-{rc.K} content list: {top}")
    lead = real_unit.loc[top[0]] * p_real
    print(f"Lead item {top[0]} '{catalog.loc[top[0], 'Description']}' shares terms "
          f"{lead[lead > 0].round(4).to_dict()} with the profile")
    return unit, real_unit, popularity, warm


def pick_warm_user(train):
    """Warm user = most training ratings with at least one positive (>=4); ties by UserID."""
    pos = train[train["Rating"] >= rc.RELEVANT_MIN_RATING]["UserID"].unique()
    counts = train[train["UserID"].isin(pos)].groupby("UserID").size()
    return sorted(counts.index, key=lambda u: (-counts[u], u))[0]


def problem2(wide, R):
    """Run Problem 2 (user/item neighbourhoods and prediction traces); returns the traced predictions."""
    banner("PROBLEM 2.1  USER-USER PEARSON WITH OVERLAP SHRINKAGE")
    print("Rating matrix (NaN = unknown):")
    print(wide.to_string(na_rep="?"))
    mask = wide.notna().astype(int)
    print("\nObservation mask:")
    print(mask.to_string())
    means = user_means(R)
    print(f"\nFull observed row means: { {u: round(m, 2) for u, m in means.items()} }")
    pred_u, src_u, cands = predict_user_cf(R, "U0", "D", trace=True)
    users = pd.DataFrame(cands)
    users["shrunk_weight"] = users["weight"].round(6)
    print("\nCandidates who rated D (target U0, min overlap 3, weight = r x n/(n+2)):")
    print(users[["neighbor", "overlap", "pearson", "shrunk_weight", "used"]].round(6).to_string(index=False))
    users.to_csv(os.path.join(EVIDENCE_DIR, "tiny_user_neighbors.csv"), index=False)
    print("Pair centring: Pearson subtracts each user's mean over the CO-RATED items only (fair comparison).")
    print("Prediction centring: residuals use each neighbour's FULL row mean (their usual rating level).")

    banner("PROBLEM 2.2  ITEM-ITEM ADJUSTED COSINE")
    rows = []
    for j in ["A", "B", "C"]:
        s, n = adjusted_cosine(R, means, "D", j)
        rows.append({"pair": f"D-{j}", "support": n, "adj_cosine": round(s, 6),
                     "shrunk_weight": round(shrink(s, n), 6), "selected": s > 0})
    items = pd.DataFrame(rows)
    print(items.to_string(index=False))
    items.to_csv(os.path.join(EVIDENCE_DIR, "tiny_item_neighbors.csv"), index=False)
    assert np.allclose(items["adj_cosine"], [0.188639, -0.543201, 0.227710], atol=1e-6)
    print("Adjusted cosine compares how USERS rate two items (behaviour, centred per user);")
    print("text cosine compares words in two descriptions. Same formula shape, different evidence.")

    banner("PROBLEM 2.3  VERIFY PREDICTIONS AND A BOUNDARY CASE")
    w3, w1 = 0.6, math.sqrt(3) / 2 * 3 / 5
    hand_u = 4 + (w3 * (4 - 4) + w1 * (5 - 3.75)) / (w3 + w1)
    print(f"User-CF by hand: 4 + [0.6(4-4) + {w1:.6f}(5-3.75)] / (0.6 + {w1:.6f}) = {hand_u:.6f}")
    print(f"User-CF by code: {pred_u:.6f} ({src_u})")
    sA = adjusted_cosine(R, means, "D", "A")[0]
    sC = adjusted_cosine(R, means, "D", "C")[0]
    pred_i, src_i, _ = predict_item_cf(R, "U0", "D")
    hand_i = 4 + (sA * (5 - 4) + sC * (4 - 4)) / (sA + sC)
    print(f"Item-CF by hand: 4 + [{sA:.6f}(5-4) + {sC:.6f}(4-4)] / ({sA:.6f} + {sC:.6f}) = {hand_i:.6f} (unrounded weights)")
    print(f"Item-CF by code: {pred_i:.6f} ({src_i}); the 4/6 shrinkage cancels because both supports are 4")
    assert close(pred_u, 4.580127) and close(hand_u, pred_u) and close(pred_i, 4.453079) and close(hand_i, pred_i)

    R_const = {u: dict(r) for u, r in R.items()}
    R_const["U0"] = {"A": 3.0, "B": 3.0, "C": 3.0}
    pc, sc, _ = predict_user_cf(R_const, "U0", "D")
    print(f"Boundary: U0 rates A,B,C all 3 -> Pearson undefined (zero variance) -> {pc:.4f} via {sc}")
    assert sc.startswith("user_mean") and math.isfinite(pc)
    return {"user_cf_U0_D": pred_u, "item_cf_U0_D": pred_i, "constant_rating_fallback": pc}


def coverage(per_user, catalog_size):
    """Return the share of the catalog that appears in at least one evaluated list."""
    items = {i for recs in per_user["recs"] for i in recs.split()}
    return len(items) / catalog_size


def problem3(train, valid, test, real_unit, popularity, warm, k):
    """Run Problem 3 (four-method comparison and stress tests); returns the table, warm-user lists and stress results."""
    catalog = sorted(set(train["ProductID"]))
    methods = {
        "popularity": rc.make_popularity_recommender(train, k),
        "content": make_content_recommender(train, real_unit, popularity, k),
        "user_cf": make_cf_recommender(train, catalog, popularity, k, kind="user"),
        "item_cf": make_cf_recommender(train, catalog, popularity, k, kind="item"),
    }

    banner("PROBLEM 3.1  SAME WARM USER, FOUR METHODS")
    R = ratings_dict(train)
    print(f"Warm user {warm}; training ratings {R[warm]}")
    lists = {}
    for name, rec in methods.items():
        lists[name] = rec(warm)
        note = ""
        if name == "content":
            note = rec.log[warm]
        elif name in ("user_cf", "item_cf"):
            srcs = pd.Series(rec.log[warm]).value_counts().to_dict()
            note = f"score sources {srcs}"
        else:
            note = "training popularity, seen items removed"
        print(f"{name:10s}: {lists[name]}  [{note}]")
    pd.DataFrame(lists).to_csv(os.path.join(EVIDENCE_DIR, "warm_user_lists.csv"), index_label="rank0")
    rows_by_user = [set(r) for r in R.values()]
    max_overlap = max(len(a & b) for n, a in enumerate(rows_by_user) for b in rows_by_user[n + 1:])
    print("CF lists equal the popularity order here because every candidate fell back to the user's mean:")
    print(f"the largest overlap between any two training users is {max_overlap} items, below MIN_OVERLAP = {MIN_OVERLAP}.")

    banner("PROBLEM 3.2  FAIR EVALUATION UNDER THE LAB 11A CONTRACT")
    rows = []
    for split_name, part in [("valid", valid), ("test", test)]:
        for name, rec in methods.items():
            per_user, s, _ = rc.evaluate(rec, train, part, set(catalog), k)
            per_user.to_csv(os.path.join(EVIDENCE_DIR, f"{name}_per_user_{split_name}.csv"), index=False)
            fb = fallback_share(name, rec, per_user)
            rows.append({"split": split_name, "method": name, "users": s["n_users_evaluated"],
                         "cold_users": s["n_cold_users_evaluated"], f"precision@{k}": s[f"precision@{k}"],
                         f"recall@{k}": s[f"recall@{k}"], f"ndcg@{k}": s[f"ndcg@{k}"],
                         "coverage": coverage(per_user, len(catalog)), "fallback_share": fb})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    table.to_csv(os.path.join(EVIDENCE_DIR, "method_comparison.csv"), index=False)
    print("Coverage = distinct recommended items across evaluated users / 70-item catalog.")
    print("Fallback share = evaluated users (content) or recommended slots (CF) served by a fallback.")
    plot_comparison(table, k)

    banner("PROBLEM 3.3  STRESS TESTS ON SEPARATE COPIES")
    stress = stress_tests()
    for name, result in stress.items():
        print(f"{name:22s}: {result}")
    return table, lists, stress


def fallback_share(name, rec, per_user):
    """Return the share of evaluated users (content) or list slots (CF) served by a fallback."""
    users = per_user["UserID"].tolist()
    if name == "popularity":
        return 1.0
    if name == "content":
        return float(np.mean([not rec.log[u].startswith("content") for u in users]))
    slots = [src for u in users for src in rec.log[u].values()]
    return float(np.mean([not s.endswith("_cf") for s in slots]))


def stress_tests():
    """Run fallback checks on modified copies of the tiny data; returns one finding per case."""
    tiny_cat, wide, R = load_tiny()
    out = {}
    cat = tiny_cat["Description"].copy()
    cat["B"] = np.nan
    unit, *_ = classroom_tfidf(cat)
    prof, _ = content_profile({"A": 5.0, "C": 4.0}, unit)
    s = content_scores(prof, unit, exclude=["A", "C"])
    assert np.isfinite(s).all() and s["B"] == 0
    out["missing metadata (B)"] = f"B vector all zero, score {s['B']:.1f}; D {s['D']:.4f} -> finite, B ranks by tie-break"

    cat2 = tiny_cat["Description"].copy()
    cat2["C"] = "tea tea herbal coffee"  # copy: coffee now in every document, so IDF(coffee) = 0
    cat2["A"] = "coffee"
    unit2, _, _, _, norms2 = classroom_tfidf(cat2)
    prof2, _ = content_profile({"A": 5.0}, unit2)
    assert norms2["A"] == 0 and prof2 is None
    out["zero text vector (A)"] = "A='coffee' with coffee in all 4 docs (IDF 0) -> norm 0, profile None -> popularity fallback"

    Rc = {u: dict(r) for u, r in R.items()}
    Rc["U0"] = {"A": 3.0, "B": 3.0, "C": 3.0}
    p, src, _ = predict_user_cf(Rc, "U0", "D")
    assert math.isfinite(p) and src.startswith("user_mean")
    out["constant ratings (U0)"] = f"Pearson undefined -> {p:.4f} via {src}"

    Rn = {u: dict(r) for u, r in R.items()}
    Rn["U9"] = {"E": 4.0}
    pu, su, _ = predict_user_cf(Rn, "U9", "D")
    pi, si, _ = predict_item_cf(Rn, "U9", "D")
    assert math.isfinite(pu) and math.isfinite(pi)
    out["no overlap (U9)"] = f"user-CF {pu:.4f} via {su}; item-CF {pi:.4f} via {si}"
    out["interpretation"] = "each case is missing data support, not an arithmetic error: no NaN, no division by zero"
    return out


def plot_comparison(table, k):
    """Save the grouped bar chart of metrics by method and split."""
    metrics = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage"]
    long = table.melt(id_vars=["split", "method"], value_vars=metrics, var_name="metric")
    g = sns.catplot(data=long, x="metric", y="value", hue="method", col="split", kind="bar", height=3.6, aspect=1.3)
    g.set_axis_labels("", "Score")
    g.figure.suptitle("Lab 11B: four methods under the Lab 11A contract", y=1.03)
    g.savefig(os.path.join(FIG_DIR, "method_comparison.png"), dpi=150)
    plt.close(g.figure)


def main():
    """Run Lab 11B end to end and write evidence/results.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--k", type=int, default=rc.K, help="List length K (contract default 5)")
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    df = rc.load_events()
    rc.validate_events(df)
    train, valid, test = rc.temporal_split(df)
    tiny_cat, wide, R_tiny = load_tiny()
    catalog = load_catalog()
    assert set(catalog.index) == set(train["ProductID"]), "Synthetic catalog must match the 70-item shelf"

    unit, real_unit, popularity, warm = problem1(tiny_cat, catalog, train)
    traces = problem2(wide, R_tiny)
    table, lists, stress = problem3(train, valid, test, real_unit, popularity, warm, args.k)

    out = {
        "source_sha256": rc.sha256_of(rc.EVENTS_CSV), "k": args.k,
        "min_overlap": MIN_OVERLAP, "shrinkage": f"n/(n+{SHRINK})", "user_neighbors": USER_NEIGHBORS,
        "tiny_traces": traces, "warm_user": warm, "warm_user_lists": lists,
        "comparison": table.to_dict(orient="records"), "stress_tests": stress,
    }
    with open(os.path.join(EVIDENCE_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    banner("OUTPUTS")
    print(f"Evidence: {os.path.relpath(EVIDENCE_DIR, HERE)}/ | Figures: {os.path.relpath(FIG_DIR, HERE)}/")


if __name__ == "__main__":
    main()
