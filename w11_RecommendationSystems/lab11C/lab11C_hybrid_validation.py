#!/usr/bin/env python3
"""
Lab 11C - Hybrid Recommendation and Validation (Course 961731, Week 11).

Business context: the CMO wants to know whether a matrix-factorization model,
blended with the Lab 11B content signal, is worth an online A/B test. The
analytics team must (1) fit and verify a biased factor model on training data
only, (2) blend it with content on a declared common scale and choose the blend
weight on validation, (3) open the untouched test once, audit every fitted
object for leakage, quantify uncertainty with a user-level bootstrap, and
(4) pre-register a randomized experiment with privacy and exposure logging.

Dataset: data/interaction_events.csv, data/synthetic_teaching_catalog.csv.
Tasks: 1.1-1.3 factor model, 2.1-2.3 hybrid, 3.1-3.3 audit/uncertainty/A-B design.
"""

import argparse
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lab11B"))
import reco_contract as rc  # noqa: E402
import lab11B_content_cf as cb  # noqa: E402

plt.style.use("default")
sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.width", 120)

FIG_DIR = os.path.join(HERE, "figures")
EVIDENCE_DIR = os.path.join(HERE, "evidence")

# Factor-model search space (declared before any validation score is read).
DIMS = [2, 4, 8]
LAMBDAS = [0.01, 0.1, 1.0]
LEARNING_RATE = 0.05
EPOCHS = 100
INIT_SCALE = 0.1
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # hybrid = alpha * content + (1 - alpha) * factor
N_BOOTSTRAP = 2000

# One-event update example from the lab brief.
EX = {"mu": 3.0, "bu": 0.2, "bi": -0.1, "p": [0.2, 0.4], "q": [0.5, 0.1], "r": 4.0, "lr": 0.05, "lam": 0.1}


def banner(title):
    """Print a section banner for the console report."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------------------------------------------ factor model
def predict_mf(state, u, i):
    """Return mu + b_u + b_i + p_u . q_i for known indices u, i."""
    return state["mu"] + state["bu"][u] + state["bi"][i] + float(state["P"][u] @ state["Q"][i])


def sgd_step(state, u, i, r, lr, lam):
    """Apply one update on the one-event loss, computing every gradient from the old state; return residual."""
    e = r - predict_mf(state, u, i)
    bu, bi = state["bu"][u], state["bi"][i]
    pu, qi = state["P"][u].copy(), state["Q"][i].copy()
    state["bu"][u] = bu + lr * (e - lam * bu)
    state["bi"][i] = bi + lr * (e - lam * bi)
    state["P"][u] = pu + lr * (e * qi - lam * pu)
    state["Q"][i] = qi + lr * (e * pu - lam * qi)
    return e


def objective(state, rows, cols, vals, lam):
    """Return 0.5 * mean squared training error + lam/2 * sum of squared factors and biases."""
    pred = state["mu"] + state["bu"][rows] + state["bi"][cols] + np.einsum("ij,ij->i", state["P"][rows], state["Q"][cols])
    reg = (state["P"] ** 2).sum() + (state["Q"] ** 2).sum() + (state["bu"] ** 2).sum() + (state["bi"] ** 2).sum()
    return 0.5 * float(np.mean((vals - pred) ** 2)) + 0.5 * lam * float(reg)


def fit_mf(rows, cols, vals, n_users, n_items, d, lam, seed=rc.RANDOM_SEED):
    """Fit the biased factor model by SGD on training coordinates; return state and per-epoch objective."""
    rng = np.random.default_rng(seed)
    state = {
        "mu": float(vals.mean()), "bu": np.zeros(n_users), "bi": np.zeros(n_items),
        "P": rng.normal(0, INIT_SCALE, (n_users, d)), "Q": rng.normal(0, INIT_SCALE, (n_items, d)),
    }
    history = []
    for _ in range(EPOCHS):
        for k in rng.permutation(len(vals)):
            sgd_step(state, rows[k], cols[k], vals[k], LEARNING_RATE, lam)
        history.append(objective(state, rows, cols, vals, lam))
    return state, history


def eligible_ratings(holdout, u2i, p2i):
    """Return held-out events whose user and item are both known in training (rating-error population)."""
    return holdout[holdout["UserID"].isin(u2i) & holdout["ProductID"].isin(p2i)]


def rating_errors(pred_fn, events):
    """Return (MAE, RMSE, n) of pred_fn(user, item) on the given events."""
    err = np.array([pred_fn(u, p) - r for u, p, r in zip(events["UserID"], events["ProductID"], events["Rating"])])
    if len(err) == 0:
        return float("nan"), float("nan"), 0
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean())), int(len(err))


# ------------------------------------------------------------- recommenders
def minmax(scores):
    """Scale a {item: score} dict to [0, 1] within one request; a constant component becomes all zeros."""
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Models:
    """Frozen training-time objects and the six recommenders built from them."""

    def __init__(self, train, item_vectors, state, u2i, p2i, k):
        """Store frozen objects; nothing here is refitted on validation or test."""
        self.train, self.vectors, self.state, self.u2i, self.p2i, self.k = train, item_vectors, state, u2i, p2i, k
        self.R = cb.ratings_dict(train)
        self.popularity = rc.popularity_ranking(train)["ProductID"].tolist()
        self.pop_rank = {p: r for r, p in enumerate(self.popularity)}
        self.catalog = sorted(p2i)
        self.route = {}

    def candidates(self, user):
        """Return eligible catalog items the user has not consumed in training."""
        seen = set(self.R.get(user, {}))
        return [p for p in self.catalog if p not in seen]

    def content_component(self, user, cands):
        """Return raw cosine scores, or None when the user has no positive profile."""
        profile, _ = cb.content_profile(self.R.get(user, {}), self.vectors)
        if profile is None:
            return None
        return cb.content_scores(profile, self.vectors.loc[cands]).to_dict()

    def factor_component(self, user, cands):
        """Return predicted ratings for known users, or None for a user without trained factors."""
        if user not in self.u2i:
            return None
        u = self.u2i[user]
        return {p: predict_mf(self.state, u, self.p2i[p]) for p in cands}

    def hybrid_scores(self, user, alpha):
        """Return blended [0,1] scores and their components; route is recorded in self.route."""
        cands = self.candidates(user)
        f = self.factor_component(user, cands)
        if f is None:
            self.route[user] = "popularity(unknown user)"
            return None, None, None
        c = self.content_component(user, cands)
        f_s = minmax(f)
        c_s = minmax(c) if c is not None else {p: 0.0 for p in cands}
        self.route[user] = "hybrid" if c is not None else "hybrid(content=0, no positive profile)"
        blend = {p: alpha * c_s[p] + (1 - alpha) * f_s[p] for p in cands}
        return blend, c_s, f_s

    def recommender(self, name, alpha=None):
        """Return recommend(user) -> top-k list for one of the six methods."""
        k = self.k
        if name == "popularity":
            return rc.make_popularity_recommender(self.train, k)
        if name == "content":
            return cb.make_content_recommender(self.train, self.vectors, self.popularity, k)
        if name in ("user_cf", "item_cf"):
            return cb.make_cf_recommender(self.train, self.catalog, self.popularity, k, kind=name.split("_")[0])

        def recommend(user):
            """Return the top-k factor or hybrid list for one user."""
            cands = self.candidates(user)
            if name == "factor":
                scores = self.factor_component(user, cands)
            else:
                scores, _, _ = self.hybrid_scores(user, alpha)
            if scores is None:
                return [p for p in self.popularity if p in set(cands)][:k]
            recs = cb.rank_with_tiebreak(scores, self.pop_rank, k)
            assert len(recs) == len(set(recs)) <= k and set(recs) <= set(cands)
            return recs

        return recommend


def intra_list_diversity(lists, vectors):
    """Return mean (1 - cosine) over distinct item pairs per list, averaged over lists with >= 2 items."""
    vals = []
    for recs in lists:
        if len(recs) < 2:
            continue
        v = vectors.loc[recs].to_numpy()
        sims = v @ v.T
        iu = np.triu_indices(len(recs), 1)
        vals.append(float(np.mean(1 - sims[iu])))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


# ---------------------------------------------------------------- problems
def problem1(train, valid):
    """Specify, fit and verify the factor model; returns the chosen state, mappings and validation record."""
    banner("PROBLEM 1.1  SPECIFY THE BIASED FACTOR MODEL")
    users, items, u2i, p2i = rc.fit_mappings(train)
    rows, cols, vals, _ = rc.coo_arrays(train, u2i, p2i)
    print("Prediction  r_hat(u,i) = mu + b_u + b_i + p_u . q_i")
    print("Objective   J = 0.5 * mean_{(u,i) in E_train} (r_ui - r_hat_ui)^2")
    print("              + lambda/2 * (||P||^2 + ||Q||^2 + ||b_user||^2 + ||b_item||^2)")
    print(f"Shapes      P: U x d = {len(users)} x d, Q: I x d = {len(items)} x d, b_user: {len(users)}, b_item: {len(items)}")
    print("Fitted on training only: mu (training global mean), biases, P, Q, ID mappings, chosen (d, lambda).")
    print(f"Optimizer   SGD over training events, {EPOCHS} epochs, learning rate {LEARNING_RATE}, "
          f"init N(0, {INIT_SCALE}^2), seed {rc.RANDOM_SEED}")

    banner("PROBLEM 1.2  FIT CANDIDATES, SELECT ON VALIDATION RMSE")
    val_events = eligible_ratings(valid, u2i, p2i)
    print(f"Candidates: d in {DIMS}, lambda in {LAMBDAS} ({len(DIMS) * len(LAMBDAS)} fits, all kept)")
    print(f"Validation rating population: {len(val_events)} of {len(valid)} events have a known user AND item")
    record, fits = [], {}
    for d in DIMS:
        for lam in LAMBDAS:
            state, hist = fit_mf(rows, cols, vals, len(users), len(items), d, lam)
            pred = lambda u, p, s=state: predict_mf(s, u2i[u], p2i[p])  # noqa: E731
            mae, rmse, n = rating_errors(pred, val_events)
            record.append({"d": d, "lambda": lam, "train_objective": hist[-1], "val_mae": mae, "val_rmse": rmse, "val_n": n})
            fits[(d, lam)] = (state, hist)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(EVIDENCE_DIR, "factor_validation_record.csv"), index=False)
    print(record.round(4).to_string(index=False))
    best = record.sort_values(["val_rmse", "d", "lambda"], ascending=[True, True, False], kind="mergesort").iloc[0]
    d, lam = int(best["d"]), float(best["lambda"])
    print(f"Selected d = {d}, lambda = {lam} (lowest validation RMSE; ties -> smaller d, larger lambda)")
    state, hist = fits[(d, lam)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, EPOCHS + 1), hist)
    ax.set(title=f"Training objective, d={d}, lambda={lam}", xlabel="Epoch", ylabel="J")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "factor_objective.png"), dpi=150)
    plt.close(fig)

    banner("PROBLEM 1.3  VERIFY STATE, UPDATE AND FALLBACKS")
    u0, i0 = rows[0], cols[0]
    manual = state["mu"] + state["bu"][u0] + state["bi"][i0] + sum(state["P"][u0][j] * state["Q"][i0][j] for j in range(d))
    print(f"Saved-state reconstruction for ({users[u0]}, {items[i0]}): manual {manual:.6f} = predict {predict_mf(state, u0, i0):.6f}")
    assert math.isclose(manual, predict_mf(state, u0, i0), rel_tol=1e-12)
    assert state["P"].shape == (len(users), d) and state["Q"].shape == (len(items), d)
    all_pred = state["mu"] + state["bu"][:, None] + state["bi"][None, :] + state["P"] @ state["Q"].T
    assert np.isfinite(all_pred).all()
    print(f"All {all_pred.size} known-pair predictions finite; range {all_pred.min():.3f} to {all_pred.max():.3f}")

    ex = {"mu": EX["mu"], "bu": np.array([EX["bu"]]), "bi": np.array([EX["bi"]]),
          "P": np.array([EX["p"]]), "Q": np.array([EX["q"]])}
    dot = float(np.dot(EX["p"], EX["q"]))
    pred0 = predict_mf(ex, 0, 0)
    e = sgd_step(ex, 0, 0, EX["r"], EX["lr"], EX["lam"])
    print(f"One-event example: dot = {dot:.2f}, prediction = {pred0:.2f}, residual = {e:.2f}")
    print(f"  b_u {EX['bu']} -> {ex['bu'][0]:.6f}; b_i {EX['bi']} -> {ex['bi'][0]:.6f}")
    print(f"  p_u {EX['p']} -> {ex['P'][0].round(6).tolist()}; q_i {EX['q']} -> {ex['Q'][0].round(6).tolist()}")
    assert math.isclose(dot, 0.14) and math.isclose(pred0, 3.24) and math.isclose(e, 0.76)
    assert np.allclose(ex["P"][0], [0.218, 0.4018]) and np.allclose(ex["Q"][0], [0.5051, 0.1147])
    print("Rating fallback for unknown IDs: unknown user or unknown item -> training global mean mu")
    print("(no random 'trained-looking' factors are created). Ranking: unknown user -> popularity list.")
    return state, users, items, u2i, p2i, record, {"d": d, "lambda": lam}


def evaluate_methods(models, holdout, methods, alpha):
    """Return per-method summaries, per-user frames and lists on one holdout window."""
    out, per_users = [], {}
    for name in methods:
        rec = models.recommender(name, alpha)
        per_user, s, _ = rc.evaluate(rec, models.train, holdout, set(models.catalog), models.k)
        per_users[name] = per_user
        lists = [r.split() for r in per_user["recs"]]
        div, n_div = intra_list_diversity(lists, models.vectors)
        warm = per_user[~per_user["cold_user"]]
        cold = per_user[per_user["cold_user"]]
        out.append({
            "method": name, "users": s["n_users_evaluated"], f"precision@{models.k}": s[f"precision@{models.k}"],
            f"recall@{models.k}": s[f"recall@{models.k}"], f"ndcg@{models.k}": s[f"ndcg@{models.k}"],
            "coverage": len({i for x in lists for i in x}) / len(models.catalog), "diversity": div,
            "diversity_lists": n_div, "warm_users": len(warm), "warm_ndcg": warm["ndcg"].mean() if len(warm) else float("nan"),
            "cold_users": len(cold), "cold_ndcg": cold["ndcg"].mean() if len(cold) else float("nan"),
        })
    return pd.DataFrame(out), per_users


def problem2(train, valid, test, item_vectors, state, u2i, p2i, k):
    """Blend, rank and compare; returns frozen models, selected alpha, tables and per-user test frames."""
    models = Models(train, item_vectors, state, u2i, p2i, k)

    banner("PROBLEM 2.1  CALIBRATE AND BLEND ON VALIDATION")
    print("Calibration: per-request min-max of each component over the user's eligible candidates;")
    print("a constant component becomes all zeros. Labels are never used in scaling.")
    print("Routing: unknown user -> popularity; no positive profile -> content component = 0 (factor drives).")
    grid = []
    for a in ALPHAS:
        rec = models.recommender("hybrid", a)
        per_user, s, _ = rc.evaluate(rec, train, valid, set(models.catalog), k)
        cov = len({i for r in per_user["recs"] for i in r.split()}) / len(models.catalog)
        grid.append({"alpha": a, f"val_ndcg@{k}": s[f"ndcg@{k}"], f"val_recall@{k}": s[f"recall@{k}"], "val_coverage": cov})
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(EVIDENCE_DIR, "hybrid_alpha_record.csv"), index=False)
    print(grid.round(4).to_string(index=False))
    best = grid.sort_values([f"val_ndcg@{k}", "val_coverage", "alpha"], ascending=[False, False, False], kind="mergesort").iloc[0]
    alpha = float(best["alpha"])
    print(f"Selected alpha = {alpha} (highest validation NDCG@{k}; ties -> higher coverage, then larger alpha)")
    print("Lab reference selected alpha = 0.75 with the companion script; differences come from the factor fit.")

    trace_user = cb.pick_warm_user(train)
    blend, c_s, f_s = models.hybrid_scores(trace_user, alpha)
    top = cb.rank_with_tiebreak(blend, models.pop_rank, k)
    i0 = top[0]
    print(f"\nOne combined score, user {trace_user}, item {i0}: {alpha} x content {c_s[i0]:.4f} + "
          f"{1 - alpha} x factor {f_s[i0]:.4f} = {blend[i0]:.4f}")

    banner("PROBLEM 2.2  RANKING TRACE FOR ONE USER")
    cands = models.candidates(trace_user)
    seen = sorted(models.R[trace_user])
    print(f"User {trace_user}: catalog {len(models.catalog)} - seen {seen} = {len(cands)} candidates")
    trace = pd.DataFrame({"content_scaled": c_s, "factor_scaled": f_s, "hybrid": blend})
    trace["pop_rank"] = trace.index.map(models.pop_rank)
    trace = trace.loc[cb.rank_with_tiebreak(blend, models.pop_rank, len(blend))]
    print(trace.head(8).round(4).to_string())
    trace.to_csv(os.path.join(EVIDENCE_DIR, "hybrid_rank_trace.csv"), index_label="ProductID")
    assert len(top) == len(set(top)) <= k and set(top) <= set(cands)
    print(f"Top-{k}: {top} (unique, eligible, length <= {k}); ties -> popularity rank, then ProductID")

    banner("PROBLEM 2.3  UNTOUCHED TEST EVALUATION (SETTINGS FIXED ABOVE)")
    methods = ["popularity", "content", "user_cf", "item_cf", "factor", "hybrid"]
    val_table, _ = evaluate_methods(models, valid, methods, alpha)
    test_table, per_users = evaluate_methods(models, test, methods, alpha)
    val_table.to_csv(os.path.join(EVIDENCE_DIR, "six_methods_valid.csv"), index=False)
    test_table.to_csv(os.path.join(EVIDENCE_DIR, "six_methods_test.csv"), index=False)
    for name, pu in per_users.items():
        pu.to_csv(os.path.join(EVIDENCE_DIR, f"{name}_per_user_test.csv"), index=False)
    cols = ["method", "users", f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "warm_users", "warm_ndcg", "cold_users", "cold_ndcg"]
    print("[test]")
    print(test_table[cols].round(6).to_string(index=False))
    print("[validation, for reference]")
    print(val_table[cols].round(4).to_string(index=False))
    print("Diversity = mean(1 - content cosine) over distinct pairs in lists of >= 2 items (synthetic features).")

    test_events = eligible_ratings(test, u2i, p2i)
    gmean = float(state["mu"])
    means = cb.user_means(models.R)
    err_rows = []
    for name, fn in [
        ("global_mean", lambda u, p: gmean),
        ("user_mean", lambda u, p: means[u]),
        ("factor", lambda u, p: predict_mf(state, u2i[u], p2i[p])),
        ("user_cf", lambda u, p: cb.predict_user_cf(models.R, u, p, means, gmean)[0]),
        ("item_cf", lambda u, p: cb.predict_item_cf(models.R, u, p, means, gmean)[0]),
    ]:
        mae, rmse, n = rating_errors(fn, test_events)
        err_rows.append({"predictor": name, "test_mae": mae, "test_rmse": rmse, "n_ratings": n})
    errors = pd.DataFrame(err_rows)
    errors.to_csv(os.path.join(EVIDENCE_DIR, "test_rating_errors.csv"), index=False)
    print(f"\nRating errors on {len(test_events)} test events with known user and item "
          f"(top-K uses {int(test_table['users'].iloc[0])} users with an eligible positive):")
    print(errors.round(4).to_string(index=False))
    unseen = int((~test["ProductID"].isin(p2i)).sum())
    cold_u = int(test.loc[~test["UserID"].isin(u2i), "UserID"].nunique())
    print(f"Exclusions: {unseen} test events on unseen items (item cold start) | {cold_u} cold test users (user cold start)")
    print("Static evaluation: the model trained before 21 May is not retrained on 13 June.")
    plot_six(test_table, val_table, k)
    return models, alpha, grid, val_table, test_table, errors, per_users


def plot_six(test_table, val_table, k):
    """Save coverage and NDCG by method for validation and test."""
    t = pd.concat([val_table.assign(split="valid"), test_table.assign(split="test")])
    long = t.melt(id_vars=["split", "method"], value_vars=[f"ndcg@{k}", "coverage"], var_name="metric")
    g = sns.catplot(data=long, x="method", y="value", hue="split", col="metric", kind="bar", height=3.6, aspect=1.4)
    g.set_axis_labels("", "Score")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g.figure.suptitle("Lab 11C: six methods, fixed protocol", y=1.03)
    g.savefig(os.path.join(FIG_DIR, "six_methods.png"), dpi=150)
    plt.close(g.figure)


def paired_bootstrap(a, b, n_boot=N_BOOTSTRAP, seed=rc.RANDOM_SEED):
    """Return (mean diff, 2.5%, 97.5%) of per-user a - b, resampling users as whole clusters."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def problem3(df, train, valid, test, models, alpha, mf_choice, record, grid, per_users_test, args):
    """Audit fitted objects, quantify uncertainty and resources, and pre-register an A/B test."""
    banner("PROBLEM 3.1  FITTED-OBJECT TIME AUDIT")
    t_max = train["EventTime"].max()
    audit = pd.DataFrame([
        {"object": o, "fitted_on": "train", "latest_event_used": t_max.isoformat(), "before_boundary": t_max < rc.TRAIN_END}
        for o in ["popularity counts", "user/global means", "ID mappings + catalog", "content vocabulary + IDF",
                  "user profiles", "CF neighbourhoods", f"factors (d={mf_choice['d']}, lambda={mf_choice['lambda']})"]
    ] + [
        {"object": "hyperparameters (d, lambda, alpha)", "fitted_on": "valid", "latest_event_used": valid["EventTime"].max().isoformat(),
         "before_boundary": valid["EventTime"].max() < rc.VALID_END},
    ])
    print(audit.to_string(index=False))
    assert audit["before_boundary"].all()
    future_only = set(valid["ProductID"]).union(test["ProductID"]) - set(train["ProductID"])
    assert not (set(models.catalog) & future_only), "Candidate universe contains future-only items"
    print(f"Candidate universe: {len(models.catalog)} training items; {len(future_only)} future-only items excluded")
    print("Sampling policy: full ranking over all eligible candidates, no negative sampling.")
    leaky = pd.concat([train, valid])
    try:
        assert leaky["EventTime"].max() < rc.TRAIN_END, "Fit uses events on/after the training boundary"
        raise RuntimeError("Leakage assertion should have failed")
    except AssertionError as err:
        print(f"Deliberate copy fitted on train+valid fails as intended -> AssertionError: {err}")
    settings = {"mf": mf_choice, "alpha": alpha, "dims": DIMS, "lambdas": LAMBDAS, "alphas": ALPHAS, "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE, "seed": rc.RANDOM_SEED, "k": rc.K}
    with open(os.path.join(EVIDENCE_DIR, "declared_settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    audit.to_csv(os.path.join(EVIDENCE_DIR, "fit_time_audit.csv"), index=False)

    banner("PROBLEM 3.2  UNCERTAINTY AND RESOURCES")
    k = rc.K
    dist = pd.DataFrame({m: pu.set_index("UserID")["ndcg"] for m, pu in per_users_test.items()})
    print(f"Per-user test NDCG@{k} (rows = users):")
    print(dist.round(4).to_string())
    boots = []
    for m in ["hybrid", "factor", "content"]:
        mean, lo, hi = paired_bootstrap(dist[m], dist["popularity"])
        boots.append({"comparison": f"{m} - popularity", "split": "test", "users": len(dist), "mean_diff": mean, "ci_low": lo, "ci_high": hi})
    _, val_users = evaluate_methods(models, valid, ["popularity", "hybrid", "factor", "content"], alpha)
    vdist = pd.DataFrame({m: pu.set_index("UserID")["ndcg"] for m, pu in val_users.items()})
    for m in ["hybrid", "factor", "content"]:
        mean, lo, hi = paired_bootstrap(vdist[m], vdist["popularity"])
        boots.append({"comparison": f"{m} - popularity", "split": "valid", "users": len(vdist), "mean_diff": mean, "ci_low": lo, "ci_high": hi})
    boots = pd.DataFrame(boots)
    boots.to_csv(os.path.join(EVIDENCE_DIR, "paired_bootstrap.csv"), index=False)
    print(f"\nPaired user bootstrap ({N_BOOTSTRAP} resamples, seed {rc.RANDOM_SEED}) of NDCG@{k} differences:")
    print(boots.round(4).to_string(index=False))
    print(f"With {len(dist)} test users percentile intervals are unstable; positions and repeated ratings are not")
    print("treated as independent samples (the user is the resampling unit).")

    tracemalloc.start()
    t0 = time.perf_counter()
    users, items, u2i, p2i = rc.fit_mappings(train)
    rows, cols, vals, _ = rc.coo_arrays(train, u2i, p2i)
    fit_mf(rows, cols, vals, len(users), len(items), mf_choice["d"], mf_choice["lambda"])
    rec = models.recommender("hybrid", alpha)
    for u in sorted(set(test["UserID"])):
        rec(u)
    wall = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    resources = {"fit_plus_score_seconds": wall, "peak_traced_mib": peak / 2**20, "python": platform.python_version(),
                 "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                 "train_events": len(train), "users_scored": test["UserID"].nunique(), "catalog_items": len(models.catalog)}
    print(f"\nResources: refit chosen factor model + score {resources['users_scored']} test users in {wall:.3f} s, "
          f"peak traced memory {resources['peak_traced_mib']:.2f} MiB")
    print(f"Environment: Python {resources['python']}, {resources['platform']}, input {len(train)} training events")

    banner("PROBLEM 3.3  PRE-REGISTERED A/B TEST")
    design = ab_design(args.baseline_rate, args.mde_relative, args.alpha_level, args.power, args.daily_users)
    for key, val in design.items():
        print(f"{key:28s}: {val}")
    return audit, boots, resources, design


def ab_design(p0, mde_rel, alpha_level, power, daily_users):
    """Return a two-proportion sample-size design for a user-level randomized test."""
    p1 = p0 * (1 + mde_rel)
    z_a = NormalDist().inv_cdf(1 - alpha_level / 2)
    z_b = NormalDist().inv_cdf(power)
    n = math.ceil((z_a + z_b) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2)
    days = max(14, math.ceil(2 * n / daily_users / 7) * 7)
    return {
        "unit": "user (sticky hash of pseudonymous UserID), 50/50",
        "control / treatment": "popularity slot / hybrid slot",
        "primary outcome": "purchase conversion of recommended items per exposed user",
        "guardrails": "revenue per user, page latency p95, complaint/return rate, catalog coverage",
        "assumed baseline rate": f"{p0:.2%} (assumption, replace with logged rate)",
        "minimum detectable effect": f"{mde_rel:.0%} relative -> {p1:.2%}",
        "alpha / power": f"{alpha_level} two-sided / {power}",
        "users per arm": f"{n:,}",
        "duration": f"{days} days at {daily_users:,} eligible users/day (whole weeks, >= 2 weekly cycles)",
        "stopping rule": "fixed horizon; analyse once at the planned end, no peeking",
    }


def main():
    """Run Lab 11C end to end and write evidence/results.json."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--k", type=int, default=rc.K, help="List length K (contract default 5)")
    parser.add_argument("--baseline_rate", type=float, default=0.05, help="Assumed control conversion rate")
    parser.add_argument("--mde_relative", type=float, default=0.10, help="Relative minimum detectable effect")
    parser.add_argument("--alpha_level", type=float, default=0.05, help="Two-sided significance level")
    parser.add_argument("--power", type=float, default=0.80, help="Statistical power")
    parser.add_argument("--daily_users", type=int, default=5000, help="Assumed eligible users per day")
    args = parser.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    df = rc.load_events()
    rc.validate_events(df)
    train, valid, test = rc.temporal_split(df)
    catalog = cb.load_catalog()
    item_vectors, *_ = cb.classroom_tfidf(catalog["Description"])

    state, users, items, u2i, p2i, record, mf_choice = problem1(train, valid)
    models, alpha, grid, val_table, test_table, errors, per_users = problem2(
        train, valid, test, item_vectors, state, u2i, p2i, args.k)
    audit, boots, resources, design = problem3(df, train, valid, test, models, alpha, mf_choice, record, grid, per_users, args)

    out = {
        "source_sha256": rc.sha256_of(rc.EVENTS_CSV), "factor_choice": mf_choice, "alpha": alpha,
        "factor_validation_record": record.to_dict(orient="records"), "alpha_record": grid.to_dict(orient="records"),
        "test_six_methods": test_table.to_dict(orient="records"), "valid_six_methods": val_table.to_dict(orient="records"),
        "test_rating_errors": errors.to_dict(orient="records"), "paired_bootstrap": boots.to_dict(orient="records"),
        "resources": resources, "ab_design": design,
    }
    with open(os.path.join(EVIDENCE_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    banner("OUTPUTS")
    print(f"Evidence: {os.path.relpath(EVIDENCE_DIR, HERE)}/ | Figures: {os.path.relpath(FIG_DIR, HERE)}/")


if __name__ == "__main__":
    main()
