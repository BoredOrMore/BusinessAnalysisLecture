# Customer Churn Practice Plan

Governing procedure: [Customer_Churn_Practice.pdf](Customer_Churn_Practice.pdf),
credit-card case, pages 2-24. The earlier [Week 08 lecture](<Week 08 - Customer Churn.pdf>)
is conceptual background. The separate e-commerce case has not been implemented.

**Status:** full historical credit-card pipeline implemented and run. See
[PIPELINE_REPORT.md](PIPELINE_REPORT.md) for generated findings and step-specific
problems. [CHURN_REPORT.md](CHURN_REPORT.md) preserves the initial business brief;
[BASELINE_REPORT.md](BASELINE_REPORT.md) contains the earlier integrity study.

## Completed steps

- [x] Pages 2-3: verify source dimensions, label mapping, five example records, and field meanings.
- [x] Page 4: audit missingness, Unknown categories, customer IDs, duplicate profiles, and class balance.
- [x] Before EDA: save stratified 6,076 / 2,025 / 2,026 train/validation/test membership with seed 42.
- [x] Pages 5-7: analyze transaction activity, inactivity, and relationship depth on training rows only; export group counts and churn rates.
- [x] Page 8: exclude CLIENTNUM and both target-derived classifier columns. Fit median/scaling and mode/one-hot preprocessing inside each CV fold.
- [x] Page 9: calculate average ticket and contact ratio using denominator floor 1; validate numeric inputs.
- [x] Pages 10-11: tune the practice's class-weighted random forest with 150 trees, three stratified folds, and its 12-configuration grid, using average precision.
- [x] ADASYN extension: compare numeric-only ADASYN with a matching numeric-only unweighted forest. Keep categorical columns outside this oversampling experiment; do not use SMOTE.
- [x] Select the model family by training CV AP; use validation only to choose a cost threshold (missed churn = 5 units, false alarm = 1 unit).
- [x] Pages 12-14: evaluate frozen choices on test; export AP, ROC-AUC, precision, recall, F1, confusion matrix, Brier score, calibration, and classification costs.
- [x] Report selected-model AP bootstrap uncertainty and descriptive Gender/Income_Category subgroup results with denominators.
- [x] Pages 15-22: explain the frozen selected forest with exact path-dependent TreeSHAP. Verify additivity; export grouped contributions, beeswarm, dependence, and local waterfall.
- [x] Page 23: reproduce the explicitly hypothetical exponential-survival illustration with configurable hazard; distinguish it from empirical survival.
- [x] Page 24: document a randomized retention proposal, future-data validation requirements, monitoring, and launch gates.
- [x] Save and reload the fitted model; verify prediction equality. Record source hash, package versions, configuration, metrics, and all CV trials.

## Remaining data-dependent work

- [ ] Verify snapshot/feature cutoff times and the actual churn event/horizon using dated source records. Current findings remain retrospective.
- [ ] Estimate empirical survival only after observation origin, event dates, and censoring dates are available.
- [ ] Estimate intervention uplift and incremental retained contribution only after treatment/control outcomes and verified margin/cost inputs are available.
- [ ] Validate on an independent future cohort before any operational deployment.

These are unresolved evidence requirements, not unimplemented classifiers. Neither
high model accuracy nor positive SHAP values demonstrates an effective intervention.

## Reproduce

Environment used: Python 3.14. Create a local environment and install pinned packages:

```bash
python3 -m venv w9/.venv
w9/.venv/bin/python -m pip install -r w9/requirements.txt
w9/.venv/bin/python w9/analyze.py
w9/.venv/bin/python w9/test_analyze.py
w9/.venv/bin/python w9/test_pipeline.py
w9/.venv/bin/python w9/churn_pipeline.py --seed 42 --fn-cost 5 --fp-cost 1
```

`--data` overrides the input path; `--jobs` controls CV workers.
`--illustrative-hazard` controls only the hypothetical survival diagram (default 0.035).
The installed local environment reuses existing system packages; requirements pin
the principal numerical/model packages for recreating an isolated environment.

Deliverables: `PIPELINE_REPORT.md`, `figures/*.png`, `outputs/*.csv`,
`outputs/pipeline_metrics.json`, and locally saved `outputs/churn_model.joblib`.
Raw input is unchanged. Model binaries, virtual environments, and Python caches
are excluded from Git. Test scores are evaluation artifacts, not a live contact list.

Success checks: deterministic/disjoint splits; no target or identifier predictors;
training-only fitted transforms and ADASYN; exact validation threshold costs;
reproducible report numbers; SHAP probability reconstruction; saved-model prediction
equality; visually inspected charts. Test data is never used to retune or reselect.
