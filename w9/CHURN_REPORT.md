# Customer Churn Report

The completed practice study is in [PIPELINE_REPORT.md](PIPELINE_REPORT.md), with
model comparisons, ADASYN, test evaluation, SHAP, and retention limitations.
The original business brief below is retained as context. Its top-10% capacity
scenario is superseded by the practice's validation-selected 5:1 error-cost rule.

## Module 1 — Business understanding

**Status:** business brief complete under explicit teaching assumptions. Historical
models have now been evaluated in the linked pipeline report; no retention campaign
has been evaluated.

**Sources:** [Customer Churn lecture](<Week 08 - Customer Churn.pdf>), Module 1, PDF pages 3–9; [analysis plan](plan.md); local `data/BankChurners.csv`. Model metric choices follow the plan and repository analytical standards. Proposed operating settings below are implementation assumptions, not dataset facts.

### 1. Business question, owner, and action

**Question:** which active credit-card customers should a retention team prioritize, within limited contact capacity, to increase retained contribution after intervention costs?

The current analytical task is narrower: benchmark classification of the supplied historical attrition label. Future churn prediction and increased retention require dated outcomes and an intervention evaluation.

| Decision | Working definition | Evidence status |
|---|---|---|
| Decision owner | Credit-card retention manager; Finance supplies contribution and cost inputs; analytics supplies evaluated rankings | Proposed roles; no actual owner identified |
| Eligible population | Customers with an active card relationship at decision time, reachable and eligible for the chosen action | Proposed operational rule; the historical CSV is not a live eligible-customer list |
| Candidate actions | Service-review outreach, a low-cost reminder, or a targeted retention offer | Options for the exercise; availability and effectiveness unverified |
| Capacity | Rank customers and evaluate the highest-risk 10% as the initial teaching scenario | Assumed contact fraction, not a known staffing limit |
| Costs | Per-contact handling cost plus expected offer/redemption cost, using a consistent currency and horizon | Unknown; do not substitute zero |
| Intended outcome | More customers retained because of the action, with positive incremental contribution after costs | Requires treatment/control evidence |

At scoring time, already-closed relationships are ineligible. The historical attrition label is used as the outcome for evaluation, never as an eligibility filter that removes positive examples from training.

### 2. Define churn and its clock

The lecture distinguishes **contract churn** (cancellation or nonrenewal) from **behavioral churn** (no qualifying activity for a defined interval). These definitions are not interchangeable. Inactivity can be a predictor without being the target itself. [Lecture, p. 5](<Week 08 - Customer Churn.pdf#page=5>)

For this CSV, define `y = 1` when `Attrition_Flag == "Attrited Customer"` and `y = 0` when it is `"Existing Customer"`. This is the supplied attrition status; the file does not establish its exact closure rule, effective date, or measurement period. Do not redefine it using `Months_Inactive_12_mon`.

For a future operational dataset, use this **proposed teaching specification**:

- Decision time `t0`: start of a monthly scoring cycle, after the previous period's data is available.
- Observation window: the previous 12 months, `[t0 - 12 months, t0)`, restricted to information actually available at `t0`. For newer customers, record the shorter observed history explicitly.
- Prediction horizon: `(t0, t0 + 30 days]`, following the lecture's example.
- Proposed event: confirmed closure of the customer's last active card relationship during that horizon. Product operations must supply the actual event definition and distinguish customer attrition from a single-card replacement.
- Label maturity: assign a negative label only after the complete horizon has elapsed and outcome records are available. Keep incomplete follow-up out of binary evaluation.

The CSV has no snapshot dates or event timestamps. Its aggregate fields cannot establish this timeline or prove that features precede attrition. Until dated data is supplied, describe results as retrospective classification, not validated next-30-day predictions.

### 3. Measure the loss correctly

| Measure | Definition | What this file supports |
|---|---|---|
| Dataset attrition prevalence | Labeled attrited customers / all sampled customers | 1,627 / 10,127 = **16.07%** |
| Logo churn over a period | Starting customers lost during the period / customers active at period start | Not measurable: starting population and event dates are absent |
| Revenue churn over a period | Recurring revenue lost from starting customers / starting recurring revenue | Not measurable: recurring revenue and period boundaries are absent |
| Cohort retention at elapsed time | Original-cohort customers still active / original cohort size | Not measurable: cohort entry dates and subsequent activity are absent |

The lecture's logo/revenue/cohort definitions require explicit denominators and clocks. Customer acquisition can offset losses in net growth without eliminating churn. Do not describe the observed 16.07% as monthly churn, annual churn, or a population estimate without sampling evidence. [Lecture, pp. 6–7](<Week 08 - Customer Churn.pdf#page=6>)

`Total_Trans_Amt` measures transaction spending; `Credit_Limit` measures available credit. Neither is observed bank revenue, profit, or CLV. These fields cannot price the business loss by themselves.

### 4. Establish a baseline and success criteria

Verified on the full file for descriptive framing only; these are not holdout model results:

| Baseline quantity | Verified value |
|---|---:|
| Customer rows / unique customer IDs | 10,127 / 10,127 |
| Existing customers | 8,500 |
| Attrited customers | 1,627 |
| Always-stay accuracy | 83.93% |
| Always-stay churn recall | 0.00% |
| Always-stay TP / FP / FN / TN | 0 / 0 / 1,627 / 8,500 |
| Constant-score average precision | 16.07% |

Always-stay misses every attrited customer. Churn precision is undefined because it predicts no positives; if software reports zero by convention, label that convention. Constant-score average precision equals class prevalence and provides a ranking baseline alongside the always-stay decision baseline.

Predefine these rules before training:

1. **Primary model-selection metric:** mean cross-validation average precision (AP). Compare against constant-score AP on the same folds; do not assume the full-file 16.07% is every fold's baseline. AP is the chosen precision–recall summary, not trapezoidal PR area.
2. **Capacity metric:** report precision@K, recall@K, and lift@K for the top 10% scenario, with `K = max(1, floor(0.10 * N))` for each evaluated population. Break equal-score ties deterministically by `CLIENTNUM`; keep IDs out of predictors. The denominator is the evaluation population, not automatically all 10,127 rows.
3. **Comparison:** random targeting has expected precision equal to prevalence and expected recall `K / N`. Define `lift@K = precision@K / prevalence`. Seek AP above its baseline and lift@K above 1, with uncertainty reported; a small noisy gain does not establish useful improvement.
4. **Selection and final gate:** choose the model and any probability cutoff from training-fold predictions only. Freeze choices before final holdout evaluation. Report failure honestly if the selected model does not beat its corresponding baseline; prefer the simpler model when improvement is unconvincing.
5. **Business gate:** require measured positive incremental retention and net contribution after costs in a later intervention evaluation. No numerical retention-lift target, minimum meaningful AP gain, or actual contact budget has been supplied; these remain planning inputs, not achieved results.

A capacity ranking is sufficient for the initial benchmark. Do not assume that a score of 0.5 is the appropriate cutoff or that an uncalibrated score is an accurate probability.

### 5. Connect value to a decision

The lecture introduces `CLV ≈ monthly margin * expected retention months - acquisition cost` as a simplified lifetime-value illustration. Its example values are not BankChurners estimates. [Lecture, p. 8](<Week 08 - Customer Churn.pdf#page=8>)

For an existing-customer retention decision, compare the **incremental future contribution caused by the action** with incremental action costs. Historical acquisition spending is already incurred; do not charge it again to a new retention decision. Keep contribution values, time horizons, and discount assumptions consistent.

Working rule: prioritize eligible customers within capacity only when evidence supports positive expected incremental net value. Before response and value data exists, a risk ranking supports review and experiment design; it does not justify an automated offer or establish ROI. Model accuracy and predicted risk alone cannot show that contact changes outcomes.

### 6. Inputs needed for operational use

| Missing input | Proposed provider | Decision it enables |
|---|---|---|
| Label definition, dated closures, snapshot dates, feature availability times | Product operations and data owner | Future-horizon labels and temporal leakage checks |
| Active population, contact eligibility, reachable channels | Retention operations | Valid campaign audience |
| Actual contact capacity, action catalogue, per-contact/offer cost | Retention manager | Feasible budget and action choice |
| Contribution margin, value horizon, expected retained value | Finance | Comparable incremental-value calculation |
| Customer-level treatment assignments and later outcomes | Experiment owner | Causal retention lift and campaign value |
| Minimum worthwhile improvement and review thresholds | Decision owner | Business acceptance criteria |

These gaps do not prevent Module 2's descriptive data audit. They do prevent claims of validated future churn, measured revenue loss, or profitable intervention. Module 1's completed output is this documented brief and baseline, with operating assumptions still provisional.

### Reproduce the dataset baseline

Run from repository root with Python's standard library. This reads the CSV without changing it; all reported baseline values are derived from rows rather than embedded as analytical constants.

```bash
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

with Path("w9/data/BankChurners.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
mapping = {"Existing Customer": 0, "Attrited Customer": 1}
assert rows and all(row["Attrition_Flag"] in mapping for row in rows)
assert all(row["CLIENTNUM"].strip() for row in rows)
assert len({row["CLIENTNUM"] for row in rows}) == len(rows)
counts = Counter(mapping[row["Attrition_Flag"]] for row in rows)
n, stay, churn = len(rows), counts[0], counts[1]
assert stay > 0 and churn > 0 and stay + churn == n
tp, fp, fn, tn = 0, 0, churn, stay
assert tp + fp + fn + tn == n
print(f"Rows / unique IDs: {n:,} / {n:,}")
print(f"Existing: {stay:,}; attrited: {churn:,}")
print(f"Attrition prevalence: {churn / n:.2%}")
print(f"Always-stay accuracy: {(tp + tn) / n:.2%}")
print(f"Always-stay recall: {tp / (tp + fn):.2%}")
print(f"TP / FP / FN / TN: {tp:,} / {fp:,} / {fn:,} / {tn:,}")
print(f"Constant-score AP: {churn / n:.2%}")
PY
```
