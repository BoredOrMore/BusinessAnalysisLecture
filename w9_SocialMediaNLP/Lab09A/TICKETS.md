# Ticket Backlog & Verification Audit: Lab 09A (Operational Sentiment & Triage)

Governing Procedure: [`Lab 09A - Operational Sentiment & Triage.pdf`](Lab%2009A%20-%20Operational%20Sentiment%20&%20Triage.pdf)  
Audit Status: **18/18 TICKETS VERIFIED [100% PASS]**  
Execution Script: [`analyze.py`](analyze.py) | Test Suite: [`test_analyze.py`](test_analyze.py)

---

## 1. Summary of Verification Status

| Ticket ID | Epic / Module | Verification Method | Status |
| :--- | :--- | :--- | :---: |
| **TICKET-01** | Ingestion & Row Conservation | `len(df) == 14640`, 0-indexed `source_row` conserved | **PASSED** |
| **TICKET-02** | Quarantine 18 Conflicting Groups | 36 rows flagged `CONFLICTING_LABEL`, `model_eligible=False`, `split='quarantined'` | **PASSED** |
| **TICKET-03** | `negativereason` Leakage Expose | Cross-tabulation proves 100% neg non-null vs 100% neu/pos null; metadata barred | **PASSED** |
| **TICKET-04** | Normalization & Negation Check | Unit tests assert `"not"`, `"cannot"` preserved; `usertoken`, `urltoken` converted | **PASSED** |
| **TICKET-05** | Graph Connected Components | 7,551 components formed; 0.0% author and text overlap between Train and Test | **PASSED** |
| **TICKET-06** | Locked 80/20 Stratified Split | 6,040 train reps (80.0%), 1,511 test reps (20.0%); manifests exported | **PASSED** |
| **TICKET-07** | Mathematical Baselines & Hand Checks | $TF^*=1.0$, $IDF=1.693147$, $L_2$ norm $(3,4) \to (0.6, 0.8)$ assert passed | **PASSED** |
| **TICKET-08** | Dummy Majority Baseline | Accuracy = 61.22%, Macro-F1 = 25.31% recorded as performance floor | **PASSED** |
| **TICKET-09** | 3-Fold Stratified CV on $C$ | $C=0.3 \to 0.5803$, $C=1.0 \to 0.6896$, $C=3.0 \to 0.7192$ (Selected) | **PASSED** |
| **TICKET-10** | Unigram vs Bigram Comparison | Bigram (0.7192) empirically beats Unigram (0.7120) under identical CV folds | **PASSED** |
| **TICKET-11** | Single-Pass Locked Test Eval | Accuracy = 77.43%, Macro-F1 = 70.19%, Neg Recall = 91.57% on test reps | **PASSED** |
| **TICKET-12** | Local Contrast Decomposition | $\Delta z_{\text{direct}} = 0.544102 == \Delta z_{\text{decomposed}}$ (`np.isclose == True`) | **PASSED** |
| **TICKET-13** | Error Analysis Audit | Confusion matrix and error skew towards negative documented in executive report | **PASSED** |
| **TICKET-14** | Multi-Label Aspect Triage | 4 aspects + `no_match` tagged; denominators $n_a$ visible in summary table | **PASSED** |
| **TICKET-15** | 6 Executive Visualizations | All 6 high-res figures generated in `figures/` (300 DPI, no warnings) | **PASSED** |
| **TICKET-16** | Automated Test Suite | `test_analyze.py` executes fresh assertions and exits with code 0 | **PASSED** |
| **TICKET-17** | Executive Findings & Defense | `Lab09A_REPORT.md` (3 observations) & `DATA_DEFENSE.md` (2 pages, citations [A1]-[A7]) | **PASSED** |
| **TICKET-18** | Executed Narrative Notebook | `Lab09A_Sentiment_Analysis.ipynb` JSON validated (11 runnable cells) | **PASSED** |

---

## 2. Detailed Ticket Specifications & Evidence

### [Epic 1] Ingestion, Leakage Audit & Graph Deduplication
- **TICKET-01: Ingestion & Row Conservation ($N=14,640$)**
  - *Evidence:* `airline_registry.parquet` contains exactly 14,640 rows.
  - *Code:* `assert len(raw_df) == 14640`
- **TICKET-02: Quarantine 18 Conflicting `tweet_id` Groups (36 Rows)**
  - *Evidence:* 18 groups identified with contradictory labels. 36 rows isolated with `split='quarantined'`.
  - *Code:* `registry.loc[is_conflicted, 'split'] = 'quarantined'`
- **TICKET-03: Expose `negativereason` Annotation Leakage Trap**
  - *Evidence:* Cross-tabulation in report shows 9,178 negative tweets have non-null reason; all 5,462 neutral/positive tweets are null.
  - *Code:* Features strictly derived from `text_clean` via `TfidfVectorizer`.
- **TICKET-04: Text Normalization & Negation Preservation**
  - *Evidence:* Passed `test_text_normalization()`.
  - *Code:* `assert "not" in clean_text("not good")`
- **TICKET-05: Connected Components Graph Deduplication**
  - *Evidence:* NetworkX graph partitioned 14,604 non-quarantined rows into 7,551 components. Earliest UTC row selected per component.
  - *Code:* `assert len(author_overlap) == 0 and len(text_overlap) == 0`
- **TICKET-06: Locked 80/20 Stratified Split**
  - *Evidence:* 6,040 train representatives, 1,511 test representatives exported in `sentiment_split_manifest.csv`.

### [Epic 2] Modeling Baselines & Cross-Validation
- **TICKET-07: Mathematical Baseline & Hand Checks**
  - *Evidence:* Unit test `test_hand_calculation_arithmetic()` passed.
- **TICKET-08: Dummy Majority Baseline Model**
  - *Evidence:* Dummy classifier achieves Accuracy 61.22%, Macro-F1 25.31% on locked test set.
- **TICKET-09: 3-Fold Stratified Cross-Validation on $C$**
  - *Evidence:* 3-fold CV scores: $C=0.3 \to 0.5803$, $C=1.0 \to 0.6896$, $C=3.0 \to 0.7192$.
- **TICKET-10: Predeclared Unigram vs Bigram Feature Comparison**
  - *Evidence:* Bigram (0.7192) outperforms Unigram-only (0.7120) by +0.72 pp in Macro-F1.

### [Epic 3] Locked Test Evaluation & Local Contrast
- **TICKET-11: Single-Pass Locked Test Set Evaluation**
  - *Evidence:* Accuracy = 77.43%, Macro-F1 = 70.19%, Log Loss = 0.5384, Negative Recall = 91.57% recorded in `classification_report.csv`.
- **TICKET-12: Exact Algebraic Local Contrast Decomposition**
  - *Evidence:* For test sample 0, $\Delta z_{\text{direct}} = 0.544102 == \Delta z_{\text{decomposed}}$.
  - *Code:* `assert np.isclose(delta_z_direct, delta_z_decomposed, atol=1e-5)`
- **TICKET-13: Error Analysis Audit**
  - *Evidence:* Documented in `Lab09A_REPORT.md` Section 2 (skew towards negative predictions reduces risk of missed complaints).

### [Epic 4] Aspect Triage & Visualizations
- **TICKET-14: Multi-Label Keyword Aspect Tagging & Audit**
  - *Evidence:* `aspect_sentiment_summary.csv` records:
    - `delay`: 2,612 tweets (89.82% negative)
    - `baggage`: 1,132 tweets (86.40% negative)
    - `staff`: 1,196 tweets (74.16% negative, 20.40% positive)
    - `refund`: 569 tweets (76.45% negative)
    - `no_match`: 9,828 tweets (52.51% negative)
- **TICKET-15: 6 Executive High-Resolution Figures**
  - *Evidence:* All 6 PNG figures saved in `figures/` at 300 DPI.

### [Epic 5] Automation, Defense & Packaging
- **TICKET-16: Validation Report & Automated Test Suite**
  - *Evidence:* `validation_report.json` valid JSON; `python3 test_analyze.py` passes 100%.
- **TICKET-17: Executive Findings & 2-Page Defense**
  - *Evidence:* `Lab09A_REPORT.md` and `DATA_DEFENSE.md` exist and match all spec guidelines.
- **TICKET-18: Executed Narrative Notebook**
  - *Evidence:* `Lab09A_Sentiment_Analysis.ipynb` JSON structure validated with 11 cells.
