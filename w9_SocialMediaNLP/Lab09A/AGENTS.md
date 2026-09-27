# Agent Guidelines: Week 09A — Social Media NLP (Operational Sentiment & Triage)

Governing Procedure: [`Lab 09A - Operational Sentiment & Triage.pdf`](Lab%2009A%20-%20Operational%20Sentiment%20&%20Triage.pdf)  
Course: **961731 Business Data Analytics** (Chiang Mai University, Data Science Program)  
Primary Dataset: CrowdFlower Twitter US Airline Sentiment (`Tweets.csv`, 14,640 records)  
Case Directory: `w9_SocialMediaNLP/Lab09A/`  
Outputs Directory: `Lab09A_outputs/` | Figures Directory: `figures/`

---

## 1. Executive Context & Objective

Social media listening dashboards frequently confuse message topic, evaluative sentiment, and operational triage. This lab establishes an audited, reproducible pipeline for three-class sentiment classification (`negative`, `neutral`, `positive`), keyword-based aspect categorization, and executive triage reporting on 14,640 airline service tweets.

### Key Analytical Distinctions
- **Sentiment vs. Aspect:** "Baggage" is an operational aspect/topic; "terrible" is evaluative sentiment. Keyword dictionary matches are descriptive heuristics, never gold-standard aspect classifiers.
- **Corpus vs. Public Opinion:** Twitter customer complaints represent an operational queue of vocal users, not an unbiased probability sample of all airline passengers.
- **Direction vs. Polarity:** Message direction or ticket existence is distinct from sentiment.

---

## 2. Anti-Leakage & Integrity Guardrails

1. **Strict Text-Only Feature Space:**  
   Models must train *only* on features extracted from `text_clean`.  
   **Forbidden predictors:** `negativereason`, `negativereason_confidence`, `airline_sentiment_confidence`, `airline`, `name`, `retweet_count`, `tweet_coord`, `tweet_location`, `user_timezone`, `tweet_id`.
2. **Annotation Leakage Trap:**  
   `negativereason` is populated almost exclusively for `negative` tweets. Utilizing its presence or content as a predictor is an invalid annotation leak. Document and demonstrate this leakage trap via a cross-tabulation of missingness against `airline_sentiment`.
3. **Quarantine Conflicting Identifiers:**  
   The source dataset contains 18 `tweet_id` groups with conflicting sentiment annotations.  
   **Rule:** Every row belonging to a conflicting `tweet_id` group must be flagged and **quarantined from training and test sets**. Never resolve conflicts by majority voting or heuristic relabeling.
4. **Author & Text Deduplication (Component Splitting):**  
   Rows sharing a non-missing author (`name`) or identical normalized text (`text_clean`) must be linked into connected components. Retain only the **earliest representative** per component (tie-break: smallest `tweet_id`, then smallest `source_row`). Reserve a stratified 20% for test, 80% for train.
5. **No Data-Dependent Tuning on Test Set:**  
   The test set must be locked. Open it **exactly once** after all preprocessing, vocabulary selection, and hyperparameter tuning are finalized using training-only cross-validation.

---

## 3. Data Schema & Ingestion Contract

Read `Tweets.csv` and construct `airline_registry.parquet` preserving all 14,640 source rows.

| Field Name | Type | Ingestion & Handling Rule |
| :--- | :--- | :--- |
| `source_row` | `int64` | Primary key: 0-indexed row position (0 to 14,639). Must be strictly conserved. |
| `tweet_id` | `string` | Source post identifier. Preserve as raw string; never cast to float. |
| `text` | `string` | Raw unmodified post language. |
| `airline_sentiment` | `category` | Supervised ground truth: `'negative'`, `'neutral'`, `'positive'`. |
| `negativereason` | `string` (nullable) | Operational category. Exclude from modeling; audit missingness. |
| `retweet_count` | `int64` | Engagement metric. Non-numeric inputs coerced; exclude from predictors. |
| `name` | `string` | Author account handle. Used strictly for clustering and deduplication. |
| `tweet_created` | `string` / `datetime` | Parse to UTC timestamp for earliest-representative determination. |
| `text_clean` | `string` | Deterministically normalized text (see Section 4). |
| `audit_flags` | `string` | Delimited flags: `CONFLICTING_LABEL`, `EMPTY_TEXT`, `INVALID_TIME`, etc. |
| `component_id` | `int64` | Connected component ID linking shared authors and duplicate texts. |
| `is_representative`| `bool` | True for earliest row per component; False for duplicate/linked rows. |
| `model_eligible` | `bool` | `True` iff valid label, non-quarantined, non-empty `text_clean`, and `is_representative == True`. |
| `split` | `category` | `'train'`, `'test'`, or `'ineligible'`/`'quarantined'`. |

### Intentional Defect Injection Audit
Scripts must test and handle:
- Whitespace padded `tweet_id` strings (detect collisions post-trim).
- Stress-copy texts with raw URLs, `@mentions`, and non-ASCII glyphs.
- Injected unknown sentiment labels (flagged as ineligible, not imputed).
- Corrupted timestamps and non-numeric retweet values.

---

## 4. Text Normalization Contract (`text_clean`)

Transform `text` into `text_clean` using reproducible, deterministic steps:
1. **Unicode & Whitespace:** Apply `unicodedata.normalize("NFKC", text)`, strip leading/trailing whitespace, and collapse internal whitespace runs to a single space.
2. **Mentions & URLs:** Replace user mentions (`@[A-Za-z0-9_]+`) with `usertoken`. Replace URLs (`http\S+|www\.\S+`) with `urltoken`.
3. **Hashtag Retention:** Remove the `#` character but preserve the word tokens (e.g., `#DelayedFlight` -> `DelayedFlight`).
4. **Contractions:** Systematically expand contractions (`don't` -> `do not`, `can't` -> `cannot`, `won't` -> `will not`, `i'm` -> `i am`, `it's` -> `it is`).
5. **Negation Preservation (Critical):** Never delete negation words (`not`, `no`, `never`, `neither`, `nor`, `without`). Compare `"good"` vs `"not good"` across transformations to confirm semantic distinction.
6. **Case Normalization:** Lowercase tokens after entity masking.
7. **Logging Requirement:** Print at least five raw/clean pairs covering negation, emojis, hashtags, mentions, and URLs.

---

## 5. Mathematical Formulations & Hand Baselines

All implementations must reproduce these documented analytical formulations:

### Sublinear Term Frequency ($TF^*$)
$$TF^*(t, d) = \begin{cases} 1 + \ln(c(t, d)) & \text{if } c(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Smoothed Inverse Document Frequency ($IDF$)
$$IDF(t) = \ln\left(\frac{N_{train} + 1}{df(t) + 1}\right) + 1$$
*Hand Verification:* For $N_{train} = 3$ documents, if term $t$ appears in 1 document ($df(t)=1$):
$$IDF(t) = \ln\left(\frac{3 + 1}{1 + 1}\right) + 1 = \ln(2) + 1 \approx 0.6931 + 1 = 1.6931$$

### $L_2$ Row Normalization
$$v_{norm} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_j v_j^2}}$$
*Hand Verification:* Vector $(3, 4) \to \left(\frac{3}{5}, \frac{4}{5}\right) = (0.6, 0.8)$. Nonzero check required before division; zero rows remain zero.

### Multiclass Softmax & Cross-Entropy Loss
$$P(y = k \mid x) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}, \quad \text{where } z_k = b_k + w_k^T x$$
$$\mathcal{L}_{CE} = -\sum_{k=1}^K y_k \ln(P(y = k \mid x))$$
*Hand Verification:* Unnormalized scores $z = (\ln 2, 0, 0) \approx (0.6931, 0, 0)$.
- Numerators: $(\exp(\ln 2), \exp(0), \exp(0)) = (2, 1, 1)$. Sum = $4$.
- Softmax vector: $(0.50, 0.25, 0.25)$.
- Loss if ground truth is class 0: $-\ln(0.50) \approx 0.6931$.
- Loss if ground truth is class 1 or 2: $-\ln(0.25) \approx 1.3863$.

---

## 6. Model Training & Evaluation Protocol

### Baseline Model
- `DummyClassifier(strategy='prior')`: predicts majority class based on training class prior. Evaluated to establish the minimum performance floor for accuracy and macro-F1.

### Primary Modeling Pipeline
- **Vectorizer:** `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, smooth_idf=True, norm='l2', max_features=5000, min_df=2)`
- **Classifier:** `LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=961731)`
- **Cross-Validation:** 3 stratified training-only folds (`StratifiedKFold(n_splits=3, shuffle=True, random_state=961731)`).
- **Hyperparameter Grid:** $C \in \{0.3, 1.0, 3.0\}$.
- **Selection Metric:** Mean Macro-F1 across folds. Record fit times, fold scores, and convergence.
- **Predeclared Comparison:** Within the same 3 folds, compare Unigram-only (`ngram_range=(1, 1)`) vs. Unigram+Bigram (`ngram_range=(1, 2)`).

### Locked Test Set Evaluation
Refit selected pipeline on all training representatives and evaluate once on locked test:
- Metrics: Support, count confusion matrix, row-normalized confusion matrix, overall accuracy, Macro-F1, per-class Precision, Recall, and F1 (with `zero_division=0`), and multiclass log loss.
- **Error Analysis:** Audit a fixed-seed sample of test errors for negation, sarcasm, spelling variants, and mixed sentiment.
- **Interpretability & Local Contrast:**
  - Extract and plot top 10 positive coefficients for each sentiment class.
  - Compute local contrast for a test case between predicted class $k$ and runner-up class $j$:
    $$\Delta z_{kj} = z_k - z_j = (b_k - b_j) + \sum_{i} (w_{ki} - w_{ji}) x_i$$
  - Verify that the analytical sum of coefficient contributions matches the score difference numerically.

---

## 7. Keyword Aspect Tagging & Visualization

### Aspect Keyword Dictionaries
Define four operational dictionaries before examining test predictions:
- `delay`: `\b(delay|delayed|delays|late|cancel|cancelled|cancellation|holding|hours? late|on hold)\b`
- `baggage`: `\b(bag|bags|baggage|luggage|suitcase|lost bag|claim)\b`
- `staff`: `\b(staff|crew|agent|attendant|representative|flight attendant|gate agent|employee|pilot)\b`
- `refund`: `\b(refund|refunds|refunded|voucher|reimburse|reimbursement|compensation|fee|credit)\b`
- `no_match`: Posts matching none of the above.

### Aspect Audit & Visualizations
- **Audit:** Sample 30 fixed-seed tagged posts and 20 no-match posts; document precision, false matches, and missed paraphrases.
- **Figures Required:**
  1. `figures/sentiment_counts.png`: Class balance bars with counts and percentages.
  2. `figures/airline_sentiment_stacked.png`: 100% stacked sentiment distribution by airline.
  3. `figures/aspect_sentiment_heatmap.png`: Aspect-by-sentiment proportion heatmap with denominators $n_a$ visible.
  4. `figures/training_top_bigrams.png`: Top bigrams by training document frequency.
  5. `figures/top_coefficients.png`: Top 10 positive linear coefficients per class.
  6. `figures/confusion_matrices.png`: Raw count and row-normalized confusion matrices.

---

## 8. Deliverables & Submission Artifacts

All outputs reside in `w9_SocialMediaNLP/Lab09A/`:
- **Code & Notebook:**
  - `analyze.py`: Standalone reproducible Python script with CLI options.
  - `Lab09A_Sentiment_Analysis.ipynb`: Fully executed narrative notebook.
- **Data & Manifests (`Lab09A_outputs/`):**
  - `airline_registry.parquet`: Exactly 14,640 rows with audit flags and component splits.
  - `sentiment_split_manifest.csv`: Row split mapping (`source_row`, `tweet_id`, `split`).
  - `sentiment_predictions.csv`: Out-of-sample test predictions, probabilities, and true labels.
  - `classification_report.csv`: Complete precision, recall, F1, and support table.
  - `aspect_sentiment_summary.csv`: Aspect tag counts, denominators, and sentiment breakdowns.
  - `validation_report.json`: Invariant check logs, random seed, system metadata, and metrics.
- **Executive Reports:**
  - `Lab09A_REPORT.md`: Findings board with 3 business insights, counterexamples, and operational triage recommendations.
  - `DATA_DEFENSE.md`: Two-page defense addressing data leakage, representation choices, and ethical listening limits.
- **Submission Archive:**
  - `961731_studentID_Lab09A.zip` packaging code, figures, manifests, and reports.

---

## 9. Evaluation Rubric Checklist (100 Points)

| Category | Points | Core Deliverable Criteria |
| :--- | :---: | :--- |
| **Diagnosis** | 25 | Ingestion & leakage audit (10); text normalization & independent units (10); TF-IDF & probability arithmetic (5). |
| **Technical Execution** | 35 | Sparse model & dummy baseline (15); metrics & coefficient explanation (10); audited aspect tags & normalized charts (10). |
| **Methodological Defense**| 25 | Valid selection / locked test split (10); sentiment vs. aspect distinction & limitations (10); academic citations (5). |
| **Automation** | 15 | Fresh-run reproducibility (5); passing invariant assertions (5); complete versioned submission packaging (5). |

---

## 10. Verification & Execution Commands

Run from `w9_SocialMediaNLP/Lab09A/`:

```bash
# Check Python syntax and argument parser
python3 -m py_compile analyze.py
python3 analyze.py --help

# Execute complete analytical pipeline
python3 analyze.py --seed 961731 --data data/Tweets.csv

# Verify invariants and outputs
git status --short
ls -lh Lab09A_outputs/
ls -lh figures/
```
