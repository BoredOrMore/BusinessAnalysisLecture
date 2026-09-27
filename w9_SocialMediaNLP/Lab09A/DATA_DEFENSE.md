# Methodological & Analytical Defense: Lab 09A

**Author:** Business Data Analytics Team (Course 961731)  
**Deliverable:** Methodological Defense Document (2-Page Defense Standard)  
**Topic:** Social Media Sentiment Analysis, Representation Choices, Leakage Controls, and Operational Listening Limits

---

## 1. Data Integrity, Leakage Controls, and Graph Deduplication

Supervised natural language processing on user-generated social media introduces subtle but catastrophic forms of data leakage. This study establishes three strict architectural safeguards:

### 1.1 Strict Isolation of the 18 Conflicting Ground-Truth Groups
The CrowdFlower dataset contains 18 `tweet_id` groups (comprising 36 individual rows) where human annotators assigned opposite sentiment categories (e.g. Row 12019 labeled `neutral` vs Row 12180 labeled `negative`).  
- **Defense against Relabeling / Majority Voting:** In a two-annotator contradiction (1 vs 1 tie), any tie-breaking heuristic or confidence thresholding is unvalidated speculation. Relabeling introduces label noise directly into evaluation benchmarks.
- **Defense against Row Deletion:** Deleting rows silently destroys data provenance and alters the verified population denominator ($N = 14,640$).
- **The Quarantine Solution:** Every conflicting record is flagged as `audit_flags = 'CONFLICTING_LABEL'` and assigned `split = 'quarantined'`. They remain 100% visible in `airline_registry.parquet` for data auditing, but are strictly excluded from training folds and test sets.

### 1.2 Neutralization of the `negativereason` Annotation Trap
Cross-tabulation across all 14,640 records proves that `negativereason` is populated in exactly 100.0% of `negative` tweets ($9,178$ records) and is 100.0% null ($0$ records) in `neutral` and `positive` classes.  
- **Mechanism:** Annotators were presented with the complaint category dropdown *only after* selecting a negative sentiment label.
- **Leakage Reality:** Creating an indicator variable `is_negativereason_null` yields a trivial decision rule with 100% precision and recall for negative tweets on test data, while possessing zero utility in production where incoming raw tweets lack manual annotations. All non-text metadata was barred from feature matrices.

### 1.3 Graph Connected Components Deduplication
Naive random row splitting permits identical complaint texts or tweets from the same prolific passenger to populate both training and test sets, artificially inflating test metrics.  
- By constructing bipartite graphs connecting rows that share non-missing author handles (`name`) or exact normalized strings (`text_clean`), we identified **7,551 independent customer components**.
- Selecting the earliest representative per component by UTC timestamp guarantees that the locked 20% test partition contains **0.0% author overlap and 0.0% text overlap** with the training set.

---

## 2. Text Representation Choices & Mathematical Foundations

### 2.1 Sublinear Term Frequency ($TF^*$)
Standard term frequency $c(t, d)$ assumes linear scaling between token frequency and document relevance. In social media, emotional users frequently repeat tokens for emphasis (e.g., `"delay delay delay"`). Sublinear scaling:
$$TF^*(t, d) = 1 + \ln(c(t, d)) \quad \text{for } c > 0$$
dampens the leverage of repetitive venting while preserving word presence.

### 2.2 Smoothed Inverse Document Frequency ($IDF$)
To prevent division-by-zero on rare tokens and stabilize weights against small training samples, we employ scikit-learn's smoothed formulation:
$$IDF(t) = \ln\left(\frac{N_{train} + 1}{df(t) + 1}\right) + 1$$
Adding $1$ to both the numerator and denominator acts as a pseudo-document prior, preventing infinite weights on singleton words.

### 2.3 $L_2$ Vector Normalization
Short tweets (e.g., `"bad service"`) and long 140-character complaints vary dramatically in raw Euclidean norm. Row-wise $L_2$ normalization:
$$v_{norm} = \frac{v}{\|v\|_2}$$
ensures that classification decisions depend strictly on the angular orientation (cosine direction) of text topics rather than message length.

### 2.4 Negation Preservation and N-Gram Expansion
Conventional NLP pipelines blindly purge punctuation and apply broad stop-word lists that eliminate `"not"`, `"never"`, `"no"`, and `"without"`. In sentiment analysis, deleting negation collapses polar opposites: `"not good"` becomes `"good"`. Our pipeline explicitly retains negation tokens and expands the vocabulary to bigrams ($1$-to-$2$ n-grams), allowing the linear model to assign large positive weights to `"good"` and large negative weights to `"not good"` simultaneously.

---

## 3. Supervised Model Selection & Explainability Protocol

### 3.1 Training-Only 3-Fold Cross-Validation
Model selection must not observe out-of-sample data. Within the training partition ($N_{train} = 6,040$), we executed a 3-fold stratified cross-validation search over inverse regularization strength $C \in \{0.3, 1.0, 3.0\}$.
- Regularization selection was driven by **Macro-averaged F1**, preventing the model from collapsing into trivial majority-class prediction.
- Candidate $C = 3.0$ achieved the highest cross-validation score ($\text{Macro-F1} = 0.7192$), demonstrating that relaxed regularization captures subtle lexical indicators in short-text social streams.

### 3.2 Exact Local Contrast Decomposition
Black-box sentiment scoring leaves operational staff unable to justify automated triage decisions. For any prediction, our pipeline provides closed-form, mathematically exact explanations:
$$\Delta z_{kj} = z_k - z_j = (b_k - b_j) + \sum_{i} (w_{ki} - w_{ji}) x_i$$
This identity separates the baseline class prior ($(b_k - b_j)$) from token-level contributions ($(w_{ki} - w_{ji}) x_i$). Every production triage decision can be audited down to exact floating-point contributions without surrogate approximations (e.g. LIME/SHAP sampling error).

---

## 4. Aspect Heuristics vs. Evaluative Sentiment & Social Listening Limits

### 4.1 Orthogonality of Aspect and Sentiment
A common industry mistake is treating topic keywords as sentiment proxies. A tweet containing `"baggage"` is not inherently negative; it merely identifies an operational department. Our empirical findings demonstrate:
- `staff` keywords encompass both severe complaints (74.16% negative) and passionate compliments (20.40% positive).
- Collapsing aspect and sentiment into a single tag destroys the ability to reward exceptional service personnel or isolate localized operational bottlenecks.

### 4.2 Transparent Denominators in Multi-Label Triage
Because a single customer message can mention both a flight cancellation and lost luggage, keyword tags overlap ($10$ posts match both `delay` and `baggage`). Summing raw aspect counts produces double-counted totals that distort complaint volume. Every metric must be published alongside its exact subset denominator $n_a$.

### 4.3 Limits of Twitter Social Listening
Customer support tweets do not represent an unbiased probability sample of all airline travelers:
1. **Sampling Bias:** Vocal social media users over-index on severe disruption events; satisfied passengers rarely tweet about on-time departures.
2. **Channel Shift:** Older demographics or corporate travelers frequently utilize telephone and corporate portal channels rather than public Twitter handles.
3. **Descriptive Scope:** Model outputs represent triage priorities for this specific digital support queue, not universal brand equity scores.

---

## 5. Primary Source & Academic Citations

- **[A1] CrowdFlower / Figure Eight (2015):** *Twitter US Airline Sentiment*. Public repository of customer support interactions across six major carriers. Supervised reference standard for 3-class sentiment.
- **[A2] Scikit-learn Developers (2024):** *Text Feature Extraction and Common Leakage Pitfalls*. Documentation on `TfidfVectorizer`, sublinear scaling, and pipeline encapsulation.
- **[A3] Scikit-learn Developers (2024):** *Logistic Regression and Multiclass Loss Optimization*. Technical reference for multinomial cross-entropy and L-BFGS convergence.
- **[A4] Lourentzou, I., Manghnani, K., & Zhai, C. (2019):** *Adapting Sequence to Sequence Models for Text Normalization in Social Media*. arXiv:1904.06100v1. Justification for entity masking and contraction normalization.
- **[A5] Eisner, B. et al. (2016):** *emoji2vec: Learning Emoji Representations from their Description*. arXiv:1609.08359v2. Semantic properties of non-alphanumeric sentiment tokens.
- **[A7] Camacho-Collados, J. et al. (2022):** *TweetNLP: Cutting-Edge Natural Language Processing for Social Media*. arXiv:2206.14774v3. Methodological standards for Twitter text preprocessing and stance/sentiment distinction.
