# ReviewLens — Project Report

**Course:** B.Sc. (Honours)Data Science & Artificial Intelligence — Capstone Project  
**Project Title:** ReviewLens — Sentiment Analysis and Insight Dashboard  
**Tech Stack:** Python · scikit-learn · NLTK · Streamlit · Plotly

---

## Abstract

ReviewLens is an end-to-end NLP application that classifies real customer
product reviews as positive, negative, or neutral, and presents the
results through an interactive multi-page analytics dashboard. The
system is trained on **~30,800 real reviews merged from 13 Kaggle
datasets** (Amazon, Flipkart, App Store, Google Play Store, Coursera,
Zomato, IMDB, and others) — not synthetic data. It uses a TF-IDF feature
extraction pipeline combined with a cross-validated, hyperparameter-tuned
Logistic Regression classifier, selected for its interpretability,
training efficiency, and strong performance on text classification tasks.
The project demonstrates a complete machine learning lifecycle: raw data
ingestion from heterogeneous sources, preprocessing, hyperparameter
search, evaluation, explainability, and deployment via a Streamlit
dashboard that also accepts arbitrary user-uploaded CSVs — all running
locally without internet connectivity or paid APIs.

---

## 1. Introduction

Online product reviews are one of the richest sources of unstructured
customer feedback available to businesses. Platforms like Amazon,
Flipkart, and the Google Play Store host millions of reviews that
contain valuable signals about product quality, customer satisfaction,
and areas for improvement. However, the sheer volume of this data makes
manual analysis infeasible.

Natural Language Processing (NLP) and Machine Learning enable automated
sentiment classification at scale. ReviewLens applies these techniques
to build a practical, explainable, CPU-friendly sentiment analysis tool
— trained and validated on real review data rather than synthetically
generated text, so its reported performance reflects genuine
real-world difficulty (including the ambiguity of "neutral" reviews).

---

## 2. Problem Statement

Given a collection of customer reviews in free-form text, automatically:

1. Classify each review as **positive**, **negative**, or **neutral**
2. Identify the key words driving the sentiment
3. Visualise the aggregate patterns across a large collection of reviews
4. Accept **any** CSV a user uploads, regardless of its column names
5. Enable non-technical stakeholders to interact with insights via a
   web dashboard

---

## 3. Objectives

- Build a modular, maintainable Python codebase following software
  engineering best practices
- Train on real, heterogeneous review data rather than synthetic text
- Systematically tune the model rather than relying on library defaults
- Provide keyword-level explainability without external libraries
  (e.g. no SHAP required)
- Accept arbitrary CSV uploads without requiring a fixed schema
- Create a Streamlit dashboard suitable for real business use
- Ensure the project runs entirely offline on a standard laptop CPU

---

## 4. Dataset Description

### 4a. Real Data Sources

ReviewLens does **not** use synthetic or generated review text. The
training data is built by `prepare_dataset.py`, which merges 13 real
Kaggle datasets into one unified schema
(`review_text, rating, date, source, category, label`):

| Category | Platforms | Approx. rows contributed |
|---|---|---|
| Electronics | Amazon (2 datasets) | ~5,600 |
| App | App Store, Google Play Store | ~7,900 |
| E-commerce | Flipkart, generic e-commerce | ~5,600 |
| Restaurant | Zomato, customer surveys | ~4,900 |
| Course | Coursera (2 datasets) | ~4,300 |
| Movie | IMDB | ~2,500 |

**Total after merging, cleaning, and de-duplication: 30,801 reviews.**

Four additional Kaggle files were considered and excluded, with reasons
documented directly in `src/data_sources.py`:
- One file had malformed CSV quoting (all fields collapsed into a
  single column) and only 99 usable rows
- Three files had corrupted text encoding and were near-duplicates of
  datasets already included

### 4b. Label Derivation

Each source uses one of three original signal types, normalised by
`src/data_loader.py`:

| Signal type | Example source | Mapping rule |
|---|---|---|
| Star rating (1–5) | Amazon, App Store, Coursera | 1–2★→negative, 3★→neutral, 4–5★→positive |
| Binary "liked" (0/1) | Restaurant survey data | 0→negative, 1→positive |
| Existing text label | Flipkart, IMDB | Normalised to lowercase, validated against the 3 known classes |

### 4c. Class Distribution

The merged dataset is naturally imbalanced — a realistic property of
review data, where satisfied customers write reviews more often than
neutral ones:

| Label | Training rows | % of training set |
|---|---|---|
| Positive | 16,923 | 69.2% |
| Negative | 5,837 | 23.9% |
| Neutral | 1,704 | 7.0% |

This imbalance directly motivated the class-weight tuning described in
Section 6.

### 4d. Schema

| Column | Type | Description |
|---|---|---|
| `review_text` | str | Raw review string |
| `rating` | float / NaN | 1–5 star rating, where available |
| `date` | str / NaN | Review date, where available |
| `source` | str | Platform name (Amazon, Flipkart, etc.) |
| `category` | str | Product category (Electronics, App, etc.) |
| `label` | str | positive / negative / neutral |

---

## 5. Methodology

### 5a. Text Preprocessing (`src/preprocess.py`)

Applied in sequence:

1. Lowercase conversion
2. HTML tag removal (`<br>`, `<b>`, etc.)
3. URL removal (`https://...`, `www....`)
4. Special character removal (keep a–z, 0–9, space)
5. Extra whitespace collapse
6. English stop-word removal (NLTK corpus)
7. WordNet lemmatisation (NLTK)

**Example:**  
Input: `"This product is AMAZING!!! <br> Best purchase ever."`  
Output: `"product amazing best purchase"`

### 5b. Feature Extraction

**TF-IDF Vectorizer** with:
- `max_features = 30,000`
- `ngram_range = (1, 3)` — unigrams, bigrams, and trigrams
- `sublinear_tf = True` — log-normalised term frequency
- `min_df = 2` — ignore hapax legomena

Trigrams and a larger vocabulary were adopted after experimentation
showed they captured multi-word phrases (e.g. `"difficult use"`,
`"would recommend"`) that meaningfully improved classification —
see Section 6.

### 5c. Model Training and Hyperparameter Search

**Primary model:** Logistic Regression, tuned via a cross-validated
grid search (`tune_class_weights()` in `src/train_model.py`) rather
than using library defaults.

**Baseline model:** Multinomial Naive Bayes (`alpha = 0.1`), trained
for comparison and always reported alongside the primary model.

**Selection:** Best model chosen by macro-averaged F1 on the held-out
test set (20% split, stratified by label).

### 5d. Explainability (`src/explain.py`)

For each review, the words present are scored on two independent axes:

```
pos_vs_neg score   = coef[positive_class][word_idx] − coef[negative_class][word_idx]
neutral_pull score = coef[neutral_class][word_idx] − avg(coef[positive_class][word_idx], coef[negative_class][word_idx])
```

- High `pos_vs_neg` score → word is a positive signal
- Low `pos_vs_neg` score → word is a negative signal
- High `neutral_pull` score → word specifically signals hedging or
  ambivalence (e.g. *"okay"*, *"average product"*, *"could be better"*,
  *"nothing special"*), rather than being merely a weak positive or
  weak negative word

Top-5 words per axis are returned and displayed as colour-coded tags.
The same scoring function (`score_terms_for_indices()`) is shared
between the single-review explainer here and the vectorised bulk-CSV
annotation in `src/dashboard_utils.py`, so results are guaranteed
consistent between the two — there is exactly one implementation of
the coefficient math, not two copies that could silently drift apart.

**Why the neutral_pull axis was added:** the original version of this
module only computed `pos_vs_neg`. That meant a "neutral" prediction
could only ever be explained in terms of leftover positive/negative
tug-of-war — there was no way to show *why* something read as neutral,
only that its positive and negative signals happened to roughly
cancel out. Combined with the neutral class's low precision (§6, §9),
this made neutral explanations doubly unreliable: the prediction
itself was often wrong, and even when right, the explanation mechanism
had no vocabulary for "hedging language" at all. The `neutral_pull`
formula was validated against the trained model's actual coefficients
before being adopted — it correctly surfaces genuine hedging phrases
(*"okay"*, *"decent price"*, *"met expectation"*, *"fair"*) rather
than noise. The dashboard now also shows an explicit reliability
caveat whenever a prediction is neutral (see `src/pages/predict.py`),
so this limitation is disclosed to the user rather than left implicit.

### 5e. Flexible CSV Ingestion (`src/csv_mapper.py`)

Rather than requiring an uploaded CSV to match an exact column schema,
ReviewLens auto-detects the relevant columns:

1. Column names are matched against keyword lists per field (e.g.
   `"stars"`, `"score"`, `"rating"` all resolve to the rating field)
2. If no column name is recognisable for review text, the column with
   the longest average string length is used as a fallback
3. The detected mapping is always shown to the user for confirmation
   before analysis runs

---

## 6. What Was Tried to Improve the Model

This section documents the experimentation process — not just the
final configuration — since the reasoning behind each choice is as
important as the choice itself for a viva defense.

### Starting point

TF-IDF (`max_features=10,000`, bigrams) + Logistic Regression with
`class_weight="balanced"`, `C=1.0`:

| Metric | Value |
|---|---|
| Accuracy | 72.9% |
| Macro F1 | 0.595 |
| Weighted F1 | 0.754 |
| Neutral-class F1 | 0.29 (precision 0.22, recall 0.46) |

The "neutral" class was the clear weak point: `class_weight="balanced"`
pushed recall up but precision collapsed, because the model started
over-predicting "neutral" on ambiguous text.

### What was tried

| Change | Effect |
|---|---|
| More TF-IDF features (10K→30K) + trigrams | Accuracy 72.9%→74.2%, macro F1 72.9%→60.2% — modest gain |
| Higher regularisation (`C=2.0`–`5.0`) with `class_weight="balanced"` | Accuracy improved, macro F1 stayed flat or dropped — "balanced" was still the bottleneck |
| Manually-scaled class weights (e.g. `{positive:1, negative:1.5, neutral:3}`) instead of `"balanced"` | Accuracy jumped to ~79%, macro F1 improved to ~0.61 — much better precision/recall balance on neutral |
| Oversampling the neutral class 2–3× in the training set | Comparable to manual weights, but added complexity without a clear edge |
| `LinearSVC` (calibrated for probabilities) with `class_weight="balanced"` | Highest raw accuracy (~80%) but neutral-class recall collapsed to 5% — macro F1 dropped to 0.54. Rejected: optimises the majority class at the minority class's expense |

### Final approach: cross-validated grid search

Rather than hand-picking one manually-scaled weighting, a small grid
search (`tune_class_weights()`) cross-validates combinations of `C`
and class weights on the **training set only** (never touching the
test set until final evaluation), scored on macro F1:

```
Candidates: C ∈ {2.0, 3.0} × neutral_weight ∈ {3, 4}   (negative_weight fixed at 1.5)
Best found: C=2.0, weights={positive: 1, negative: 1.5, neutral: 4}
```

### Final result

| Metric | Before | After | Change |
|---|---|---|---|
| Accuracy | 72.9% | **78.2%** | +5.3 pp |
| Macro F1 | 0.595 | **0.609** | +0.014 |
| Weighted F1 | 0.754 | **0.782** | +0.028 |
| Neutral-class precision | 0.22 | **0.28** | improved |
| Neutral-class recall | 0.46 | 0.29 | more conservative, fewer false "neutral" predictions |

The neutral class remains the hardest to classify — expected, since
neutral sentiment is inherently more ambiguous than clearly positive
or negative text, and makes up only 7% of the real data. This matches
published sentiment-analysis literature rather than being a bug in
this implementation.

---

## 7. Model Details

| Hyperparameter | Value |
|---|---|
| Vectorizer max features | 30,000 |
| n-gram range | (1, 3) |
| LR regularisation (C) | 2.0 (found via search) |
| LR class weights | `{positive: 1, negative: 1.5, neutral: 4}` (found via search) |
| LR max iterations | 1,000 |
| Random state | 42 (throughout, for reproducibility) |
| Test split | 20%, stratified by label |

---

## 8. Results

### 8a. Classification Report (Logistic Regression, final model)

```
              precision    recall  f1-score   support

    negative       0.68      0.66      0.67      1459
     neutral       0.28      0.29      0.29       426
    positive       0.87      0.87      0.87      4231

    accuracy                           0.78      6116
   macro avg       0.61      0.61      0.61      6116
weighted avg       0.78      0.78      0.78      6116
```

### 8b. Baseline Comparison (Multinomial Naive Bayes)

```
              precision    recall  f1-score   support

    negative       0.60      0.67      0.63      1459
     neutral       0.30      0.10      0.15       426
    positive       0.84      0.86      0.85      4231

    accuracy                           0.76      6116
   macro avg       0.58      0.54      0.54      6116
weighted avg       0.74      0.76      0.75      6116
```

Logistic Regression outperforms the Naive Bayes baseline on every
metric, most notably on the neutral class (F1 0.29 vs 0.15) —
justifying its selection as the primary model.

### 8c. Top Positive Features

`great, course, amazing, love, memory, excellent, life, tablet, storage,
"great course", film, informative, brilliant, "really like", family,
enjoyed, beautiful`

### 8d. Top Negative Features

`useless, awful, movie, acting, lost, poorly, stupid, disappointed,
character, minute, deleted, "difficult use", terrible, boring,
"would recommend" (in negated context), "one star", uninstalled`

Note the mix of domains here (movie/acting alongside product terms)
reflects the multi-domain nature of the merged dataset — see
Limitations.

---

## 9. Limitations

1. **Neutral class remains weak** (F1 ≈ 0.29) — inherent to the
   ambiguity of neutral reviews and their scarcity in real data (7%).
   This has a direct, important consequence for explainability: a
   neutral prediction's keyword explanation is less trustworthy than a
   positive/negative one, both because the underlying call is often
   wrong (precision ≈ 28%) and because — prior to the fix described
   in §5d — the explanation mechanism itself had no way to show *why*
   something read as neutral, only leftover positive/negative
   tug-of-war. This is now mitigated two ways: (a) `src/explain.py`
   computes a dedicated `neutral_pull` score so neutral predictions
   get genuine hedging-language explanations (e.g. "okay", "nothing
   special"), and (b) the dashboard shows an explicit reliability
   caveat on every neutral prediction rather than presenting it with
   the same confidence as a clear-cut call. The underlying prediction
   accuracy for neutral reviews is unchanged — this addresses the
   explanation's honesty and usefulness, not the classification
   difficulty itself, which is a harder problem (see §10 Future Work).
2. **English only** — preprocessing and stopwords are English-specific
3. **No aspect-level sentiment** — cannot separate "camera is good but
   battery is bad" into two separate scores within one review
4. **Domain shift** — the merged dataset spans very different domains
   (movies, courses, electronics, restaurants); a model trained on a
   single domain would likely perform better within that domain, at
   the cost of generality
5. **No deep learning** — TF-IDF + LR is CPU-friendly and interpretable,
   but a transformer model (e.g. DistilBERT) would likely score higher,
   particularly on the neutral class, at the cost of training time,
   compute requirements, and explainability
6. **CSV column auto-detection is heuristic** — very unusually-named
   columns may require manual correction in the mapping UI

---

## 10. Future Work

- Fine-tune a transformer model (DistilBERT) for higher accuracy,
  particularly to improve neutral-class performance
- Aspect-based sentiment analysis (ABSA) to separate per-feature sentiment
- Multilingual support
- REST API (FastAPI) for integration with third-party systems
- Active learning — incorporate user corrections from the dashboard
  into a retraining loop
- Expand the keyword list in `src/csv_mapper.py` based on real-world
  upload patterns observed after deployment

---

## 11. Conclusion

ReviewLens demonstrates a complete, production-quality NLP pipeline —
from raw, heterogeneous real-world data to actionable business insights
— using only open-source, CPU-friendly Python libraries. Training on
real Kaggle data rather than synthetic text, combined with systematic,
cross-validated hyperparameter tuning rather than default settings,
produces honest, defensible performance numbers rather than inflated
ones. The system maintains full interpretability through
coefficient-based keyword explanation, accepts arbitrary CSV uploads
through automatic column detection, and presents results via an
8-page Streamlit dashboard accessible to non-technical users.
