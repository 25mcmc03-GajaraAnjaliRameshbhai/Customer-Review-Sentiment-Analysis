# ReviewLens — Architecture Document

## 1. High-Level Overview

```
13 Raw Kaggle CSVs (different schemas)
        │
        ▼
┌──────────────────┐
│  data_sources.py │  Config: which column = text/rating/sentiment
│  data_loader.py  │  per source; merges + dedupes + shuffles
└────────┬─────────┘
         │  unified schema
         ▼
   data/raw/reviews.csv   (~30,800 real reviews)
         │
         ▼
┌──────────────────┐
│  Preprocessing   │  src/preprocess.py
│  Pipeline        │  lowercase → strip HTML/URLs → remove
│                  │  special chars → stopwords → lemmatize
└────────┬─────────┘
         │  clean text
         ▼
┌──────────────────┐
│  Hyperparameter  │  src/train_model.py :: tune_class_weights()
│  Search          │  cross-validated grid search over C and
│                  │  per-class weights (macro F1 scoring)
└────────┬─────────┘
         │  best (C, class_weight)
         ▼
┌──────────────────┐
│  TF-IDF          │  sklearn TfidfVectorizer
│  Vectorizer      │  max_features=30,000  ngram=(1,3)
│                  │  sublinear_tf=True    min_df=2
└────────┬─────────┘
         │  sparse feature matrix
         ▼
┌──────────────────┐
│  Classifier      │  LogisticRegression (primary, tuned)
│                  │  ─────────────────────────────
│                  │  MultinomialNB (baseline)
└────────┬─────────┘
         │  label + probabilities
         ▼
┌──────────────────┐
│  Explainability  │  src/explain.py
│                  │  LR coef[positive] − coef[negative]
│                  │  per word present in the review
└────────┬─────────┘
         │  positive_terms / negative_terms
         ▼
┌──────────────────┐
│  Streamlit App   │  app.py → src/pages/*.py (8 pages)
│  Dashboard       │  Plotly charts + Matplotlib word clouds
└──────────────────┘
```

---

## 2. Data Flow

### 2a. Building the Real Dataset (one-time / on-demand)

```
prepare_dataset.py
        │
        ▼
src/data_loader.py :: build_unified_dataset()
  For each of 13 SourceConfig entries in src/data_sources.py:
    1. Read the raw Kaggle CSV (with a UTF-8 → latin-1 encoding
       fallback, since some exports aren't strictly UTF-8)
    2. Map its text/rating/sentiment column to the unified schema
       using one of three label-mapping functions:
         _rating_to_label()    — 1-2★→negative, 3★→neutral, 4-5★→positive
         _binary_to_label()    — 0→negative, 1→positive
         _sentiment_to_label() — normalise an existing text label
    3. Drop rows with missing/empty text or unmapped labels
    4. Cap the source at its configured sample_size (keeps one huge
       source, e.g. the 205K-row Flipkart dataset, from dominating)
  Concatenate all 13 → drop duplicate review_text → shuffle
        │
        ▼
data/raw/reviews.csv   (ships with the repo — ~30,800 rows, 11MB)
```

Why this design: every raw file has a *different* schema, but the
loader logic itself never needs to know that — `SourceConfig` entries
are pure data, so adding a 14th source is a one-line addition to
`src/data_sources.py`, not a code change.

### 2b. Training Flow

```
train.py
  1. load_dataset()                  # data/raw/reviews.csv
  2. preprocess_batch()              # src/preprocess.py
  3. train_test_split(stratify=label, test_size=0.2, random_state=42)
  4. tune_class_weights(X_train, y_train)     # src/train_model.py
       - cross-validated grid search (StratifiedKFold, 2-fold)
       - candidates: C ∈ {2.0, 3.0} × neutral_weight ∈ {3, 4}
         (negative_weight fixed at 1.5 — see §4 for why)
       - scored on macro F1 (fair to all 3 classes, not just the
         majority "positive" class)
       - returns the winning (C, class_weight)
  5. build_lr_pipeline(C, class_weight) → fit on full training set
  6. build_nb_pipeline() → fit on full training set (baseline)
  7. evaluate_pipeline() for both → select by macro F1
  8. joblib.dump(pipeline)           → models/pipeline.pkl
  9. save_metadata()                 → models/metadata.json
     (includes metrics, confusion matrix, top features, and the
      winning hyperparameters, for full transparency)
```

### 2c. Prediction Flow — Single Review

```
User types text on the "Predict Sentiment" page
        │
        ▼
preprocess_text(raw_text)     # lowercase, clean, lemmatize
        │
        ▼
pipeline.predict([clean])     # TF-IDF → LR → label
pipeline.predict_proba([clean])  # → [p_neg, p_neu, p_pos]
        │
        ▼
explain_prediction(clean, pipeline)   # src/explain.py
  - vectorizer.transform([clean])   # get TF-IDF scores
  - for each non-zero feature:
      score = coef[pos_class] − coef[neg_class]
  - sort → positive_terms (score > 0)
          → negative_terms (score < 0)
        │
        ▼
Display: label, confidence, prob bar chart, keyword tags, explanation
```

### 2d. Prediction Flow — Bulk CSV (any schema)

```
User uploads a CSV on the "Review Explorer" page
        │
        ▼
src/csv_mapper.py :: detect_columns(df)
  - keyword-match each column name against known patterns per field
    (e.g. "stars"/"score"/"rating" all → rating)
  - if no column name matches review_text, fall back to the
    string column with the longest average length
        │
        ▼
User confirms / corrects the suggested mapping in 5 dropdowns
        │
        ▼
src/csv_mapper.py :: apply_mapping(df, mapping)
  → DataFrame with standard columns (unmapped optional fields omitted)
        │
        ▼
src/dashboard_utils.py :: predict_batch(df, pipeline)
  - preprocess_batch(all_texts)
  - pipeline.predict(clean_texts)        # → array of labels
  - pipeline.predict_proba(clean_texts)  # → (N, 3) proba matrix
  - per-row term extraction using the same coefficient-difference
    trick as explain_prediction, vectorised across all rows at once
    for speed
        │
        ▼
Annotated DataFrame becomes the dashboard's "active dataset"
(stored in st.session_state) — every other page reads from it:
        │
        ├──► Overview        (snapshot + model info)
        ├──► Trends & Charts (distribution, keywords, word clouds, time trend)
        ├──► Product Comparison (category / source breakdowns)
        ├──► Summary         (auto-generated plain-language summary)
        ├──► Model Performance (accuracy / F1 / confusion matrix)
        └──► Export Reports  (download annotated CSV + text report)
```

---

## 3. Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/data_sources.py` | Config: which raw Kaggle CSV maps to which schema |
| `src/data_loader.py` | Generic loader that merges all configured sources |
| `prepare_dataset.py` | CLI entry point that runs the loader and writes `data/raw/reviews.csv` |
| `src/preprocess.py` | Text normalisation pipeline |
| `src/features.py` | TF-IDF vectorizer factory and persistence |
| `src/train_model.py` | Pipeline factories, hyperparameter search, evaluation, top-features |
| `src/evaluate.py` | Standalone metrics helpers (accuracy, F1, confusion matrix) |
| `src/explain.py` | Coefficient-based keyword explainability |
| `src/csv_mapper.py` | Auto-detects columns in any user-uploaded CSV |
| `src/utils.py` | Path constants, metadata I/O, display helpers |
| `src/dashboard_utils.py` | Shared prediction/chart helpers used by every dashboard page |
| `train.py` | Orchestrates end-to-end training |
| `app.py` | Streamlit entry point — thin router to `src/pages/*.py` |
| `src/pages/*.py` | One file per dashboard sidebar item (8 pages) |
| `tests/` | pytest unit, integration, and Streamlit `AppTest` dashboard tests |

---

## 4. Design Decisions

### Why TF-IDF + Logistic Regression?

- **Interpretable** — coefficients directly map to word importance
- **Fast** — trains (including hyperparameter search) in under a
  minute on CPU, no GPU needed
- **Sparse-friendly** — TF-IDF produces sparse matrices; LR handles them well
- **Suitable for viva** — every step can be explained without hand-waving

### Why not scikit-learn's `class_weight="balanced"`?

It was the starting point, but on this real, imbalanced dataset
(~69% positive, ~24% negative, ~7% neutral) it over-corrects: the
"neutral" class's recall goes up but its precision collapses, because
the model starts predicting "neutral" far too often. A small
cross-validated grid search over manually-scaled class weights
(`tune_class_weights()` in `src/train_model.py`) found a better
precision/recall trade-off — see `docs/PROJECT_REPORT.md` for the
full before/after comparison table.

### Why cross-validate the search on the *training* set only?

To avoid tuning hyperparameters against the same test set used for
the final reported metrics, which would make the reported accuracy
optimistic. The grid search uses `StratifiedKFold` cross-validation
strictly within `X_train`/`y_train`; the held-out test set is only
touched once, for final evaluation.

### Why not BERT / transformers?

- Requires significant GPU or is very slow on CPU
- Harder to explain end-to-end in a viva
- Overkill for a capstone scope — see Limitations in `README.md`

### Why sklearn Pipeline?

- Single object to save/load (one `joblib.dump`)
- Vectorizer and classifier always stay in sync
- Prevents train/test leakage (vectorizer only sees training data)

### Explainability approach

Uses LR coefficients on two independent axes per word:

```
pos_vs_neg score   = coef[positive_class][word_idx] − coef[negative_class][word_idx]
neutral_pull score = coef[neutral_class][word_idx] − avg(coef[positive_class][word_idx], coef[negative_class][word_idx])
```

High `pos_vs_neg` → word pushes toward **positive**; low → **negative**.
High `neutral_pull` → word specifically signals hedging/ambivalence
(e.g. "okay", "nothing special"), which the `pos_vs_neg` axis alone
cannot express — a word can score near-zero on `pos_vs_neg` either
because it's genuinely neutral language, or just because it's
irrelevant to sentiment entirely. The `neutral_pull` axis distinguishes
these two cases.

This is lightweight, needs no external library (no SHAP), and the
exact same scoring function — `score_terms_for_indices()` in
`src/explain.py` — is called by both the single-review explainer and
the vectorised bulk version (`src/dashboard_utils.py :: predict_batch`),
so there is one implementation of the coefficient math, not two that
could silently drift apart.

**Known limitation:** explanation reliability is tied to the
underlying class's prediction accuracy. Positive/negative explanations
are reliable (those classes have F1 0.87 / 0.67). Neutral explanations
are inherently weaker — not just because the neutral_pull axis is a
secondary signal, but because a "neutral" prediction itself is often
wrong (precision ≈ 28%, see docs/PROJECT_REPORT.md §6). The dashboard
discloses this directly with a caveat on every neutral prediction
(`src/pages/predict.py`) rather than leaving it implicit.

### Why a config-driven column mapper instead of a fixed CSV schema?

Real-world CSVs never agree on column names. Hard-coding
`review_text`/`rating`/`date`/`source`/`category` as required exact
names would make the dashboard fail on any file the user didn't
specifically prepare for it. `src/csv_mapper.py` instead:

1. Tries keyword matching first (fast, usually right)
2. Falls back to a content-based heuristic (longest average string
   length) if no column name is recognisable
3. Always shows its guess to the user before running anything, so a
   wrong guess costs one dropdown click, not a failed upload

---

## 5. Scalability Notes

- For >100K reviews, batch the `predict_batch` loop in chunks of 5,000
- The TF-IDF vectorizer vocabulary is capped at 30,000 features — adjust `max_features` if needed
- The hyperparameter search in `tune_class_weights()` deliberately
  uses a smaller/faster TF-IDF config than the final model — only the
  winning (C, weights) combination is reused with the full-sized
  vectorizer, keeping search time low regardless of dataset size
- To add a new language: swap NLTK stopwords and lemmatizer for that language
- To add a new Kaggle source: add one `SourceConfig` entry to
  `src/data_sources.py` — no other code changes needed
