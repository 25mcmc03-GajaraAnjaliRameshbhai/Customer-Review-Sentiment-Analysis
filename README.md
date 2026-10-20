# 🔍 Customer Review Sentiment Analysis Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
[![Streamlit](https://customer-review-sentiment-analysis-anjali.streamlit.app/)

> A machine learning project for classifying real customer reviews as **positive, negative, or neutral** using supervised learning techniques. The project merges **13 real Kaggle datasets**, handles imbalanced sentiment classes, compares multiple models, and selects the best-performing model based on **Macro F1-score**.

---

## 📑 Table of Contents

- [📖 Project Overview](#-project-overview)
- [📌 Project Workflow](#-project-workflow)
- [🌍 Real-World Use Cases](#-real-world-use-cases)
- [📂 Dataset](#-dataset)
- [🛠 Technologies Used](#-technologies-used)
- [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [⚙️ Data Preprocessing](#️-data-preprocessing)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📈 Model Evaluation](#-model-evaluation)
- [📊 Test Set Performance](#-test-set-performance)
- [📷 Results](#-results)
- [💡 Key Findings](#-key-findings)
- [⚠️ Limitations](#️-limitations)
- [📁 Repository Structure](#-repository-structure)
- [📓 Installation](#-installation-notebook)
- [🚀 Try it yourself](#-try-it-yourself-web-app)
- [☁️ Deployment](#️-deployment)
- [🧪 Running Tests](#-running-tests)
- [🔮 Future Improvements](#-future-improvements)
- [🎯 Conclusion](#-conclusion)
- [👨‍💻 Author](#-author)

---

## 📊 Project Overview

Businesses receive thousands of product and service reviews every day across multiple platforms. Reading and categorising them manually is impossible at scale. This project automates sentiment classification and surfaces actionable insights: what customers love, what they hate, and why.

This project develops an end-to-end sentiment analysis pipeline using Python and Scikit-learn. It follows the complete machine learning workflow, from data merging and preprocessing to model training, evaluation, explainability, and deployment as an interactive dashboard.

Since real review data is **noisy and imbalanced** (neutral reviews make up only about **7%** of the dataset), plain accuracy can be misleading. Therefore, this project focuses on **Macro F1-score** alongside accuracy and per-class precision/recall to evaluate model performance.

---

## 📌 Project Workflow

```
Data Collection (13 Kaggle datasets)
        │
        ▼
Schema Unification & Merging
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Text Preprocessing
        │
        ▼
Machine Learning Pipelines
        │
        ▼
Hyperparameter Search (Cross Validation)
        │
        ▼
Model Selection
        │
        ▼
Test Set Evaluation & Explainability
        │
        ▼
Streamlit Dashboard Deployment
```

---

## 🌍 Real-World Use Cases

| Industry | Application |
|---|---|
| E-commerce | Classify product reviews from Amazon / Flipkart |
| EdTech | Analyse course feedback to improve content |
| Mobile / SaaS | Detect negative app-store reviews before they go viral |
| Restaurants | Monitor sentiment across Zomato / survey channels |

---

## 📂 Dataset

The project uses **~30,800 real customer reviews**, merged from **13 Kaggle datasets** covering 6 categories and 8 platforms.

| Property | Value |
|----------|-------|
| Total Reviews | **~30,800** |
| Source Datasets | **13** |
| Categories | **6** |
| Platforms | **8** |
| Target | **Sentiment** (Positive / Negative / Neutral) |
| Neutral Share | **~7%** |

### Categories and Platforms

| Category | Platforms |
|---|---|
| Electronics | Amazon |
| App | App Store, Google Play Store |
| E-commerce | Flipkart, generic e-commerce sites |
| Restaurant | Zomato, customer surveys |
| Course | Coursera |
| Movie | IMDB |

### Unified Schema

Every source file has a different schema (different column names, numeric ratings, star-text, or binary "liked" flags). `src/data_sources.py` maps each one onto a single unified schema, and `src/data_loader.py` merges them. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline.

The processed dataset (`data/raw/reviews.csv`, ~11 MB) ships with this repo, so `train.py` works immediately. The much larger raw Kaggle CSVs (~250 MB) are not included.

### Regenerating the Dataset (optional)

1. Download the source datasets (file names are listed in `src/data_sources.py`)
2. Place them in `data/kaggle_raw/`
3. Run:
   ```bash
   python prepare_dataset.py
   ```
4. This rewrites `data/raw/reviews.csv`; re-run `python train.py` afterward.

To add a new source, add one `SourceConfig` entry to `src/data_sources.py`. No other code changes are needed.

---

## 🛠 Technologies Used

- Python 3.10+
- scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
- NLTK (stopwords, lemmatization)
- Pandas
- Plotly
- Matplotlib
- WordCloud
- Streamlit
- Joblib
- Pytest + Streamlit AppTest
- Jupyter Notebook

---

## 🔍 Exploratory Data Analysis (EDA)

The exploratory analysis includes:

- Dataset overview and merged-source summary
- Sentiment class distribution (imbalance analysis)
- Category and platform distribution
- Top keywords per sentiment
- Word clouds
- Review length distribution
- Rating-vs-sentiment comparison

The full EDA is available in `docs/Customer_Review_Sentiment_Analysis_Notebook.ipynb`.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied before model training:

- Schema unification across 13 sources
- Lowercasing
- HTML and URL removal
- Stopword removal
- Lemmatization
- Stratified train-test split (80/20)
- TF-IDF vectorization
- Pipeline implementation so the vectorizer and classifier are trained and saved together

---

## 🤖 Machine Learning Models

Two supervised learning algorithms were evaluated:

- Multinomial Naive Bayes (baseline)
- Logistic Regression (hyperparameter-tuned)

Each model is implemented as a Scikit-learn **Pipeline**, consisting of:

```
Text Preprocessing
        │
        ▼
TF-IDF Vectorizer
        │
        ▼
Classifier
```

---

## 📈 Model Evaluation

Logistic Regression was tuned across a small hyperparameter grid using **cross-validation on Macro F1**, and compared against the Naive Bayes baseline.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score (Macro and Weighted)
- Confusion Matrix

> Since the dataset is imbalanced and the neutral class is small, **Macro F1** was selected as the primary metric for model comparison.

### Best Model

**Logistic Regression + TF-IDF**

---

## 📊 Test Set Performance

| Metric | Score |
|---------|-------|
| Accuracy | **78.2%** |
| Macro F1 | **0.609** |
| Weighted F1 | **0.782** |
| Training / Test Samples | **24,464 / 6,116** |
| Best Hyperparameters | `C = 2.0`, class weights: positive 1, negative 1.5, neutral 4 |

These are honest numbers from real, noisy, imbalanced review data, not inflated synthetic-data numbers. See `models/metadata.json` for the exact results from your own training run, and [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the full breakdown of what was tried to improve them.

> The **neutral** class remains the hardest to classify (precision ≈ 28%). This matches published sentiment-analysis literature: neutral reviews are inherently ambiguous and form a small minority of real review data.

---

## 📷 Results

## Model Evaluation

The final evaluation includes:

- Confusion Matrix
- Per-class Precision / Recall / F1
- Top influential words (coefficient-based explainability)
- Tuned hyperparameters

### Example Single Review

**Input:**
```
The phone camera is excellent, but battery backup is very poor.
```

**Output:**
```
Sentiment  : Positive
Positive   : excellent, backup, battery backup, phone camera, camera, battery
Negative   : poor, phone
```

The model identifies both signals, with "excellent" pulling toward positive and "poor" pulling toward negative. This is why mixed reviews like this get a more moderate confidence score than a clearly one-sided review.

---

## 💡 Key Findings

- The dataset is imbalanced, with neutral reviews representing only about **7%** of all observations.
- Logistic Regression with TF-IDF achieved **78.2% accuracy** and a **Macro F1 of 0.609** on real, noisy data.
- Accuracy alone hides weak neutral-class performance, so **Macro F1** proved more informative.
- Neutral reviews are inherently ambiguous; the dashboard shows an explicit warning whenever a prediction is neutral and surfaces hedging-language keywords (e.g. "okay", "nothing special").
- Merging 13 sources across 6 categories gives broad coverage, but introduces domain shift.
- Coefficient-based explainability keeps the model transparent and interpretable.

---

## ⚠️ Limitations

1. **English only**: preprocessing and stopwords are English-specific.
2. **Neutral class is weak** (precision ≈ 28%): inherently ambiguous in real review data.
3. **No deep learning**: TF-IDF + LR is CPU-friendly and interpretable, but a transformer (BERT etc.) would likely score higher.
4. **No aspect-level sentiment**: it cannot separate "camera good, battery bad" into two scores within one review.
5. **Domain shift**: the merged dataset spans very different domains (movies, courses, electronics); a model fine-tuned on one domain would likely perform better within it.
6. **Column auto-detection is heuristic**: unusually named columns may need manual correction in the mapping dropdown.

---

## 📁 Repository Structure

```text
Customer-Review-Sentiment-Analysis/
├── app.py                    # Streamlit entry point (thin router)
├── train.py                  # Model training script
├── prepare_dataset.py        # Builds data/raw/reviews.csv from Kaggle sources
├── build_notebook.py         # Generates docs/Customer_Review_Sentiment_Analysis_Notebook.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml           # Theme + server settings used in deployment
├── .devcontainer/
│   └── devcontainer.json     # GitHub Codespaces config (Python 3.11)
│
├── data/
│   ├── raw/
│   │   └── reviews.csv       # Unified real dataset (ships with repo)
│   ├── processed/
│   └── kaggle_raw/           # Raw Kaggle CSVs (not shipped)
│
├── models/
│   ├── pipeline.pkl          # Trained sklearn Pipeline
│   └── metadata.json         # Metrics, confusion matrix, top features
│
├── src/
│   ├── preprocess.py         # Text cleaning pipeline
│   ├── features.py           # TF-IDF utilities
│   ├── train_model.py        # Pipeline factories + evaluation
│   ├── evaluate.py           # Metrics helpers
│   ├── explain.py            # Coefficient-based explainability
│   ├── utils.py              # Paths, constants, helpers
│   ├── data_sources.py       # Which Kaggle CSV maps to which schema
│   ├── data_loader.py        # Generic loader that merges all sources
│   ├── csv_mapper.py         # Auto-detects columns in any uploaded CSV
│   ├── dashboard_utils.py    # Shared prediction/chart helpers
│   └── pages/                # One file per dashboard page
│       ├── overview.py
│       ├── predict.py
│       ├── explorer.py
│       ├── trends.py
│       ├── comparison.py
│       ├── summary.py
│       ├── performance.py
│       └── export.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_data_loader.py
│   ├── test_csv_mapper.py
│   ├── test_model.py
│   ├── test_app.py
│   └── test_dashboard.py
│
└── docs/
    ├── PROJECT_REPORT.md
    ├── ARCHITECTURE.md
    ├── Customer_Review_Sentiment_Analysis_IEEE_Report.tex
    └── Customer_Review_Sentiment_Analysis_Notebook.ipynb   # Full pipeline walkthrough (EDA → training → explainability)
```

---

## 📓 Installation (Notebook)

Clone the repository
```bash
git clone <repo-url>
```

Navigate to the project
```bash
cd Customer-Review-Sentiment-Analysis
```

Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook
```bash
jupyter notebook docs/Customer_Review_Sentiment_Analysis_Notebook.ipynb
```

The notebook is a full, narrated walkthrough of the pipeline (EDA, preprocessing, hyperparameter search, training, evaluation, explainability) with outputs already embedded. It imports directly from `src/`, so its results match the shipped model.

To regenerate it (e.g. after retraining with different data):
```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace docs/Customer_Review_Sentiment_Analysis_Notebook.ipynb
```

---

## 🚀 Try it yourself (Web App)

An interactive Streamlit dashboard lets you paste a single review or upload **any** CSV of reviews and get sentiment, confidence, and keywords back instantly.

Train the model
```bash
python train.py
```

Launch the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

**[Live demo →](https://customer-review-sentiment-analysis-anjali.streamlit.app/)**

> Full deployment steps are in the [Deployment](#️-deployment) section below.

### Dashboard Pages

| Page | What it does |
|---|---|
| 📊 **Overview** | Snapshot of the active dataset and the trained model |
| 🤖 **Predict Sentiment** | Paste one review → sentiment + confidence + keywords |
| 🔍 **Review Explorer** | Upload any CSV, confirm the auto-detected column mapping, then search/filter/browse |
| 📈 **Trends & Charts** | Sentiment distribution, keywords, word clouds, time trend, confidence distribution, rating-vs-sentiment mismatch detector |
| ⚖️ **Product Comparison** | Compare sentiment across categories and sources |
| 📄 **Summary** | Plain-language auto-summary with top praise/complaints and sentiment trend |
| 🧠 **Model Performance** | Accuracy, F1, hyperparameters, confusion matrix, influential words |
| 📥 **Export Reports** | Download the annotated dataset and a text summary report |

### Flexible CSV Upload

No fixed format is required. `src/csv_mapper.py` auto-detects which columns hold the review text, rating, date, source, and category, and shows every guess in dropdowns you can correct before analysis runs.

```csv
Comments,Stars,Timestamp,Platform
"Great product! Very happy with my purchase.",5,2024-01-15,Amazon
"Terrible quality. Stopped working after a week.",1,2024-02-20,Flipkart
```

This is auto-mapped as `Comments → review_text`, `Stars → rating`, `Timestamp → date`, `Platform → source`. A minimal CSV with only a text column also works.

- Columns are validated against their **actual content** (e.g. a "rating" column must be mostly numeric, a "date" column must parse as dates)
- Clear, specific messages for empty files, header-only files, non-UTF-8 encoding, corrupted CSVs, and files with no detectable text column
- **Upload limits:** max file size **50 MB**, hard cap **150,000 rows**, recommended under **20,000 rows** (larger files are processed in chunks with a progress bar)

---

## ☁️ Deployment

The dashboard is deployed on **Streamlit Community Cloud** straight from this GitHub repository.

**[🌐 Live App →](https://customer-review-sentiment-analysis-anjali.streamlit.app/)**

### Deploy Your Own Copy

1. Push the project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and select your repository and branch.
4. Set **Main file path** to `app.py`.
5. (Optional) Under **Advanced settings**, choose Python 3.11 to match the dev container.
6. Click **Deploy**. Streamlit installs everything from `requirements.txt` and applies the theme in `.streamlit/config.toml`.

### How the Deployment Is Set Up

| Concern | How it is handled |
|---|---|
| **Entry point** | `app.py` (thin router that loads the model and dispatches to `src/pages/`) |
| **Dependencies** | `requirements.txt` |
| **Theme & server** | `.streamlit/config.toml` (light theme, headless server, usage stats off) |
| **Trained model** | `models/pipeline.pkl` (~1.9 MB) is committed to the repo, so the app starts with the model already available |
| **Missing model** | If `pipeline.pkl` is absent, the app auto-trains once on first launch from `data/raw/reviews.csv` (about 1-3 minutes), then continues normally |
| **Training data** | `data/raw/reviews.csv` (~11 MB) ships with the repo |
| **NLTK data** | Downloaded on first run into a repo-local `.nltk_data/` folder, with a built-in stopword fallback if the download is unavailable |
| **Upload limits** | 50 MB file size and 150,000 rows, enforced in the app |
| **Local / cloud dev** | `.devcontainer/devcontainer.json` runs the app in GitHub Codespaces on port 8501 |

> If auto-training cannot run (for example, required data files are missing), run `python prepare_dataset.py` and `python train.py` locally, commit the generated files, and redeploy.

### Deployment Tips

- **Keep `models/pipeline.pkl` and `data/raw/reviews.csv` in Git.** `.gitignore` already has exceptions for them, so just commit them.
- **First load can be slow** on the free tier because the app may need to wake up and download NLTK data.
- **Updating the app:** push a commit to the deployed branch and Streamlit Cloud redeploys automatically.
- **Retraining:** run `python train.py` locally, then commit the new `models/pipeline.pkl` and `models/metadata.json`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

**139 automated tests** (unit + dashboard integration):

- `test_preprocess.py`: text cleaning pipeline
- `test_data_loader.py`: label-mapping logic and source config checks
- `test_csv_mapper.py`: flexible column detection for CSV upload
- `test_model.py`: trained model correctness (skipped if not trained)
- `test_app.py`: module/page import checks
- `test_dashboard.py`: runs the Streamlit app and every page using `AppTest`

---

## 🔮 Future Improvements

Possible extensions include:

- Fine-tune a transformer model (DistilBERT) for higher accuracy
- Aspect-based sentiment analysis (separate scores per product feature)
- Multi-language support
- REST API (FastAPI) for integration with other systems
- Active learning: feed user corrections back into retraining

---

## 🎯 Conclusion

This project demonstrates a complete machine learning workflow for classifying customer review sentiment. Through merging 13 real-world datasets, careful text preprocessing, and cross-validated hyperparameter tuning, a TF-IDF + Logistic Regression model was selected over a Naive Bayes baseline using Macro F1 as the primary metric.

The final model achieved **78.2% accuracy**, a **Macro F1 of 0.609**, and a **Weighted F1 of 0.782** on real, noisy, imbalanced data. The neutral class remains the main challenge, which is consistent with published sentiment-analysis research.

To make the model accessible beyond the notebook, the trained pipeline was deployed as an **interactive Streamlit dashboard** with flexible CSV upload, trend analysis, explainability, and exportable reports.

Overall, this project highlights the importance of using real data, choosing metrics suited to imbalanced classification, and keeping models interpretable.

---

## 👨‍💻 Author

**GAJARA ANJALI RAMESHBHAI**

Machine Learning • Data Science • Python

> ⭐ If you found this project useful, consider giving it a star!
