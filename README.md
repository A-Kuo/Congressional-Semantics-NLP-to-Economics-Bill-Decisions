# Congressional Semantics: NLP to Economics — Bill Pass/Fail Prediction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor&style=plastic&logoColor=blue)](https://www.python.org/downloads/release/python-3120/)
[![App](https://img.shields.io/badge/Fullstack-Ubuntu-orange?logo=Ubuntu&style=plastic)](https://ubuntu.com/ai/data-science)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.9-orange?logo=scikit-learn&style=plastic)](https://scikit-learn.org/stable/whats_new/v1.9.html)

**Research Question:** Can the language and tone of U.S. Congressional floor speeches on economic legislation predict whether the associated bill passes or fails?

## Project Overview

This project combines NLP (TF-IDF, BERT) with econometric and machine learning methods to predict bill outcomes from congressional speech text. We use speeches from the 110th–114th Congress (2007–2016) for bills with economic subjects (taxation, labor, trade, budget).

**Type:** Predictive (not causal)  
**Y (outcome):** Bill pass/fail (binary: 1 = enacted into law, 0 = failed/died)  
**X (features):** TF-IDF word frequencies from aggregated floor speeches  
**Scope:** 1,133 bills (110th–114th Congress, 2007–2016); economic legislation only; 12.5% pass rate

## Repository Structure

```
congressional-nlp-econ/
├── README.md
├── data/
│   ├── raw/                  # Raw downloaded data (do not modify)
│   │   ├── speeches/          # Stanford Congressional Record (speech_id|text)
│   │   ├── descriptions/      # Stanford speech metadata
│   │   ├── speakermap/        # speech_id -> speaker/party
│   │   ├── bills_cache/        # Cached Congress.gov API responses
│   │   └── speech_links/       # speech_id -> bill_id links (regex-matched)
│   ├── processed/            # Cleaned, merged datasets
│   └── README_data.md        # Data sources and download instructions
├── scripts/
│   ├── fetch_real_bills.py           # Congress.gov: real bill metadata + outcomes
│   ├── link_speeches_to_bills.py     # Regex-link raw speeches to real bills
│   └── build_merged_dataset.py       # Assemble final bill-level dataset
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_preprocessing.ipynb
│   ├── 03_tfidf_baseline_enhanced.ipynb
│   ├── 04_lasso_ridge.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_bert_classifier.ipynb      # Not used in final report (see Methods)
│   └── 07_results_comparison.ipynb
├── src/
│   ├── data_utils.py         # Data loading, cleaning, merging
│   ├── nlp_utils.py          # Tokenization, TF-IDF, preprocessing
│   ├── model_utils.py        # Model training, CV, evaluation helpers
│   ├── viz_utils.py          # All plotting functions
│   └── synthetic_data.py     # Synthetic data generator (testing only, --synthetic flag)
├── results/
│   ├── tables/               # CSV tables of model performance
│   └── figures/              # PNG plots (confusion matrices, ROC, etc.)
├── run_pipeline.py           # End-to-end runner (default: real data)
└── requirements.txt
```

**Real-data pipeline order:** `fetch_real_bills.py` → `link_speeches_to_bills.py` →
`build_merged_dataset.py` → `run_pipeline.py`. See `data/README_data.md` for details
on each stage.

## Data Sources

### 1. Congressional Speeches
- **Stanford Congressional Record Dataset:** https://data.stanford.edu/congress_text  
  Parsed speeches (43rd–114th Congress) with speaker metadata. Use 110th–114th Congress.

### 2. Bill Outcomes (Labels)
- **Congress.gov API:** https://api.congress.gov  
  Free API key required. Endpoints: `/v3/bill/{congress}/{billType}` for bill metadata and outcomes.  
  Definition: PASSED = latestAction contains "Became Public Law" (Y=1), all others = FAILED (Y=0)

### 3. Bill Types (Economic Filter)
- HR (House bills) and S (Senate bills)  
- Subjects: "Taxation", "Labor and Employment", "Trade", "Economics and Public Finance", "Budget and Appropriations"

## Methods Pipeline (Report: April 2026)

**Primary Analysis (TF-IDF + Classical ML):**
All models use TF-IDF features (5,000 max features, min_df=5, max_df=0.95) with 5-fold stratified cross-validation.

### 1. **Baseline: Logistic Regression**
- Linear benchmark for interpretability  
- Metrics: accuracy, AUC-ROC (mean test AUC: 0.897)

### 2. **Regularization: LASSO + Ridge**
- LASSO: sparse feature selection (mean test AUC: 0.843)
- Ridge: dense shrinkage under correlated features (mean test AUC: 0.889)
- Top-selected procedural terms: *consent*, *suspend*, *unanimous*, *ordered*

### 3. **Ensemble: Random Forest**
- 200 trees, sqrt(5000) features per split, default depth
- Highest cross-validated AUC (0.958) but largest train–test gap (0.107)
- Mean test AUC: 0.958 ± larger generalization gap than linear models

### 4. **Transformer Models (Not in Report)**
- BERT notebook 06 exists for reference but **not used in final analysis**
- Report notes: "Transformer models may be valuable extensions" (Section 5.2)
- Future work: evaluate with temporal splits and calibration metrics

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get Congress.gov API key:
   - Go to https://api.congress.gov
   - Register for a free key
   - Export: `export CONGRESS_API_KEY=<your-key>`

3. Run notebooks in order:
   ```bash
   jupyter notebook notebooks/01_data_collection.ipynb
   jupyter notebook notebooks/02_eda_preprocessing.ipynb
   ...
   ```

## Key Deliverables

| Notebook | Output | Description |
|----------|--------|-------------|
| 01_data_collection | `data/processed/bills_speeches_merged.csv` | Merged bill metadata + speeches |
| 02_eda_preprocessing | `results/figures/class_balance.png` | Exploratory data analysis |
| 03_tfidf_baseline | `results/tables/logistic_cv_scores.csv` | Baseline logistic regression CV |
| 04_lasso_ridge | `results/figures/lasso_coefficients.png` | Top-20 LASSO words |
| 05_random_forest | `results/figures/rf_importance.png` | Top-30 RF feature importances |
| 07_results_comparison | `results/figures/roc_curves.png` + `results/tables/model_comparison.csv` | Final model comparison |

## Coding Standards

All notebooks must include:
```python
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# sklearn models: random_state=SEED
```

## References

- Inspiration: Prof. Harold D. Chiang, UW-Madison Dept. of Economics; *Econometrics for Big Data*
- Data: Stanford Congressional Record, Congress.gov API
- Framework: scikit-learn, Hugging Face transformers

---

> *"Politics is just like show business" —that other actor-turned-US-president*
