# Congressional Semantics: NLP to Economics | Bill Pass/Fail Prediction

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor&style=plastic&logoColor=blue)](https://www.python.org/downloads/release/python-3120/)
[![App](https://img.shields.io/badge/Fullstack-Ubuntu-orange?logo=Ubuntu&style=plastic)](https://ubuntu.com/ai/data-science)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.9-orange?logo=scikit-learn&style=plastic)](https://scikit-learn.org/stable/whats_new/v1.9.html)

**Research Question:** Can the language and tone of U.S. Congressional floor speeches on economic legislation predict whether the associated bill passes or fails?

## Project Overview

This project combines NLP (TF-IDF, BERT) with econometric and machine learning methods to predict bill outcomes from congressional speech text. We use speeches from the 110th–114th Congress (2007–2016) for bills with economic subjects (taxation, labor, trade, budget).

**Type:** Predictive (not causal)  
**Y (outcome):** Bill pass/fail (binary: 1 = enacted into law, 0 = failed/died)  
**X (features):** TF-IDF word frequencies from aggregated floor speeches  
**Scope:** 1,120 bills (110th–114th Congress, 2007–2016); economic legislation only; 6.8% pass rate (76 passed)

This is the real, reproducible sample this repository's pipeline produces end-to-end
(`scripts/fetch_real_bills.py` → `scripts/link_speeches_to_bills.py` →
`scripts/build_merged_dataset.py` → `run_pipeline.py`), verified against Congress.gov
and the Stanford Congressional Record corpus on 2026-09-02. It differs from the
1,133-bill / 12.5%-pass-rate sample described in earlier report drafts, because the
Stanford corpus has no native bill-ID field — bills are linked to speeches here via
regex-matched in-text citations ("H.R. 1234", "S. 815"), a real but different (and
broader/noisier) method than whatever produced the original sample. See
`report/results_real_data.tex` for the full real-data Results section and
`data/README_data.md` for the linkage methodology.

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
│   ├── figures/              # PNG plots (confusion matrices, ROC, etc.)
│   ├── statistical_analysis/ # p-values, CIs, permutation-test results
│   └── checkpoint_pipeline.pkl  # Fitted pipelines (used by app.py's live demo)
├── run_pipeline.py           # End-to-end runner (default: real data)
├── app.py                    # Streamlit dashboard (streamlit run app.py)
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

## Methods Pipeline

**Primary Analysis (TF-IDF + Classical ML):**
All models are TF-IDF -> classifier `sklearn` Pipelines (5,000 max features, min_df=5, max_df=0.95), cross-validated on raw text with 5-fold stratified CV so the vectorizer refits fresh on each training fold (see Known Limitation (Fixed) below). Numbers below are from this repo's real pipeline output (`results/tables/model_comparison.csv`, `results/statistical_analysis/`), run 2026-09-08 on the 1,120-bill real sample described above.

### 1. **Baseline: Logistic Regression**
- Linear benchmark for interpretability
- Mean test AUC: 0.919 (train AUC 0.996 — large train–test gap)

### 2. **Regularization: LASSO + Ridge**
- LASSO: sparse feature selection (mean test AUC: 0.918); only 2 of the top-20
  coefficients are statistically significant at p<0.05 given the small (76-bill)
  positive class — most top-magnitude words are not distinguishable from noise
- Ridge: dense shrinkage under correlated features (mean test AUC: 0.928)
- Top procedural terms still selected: *senate*, *suspend*, *passage*, *house*

### 3. **Ensemble: Random Forest**
- 200 trees, sqrt(5000) features per split, default depth
- Highest cross-validated AUC (0.955); smallest RMSE train–test gap of the four
  models here (unlike the original report, where RF had the largest gap — see
  `report/results_real_data.tex` for discussion)
- Top features: *motion*, *suspend*, *proceed*, *senate*, *consent*, *unanimous*,
  *rules* — closely reproduces the procedural-language finding

### 4. **Transformer Models (Not in Report)**
- BERT notebook 06 exists for reference but **not used in final analysis**
- Report notes: "Transformer models may be valuable extensions" (Section 5.2)
- Future work: evaluate with temporal splits and calibration metrics

### Is This Real Signal? Leakage Audit + Generalization Tests

Reported AUCs of 0.92–0.96 are high enough to warrant checking for leakage or
overfitting before calling this reproducible. Two things were done:

- **Known Limitation (Fixed):** the pipeline originally fit `TfidfVectorizer`
  once on the full 1,120-bill corpus *before* cross-validation (in both
  `run_pipeline.py` and `statistical_rigor_analysis.py`), leaking held-out
  test-fold document-frequency statistics into the training feature space.
  Fixed by wrapping every model as a TF-IDF -> classifier `Pipeline`
  (`src/model_utils.build_*_pipeline`) cross-validated on raw text, so the
  vectorizer refits per training fold. **Effect: AUCs moved by at most 0.008**
  (Random Forest 0.963 → 0.955; others within 0.001–0.011) — this leakage path
  was not the main driver of the scores.
- **Permutation test** (`scripts/permutation_test.py`): shuffles labels and
  repeats cross-validation to build a null AUC distribution per model. True
  AUCs sit far outside their null bands (null distributions center on ~0.50,
  as expected under no signal) — see `results/statistical_analysis/permutation_test_results.csv`
  and `results/figures/permutation_null_distribution.png`.
- **Leave-one-Congress-out** (`scripts/loco_generalization.py`): trains on 4
  Congresses, tests on the 5th entirely unseen one, repeated per Congress —
  the pooled 5-fold CV above still draws test folds from sessions the model
  has seen elsewhere in training. Results (`results/tables/loco_generalization.csv`):

  | Model | Pooled CV AUC | LOCO Mean AUC (Std.) | Drop |
  |---|---|---|---|
  | Random Forest | 0.955 | 0.940 (0.041) | 0.015 |
  | Ridge | 0.928 | 0.909 (0.039) | 0.019 |
  | Logistic | 0.919 | 0.890 (0.053) | 0.029 |
  | LASSO | 0.918 | 0.877 (0.085) | 0.042 |

  Every model retains AUC > 0.85 on a Congressional session it never trained
  on; Random Forest (best model) also generalizes best (smallest drop).

See `report/results_real_data.tex` Section "Is This Leakage or Real Signal?"
for the full writeup, including permutation-test p-values.


## Screenshots

<img width="2968" height="1769" alt="image" src="https://github.com/user-attachments/assets/dd0f1684-caa8-4c7c-a143-56db24224f01" />
<img width="2970" height="2369" alt="image" src="https://github.com/user-attachments/assets/a49335ec-0f6e-43ab-a694-ec629ccf5898" />
<img width="3569" height="2968" alt="image" src="https://github.com/user-attachments/assets/2978f222-fd6a-48db-8f0f-1e26f514e74c" />


Ever wonder why your bill idea never gets passed? Look at the probability:
<img width="2370" height="1468" alt="image" src="https://github.com/user-attachments/assets/5053aaee-ece1-41e7-ba2f-4757ccdb2ca4" />


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

## Streamlit Dashboard

`app.py` is an interactive dashboard over this repo's actual `results/` output
— no synthetic/placeholder content, it reads the real CSVs/figures/checkpoint
the pipeline scripts above produce (and tells you which file is missing if you
haven't run a given script yet).

```bash
streamlit run app.py
```

Sections: **Overview** (headline numbers), **Dataset** (per-Congress breakdown,
word clouds), **Model Performance** (CV comparison table, ROC curves, confusion
matrices), **Feature Interpretation** (top LASSO/RF words), **Leakage Audit &
Generalization** (permutation-test and leave-one-Congress-out results — the
"is this real signal" answer), and **Try It Yourself** (paste floor-speech
text, get a live prediction + which words drove it, from the fitted
`results/checkpoint_pipeline.pkl` pipelines).

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
