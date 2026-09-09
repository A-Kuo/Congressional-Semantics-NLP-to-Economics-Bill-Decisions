"""
End-to-end pipeline runner.

Executes the full modeling pipeline (equivalent to notebooks 02–07) and saves
all results tables and figures to results/. This reproduces the findings from
"Predicting Congressional Bill Passage from Floor-Speech Language" (April 2026).

DEFAULT: Uses real data from data/processed/bills_speeches_preprocessed.csv
(1,120 bills, 6.8% pass rate, 110th–114th Congress, economic legislation only).

All four models are built as TF-IDF -> classifier sklearn Pipelines
(src/model_utils.build_*_pipeline) and cross-validated on raw text, so the
TfidfVectorizer refits fresh on each training fold instead of being fit once
on the full corpus before splitting -- see src/model_utils.py docstrings for
why the earlier global-fit approach was a leakage bug.

Usage:
    python run_pipeline.py                        # RECOMMENDED: use real data
    python run_pipeline.py --synthetic            # test mode: use synthetic data
    python run_pipeline.py --synthetic --n 500    # test: N synthetic bills
"""

import argparse
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

sys.path.insert(0, ".")
from src import data_utils, nlp_utils, model_utils, viz_utils

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED_MERGED = "data/processed/bills_speeches_merged.csv"
PROCESSED_PREP   = "data/processed/bills_speeches_preprocessed.csv"
TABLES_DIR       = "results/tables"
FIGURES_DIR      = "results/figures"

for d in [TABLES_DIR, FIGURES_DIR, "results"]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ── Data Loading & Validation
# ══════════════════════════════════════════════════════════════════════════════

def step1_load_data(use_synthetic: bool = False, n_synthetic: int = 2000) -> pd.DataFrame:
    """Load (or generate) the preprocessed dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: Data Loading & Validation")
    print("=" * 60)

    if use_synthetic or not os.path.exists(PROCESSED_PREP):
        print("Generating synthetic dataset...")
        from src.synthetic_data import generate_preprocessed_dataset
        df = generate_preprocessed_dataset(n_bills=n_synthetic,
                                            output_path=PROCESSED_PREP)
    else:
        df = data_utils.load_processed_data(PROCESSED_PREP)
        print(f"Loaded real dataset: {PROCESSED_PREP}")

    # Assertions
    assert len(df) >= 100, f"Dataset too small ({len(df)})"
    assert "speeches_combined" in df.columns
    assert "passed" in df.columns
    assert df["speeches_combined"].isnull().sum() == 0
    assert set(df["passed"].unique()) == {0, 1}

    pass_rate = df["passed"].mean()
    print(f"✓ Loaded {len(df)} bills  |  pass rate: {pass_rate:.1%}")

    # Class balance figure
    viz_utils.plot_class_balance(df["passed"].values,
                                  output_path=f"{FIGURES_DIR}/class_balance.png")
    print("✓ class_balance.png saved")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ── TF-IDF Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def step2_tfidf(df: pd.DataFrame):
    """
    Descriptive TF-IDF stats + word clouds (vocabulary size, sparsity).

    This fits a TfidfVectorizer on the FULL corpus, but only for descriptive/EDA
    output (sparsity %, word clouds). That fit is never reused for evaluation --
    doing so would reintroduce global-vocabulary leakage (test-fold document
    frequencies leaking into "training" features). Steps 3-6 instead build their
    own TF-IDF-inside-Pipeline objects (src/model_utils.build_*_pipeline) that
    are cross-validated on raw text, refitting the vectorizer fresh per fold.

    Returns (X_text, y) for use by steps 3-6.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TF-IDF Feature Engineering")
    print("=" * 60)

    X_text = df["speeches_combined"].values
    y = df["passed"].values

    X_tfidf, vectorizer, feature_names = nlp_utils.create_tfidf_features(
        X_text, max_features=5000, min_df=5, max_df=0.95
    )

    # Assertions (descriptive only -- this matrix is not used for modeling)
    assert X_tfidf.shape[0] == len(y)
    assert X_tfidf.shape[1] > 10
    sparsity = 1 - (X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))
    # Sparsity threshold is lower for small vocabularies (synthetic data has ~90 features;
    # real data with 5000 features will naturally exceed 90%).
    min_sparsity = 0.50 if X_tfidf.shape[1] >= 500 else 0.10
    assert sparsity > min_sparsity, f"Sparsity {sparsity:.1%} < {min_sparsity:.0%}"

    print(f"✓ TF-IDF matrix: {X_tfidf.shape[0]} docs × {X_tfidf.shape[1]} features (full-corpus, descriptive only)")
    print(f"  Sparsity: {sparsity:.1%}")

    # Word clouds (built from raw text directly, not the TF-IDF matrix above)
    viz_utils.plot_wordcloud(df[df["passed"] == 1]["speeches_combined"].values,
                              output_path=f"{FIGURES_DIR}/tfidf_wordcloud_passed.png",
                              title="Top Terms in PASSED Bills")
    viz_utils.plot_wordcloud(df[df["passed"] == 0]["speeches_combined"].values,
                              output_path=f"{FIGURES_DIR}/tfidf_wordcloud_failed.png",
                              title="Top Terms in FAILED Bills")
    print("✓ Word cloud figures saved")

    return X_text, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ── Baseline Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def step3_logistic(X_text, y) -> dict:
    """Train and evaluate baseline logistic regression (TF-IDF -> classifier pipeline)."""
    print("\n" + "=" * 60)
    print("STEP 3: Baseline Logistic Regression")
    print("=" * 60)

    pipeline = model_utils.build_logistic_pipeline(random_state=SEED)
    evaluator = model_utils.ModelEvaluator(random_state=SEED)
    # Leak-free: evaluate_classifier -> cross_validate clones this pipeline per
    # fold, so the TF-IDF step refits on each training fold only.
    results = evaluator.evaluate_classifier(pipeline, X_text, y,
                                             model_name="Logistic Regression (Baseline)")

    # Assertions
    for k in ["accuracy_mean", "auc_roc_mean", "f1_mean"]:
        assert 0 <= results[k] <= 1, f"{k} out of range"

    pd.DataFrame([results]).to_csv(f"{TABLES_DIR}/logistic_cv_scores.csv", index=False)
    print(f"✓ Accuracy: {results['accuracy_mean']:.3f}  AUC-ROC: {results['auc_roc_mean']:.3f}")

    beats = results["accuracy_mean"] > 0.50
    print(f"  Beats random chance: {'YES' if beats else 'NO'}")

    # Full-sample fit for interpretation / step6's cross_val_predict (which
    # clones this pipeline fresh per fold regardless of its current fit state).
    pipeline.fit(X_text, y)
    return {"pipeline": pipeline, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ── LASSO & Ridge Regularization
# ══════════════════════════════════════════════════════════════════════════════

def step4_regularization(X_text, y) -> dict:
    """Train LASSO and Ridge (TF-IDF -> classifier pipelines); extract top features."""
    print("\n" + "=" * 60)
    print("STEP 4: LASSO & Ridge Regularization")
    print("=" * 60)

    lasso_pipeline = model_utils.build_lasso_pipeline(random_state=SEED)
    ridge_pipeline = model_utils.build_ridge_pipeline(random_state=SEED)

    evaluator = model_utils.ModelEvaluator(random_state=SEED)
    lasso_res = evaluator.evaluate_classifier(lasso_pipeline, X_text, y, model_name="LASSO")
    ridge_res = evaluator.evaluate_classifier(ridge_pipeline, X_text, y, model_name="Ridge")

    pd.DataFrame([lasso_res, ridge_res]).to_csv(
        f"{TABLES_DIR}/lasso_ridge_comparison.csv", index=False)

    # Full-sample fit for feature interpretation (not a held-out performance number
    # -- the CV metrics above already came from the leak-free per-fold evaluation).
    lasso_pipeline.fit(X_text, y)
    ridge_pipeline.fit(X_text, y)

    # Top LASSO features
    top_lasso = model_utils.get_top_features_lasso_from_pipeline(lasso_pipeline, top_n=20)
    top_lasso.to_csv(f"{TABLES_DIR}/lasso_top_features.csv", index=False)

    viz_utils.plot_feature_coefficients(
        top_lasso,
        output_path=f"{FIGURES_DIR}/lasso_coefficients.png",
        title="LASSO Feature Coefficients (Top-20)",
        max_features=20,
    )

    lasso_clf = lasso_pipeline.named_steps["clf"]
    n_features = lasso_pipeline.named_steps["tfidf"].get_feature_names_out().shape[0]
    n_zero = (np.asarray(lasso_clf.coef_).squeeze() == 0).sum()
    sparsity = n_zero / n_features
    print(f"✓ LASSO: AUC {lasso_res['auc_roc_mean']:.3f}  |  Ridge: AUC {ridge_res['auc_roc_mean']:.3f}")
    print(f"  LASSO sparsity: {sparsity:.1%}  ({n_zero} features zeroed)")
    print(f"  Top positive word: {top_lasso[top_lasso['coefficient'] > 0].iloc[0]['feature'] if (top_lasso['coefficient'] > 0).any() else 'N/A'}")
    print(f"  Top negative word: {top_lasso[top_lasso['coefficient'] < 0].iloc[0]['feature'] if (top_lasso['coefficient'] < 0).any() else 'N/A'}")

    return {
        "lasso": {"pipeline": lasso_pipeline, "results": lasso_res},
        "ridge": {"pipeline": ridge_pipeline, "results": ridge_res},
        "top_features": top_lasso,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ── Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def step5_random_forest(X_text, y, lasso_top_features) -> dict:
    """Train RF (TF-IDF -> classifier pipeline); compare feature agreement with LASSO."""
    print("\n" + "=" * 60)
    print("STEP 5: Random Forest")
    print("=" * 60)

    rf_pipeline = model_utils.build_random_forest_pipeline(n_estimators=200, random_state=SEED)

    evaluator = model_utils.ModelEvaluator(random_state=SEED)
    rf_res = evaluator.evaluate_classifier(rf_pipeline, X_text, y, model_name="Random Forest")
    pd.DataFrame([rf_res]).to_csv(f"{TABLES_DIR}/rf_cv_scores.csv", index=False)

    # Full-sample fit for feature interpretation (not a held-out performance number).
    rf_pipeline.fit(X_text, y)

    # Feature importances
    top_rf = model_utils.get_top_features_rf_from_pipeline(rf_pipeline, top_n=30)
    top_rf.to_csv(f"{TABLES_DIR}/rf_top_features.csv", index=False)
    viz_utils.plot_feature_importance(
        top_rf,
        output_path=f"{FIGURES_DIR}/rf_importance.png",
        title="Random Forest Feature Importances (Top-30)",
        max_features=30,
    )

    # Overlap with LASSO
    lasso_set = set(lasso_top_features["feature"].head(20))
    rf_set = set(top_rf["feature"].head(20))
    overlap = lasso_set & rf_set
    print(f"✓ RF AUC: {rf_res['auc_roc_mean']:.3f}")
    print(f"  LASSO/RF top-20 overlap: {len(overlap)} features  {sorted(overlap)[:5]}")

    # Assertions
    rf_clf = rf_pipeline.named_steps["clf"]
    assert np.isclose(rf_clf.feature_importances_.sum(), 1.0, atol=1e-3)

    return {"pipeline": rf_pipeline, "results": rf_res, "top_features": top_rf}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ── Model Comparison & Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def step6_comparison(models_info: dict, X_text, y) -> pd.DataFrame:
    """Aggregate all results, produce ROC curves and confusion matrices."""
    print("\n" + "=" * 60)
    print("STEP 6: Model Comparison & Visualizations")
    print("=" * 60)

    from sklearn.metrics import confusion_matrix

    all_results = [
        models_info["logistic"]["results"],
        models_info["lasso"]["results"],
        models_info["ridge"]["results"],
        models_info["rf"]["results"],
    ]
    comparison = pd.DataFrame(all_results)
    comparison.to_csv(f"{TABLES_DIR}/model_comparison.csv", index=False)

    # Cross-val predictions for ROC & confusion matrices.
    # cross_val_predict clones each pipeline per fold (discarding any prior
    # full-data fit) and refits TF-IDF on that fold's training text only --
    # this is the same leakage bug fixed in evaluate_classifier above, now
    # fixed here too for the ROC/confusion-matrix plots.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    lr_prob  = cross_val_predict(models_info["logistic"]["pipeline"], X_text, y, cv=cv, method="predict_proba")[:, 1]
    las_prob = cross_val_predict(models_info["lasso"]["pipeline"],    X_text, y, cv=cv, method="predict_proba")[:, 1]
    rid_prob = cross_val_predict(models_info["ridge"]["pipeline"],    X_text, y, cv=cv, method="predict_proba")[:, 1]
    rf_prob  = cross_val_predict(models_info["rf"]["pipeline"],       X_text, y, cv=cv, method="predict_proba")[:, 1]

    viz_utils.plot_roc_curves(
        [("Logistic", y, lr_prob), ("LASSO", y, las_prob),
         ("Ridge", y, rid_prob), ("Random Forest", y, rf_prob)],
        output_path=f"{FIGURES_DIR}/roc_curves.png",
    )

    lr_pred  = (lr_prob >= 0.5).astype(int)
    las_pred = (las_prob >= 0.5).astype(int)
    rid_pred = (rid_prob >= 0.5).astype(int)
    rf_pred  = (rf_prob >= 0.5).astype(int)

    viz_utils.plot_confusion_matrices(
        [("Logistic", confusion_matrix(y, lr_pred)),
         ("LASSO",    confusion_matrix(y, las_pred)),
         ("Ridge",    confusion_matrix(y, rid_pred)),
         ("Random Forest", confusion_matrix(y, rf_pred))],
        output_path=f"{FIGURES_DIR}/confusion_matrices.png",
    )

    # Assertions
    assert len(comparison) >= 3
    assert comparison["accuracy_mean"].between(0, 1).all()
    assert comparison["auc_roc_mean"].between(0, 1).all()

    best = comparison.sort_values("auc_roc_mean", ascending=False).iloc[0]
    print(f"✓ Best model: {best['model']}  AUC-ROC: {best['auc_roc_mean']:.3f}")
    print("\nFull ranking:")
    for _, row in comparison.sort_values("auc_roc_mean", ascending=False).iterrows():
        print(f"  {row['model']:35s}  acc={row['accuracy_mean']:.3f}  auc={row['auc_roc_mean']:.3f}  f1={row['f1_mean']:.3f}")

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(use_synthetic: bool = False, n_synthetic: int = 2000):
    print("=" * 60)
    print("Congressional NLP — Full Pipeline")
    print("=" * 60)

    df = step1_load_data(use_synthetic=use_synthetic, n_synthetic=n_synthetic)
    X_text, y = step2_tfidf(df)  # descriptive TF-IDF stats/word clouds only

    logistic_info = step3_logistic(X_text, y)
    reg_info      = step4_regularization(X_text, y)
    rf_info       = step5_random_forest(X_text, y, reg_info["top_features"])

    models_info = {
        "logistic": logistic_info,
        "lasso":    reg_info["lasso"],
        "ridge":    reg_info["ridge"],
        "rf":       rf_info,
    }
    comparison = step6_comparison(models_info, X_text, y)

    # Save checkpoint: the full-sample-fit pipelines (each bundles its own
    # fitted TF-IDF vectorizer + classifier) plus the raw text/labels.
    checkpoint = {
        "pipelines": {name: info["pipeline"] for name, info in models_info.items()},
        "X_text": X_text,
        "y": y,
    }
    with open("results/checkpoint_pipeline.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Tables saved to  {TABLES_DIR}/")
    print(f"  Figures saved to {FIGURES_DIR}/")
    print("\nResearch Question Answers:")
    print(f"  H0: Does text predict passage?  → AUC baseline: {logistic_info['results']['auc_roc_mean']:.3f}")
    print(f"  H1: Which words matter?         → See {TABLES_DIR}/lasso_top_features.csv")
    rf_auc  = models_info["rf"]["results"]["auc_roc_mean"]
    lr_auc  = logistic_info["results"]["auc_roc_mean"]
    print(f"  H2: Non-linear interactions?    → RF ({rf_auc:.3f}) vs Logistic ({lr_auc:.3f}): {'+' if rf_auc > lr_auc else ''}{rf_auc - lr_auc:+.3f}")
    print(f"  H3: BERT needed?                → Run notebook 06 on Colab (optional)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full congressional bill prediction pipeline.\n"
                    "Default: uses real data from data/processed/ (reproduces report findings).\n"
                    "Use --synthetic to run on synthetic data for testing."
    )
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use synthetic data for testing (default: False, uses real data)")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of synthetic bills to generate (only with --synthetic)")
    args = parser.parse_args()

    use_synthetic = args.synthetic
    main(use_synthetic=use_synthetic, n_synthetic=args.n)
