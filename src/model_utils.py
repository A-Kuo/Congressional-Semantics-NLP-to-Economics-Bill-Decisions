"""Model training, CV, and evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


SEED = 42

# Shared TF-IDF config (matches report Appendix A and src/nlp_utils.create_tfidf_features).
# Centralized here so build_*_pipeline functions and any script that needs the same
# vectorizer settings (e.g. statistical_rigor_analysis.py) share one source of truth.
TFIDF_KWARGS = dict(
    max_features=5000,
    min_df=5,
    max_df=0.95,
    stop_words="english",
    lowercase=True,
    token_pattern=r"\b[a-z]+\b",
)


def _make_tfidf():
    """Fresh, unfit TfidfVectorizer using the shared config."""
    return TfidfVectorizer(**TFIDF_KWARGS)


class ModelEvaluator:
    """Helper class for model evaluation and comparison."""

    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def evaluate_classifier(
        self,
        model,
        X,
        y,
        model_name: str = "Model",
        cv_splits: int = 5,
    ) -> dict:
        """
        Evaluate a classifier with cross-validation.

        Args:
            model: Fitted sklearn classifier
            X: Feature matrix
            y: Target vector
            model_name: Name for reporting
            cv_splits: Number of CV folds

        Returns:
            Dictionary with metrics: accuracy, auc_roc, precision, recall, f1
        """
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        # Cross-validation scores
        scoring = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

        results = {
            "model": model_name,
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "auc_roc_mean": cv_results["test_roc_auc"].mean(),
            "auc_roc_std": cv_results["test_roc_auc"].std(),
            "precision_mean": cv_results["test_precision"].mean(),
            "precision_std": cv_results["test_precision"].std(),
            "recall_mean": cv_results["test_recall"].mean(),
            "recall_std": cv_results["test_recall"].std(),
            "f1_mean": cv_results["test_f1"].mean(),
            "f1_std": cv_results["test_f1"].std(),
        }

        return results

    def get_confusion_matrix(self, model, X, y, threshold=0.5) -> np.ndarray:
        """Get confusion matrix from model predictions."""
        if hasattr(model, "predict_proba"):
            y_pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        else:
            y_pred = model.predict(X)
        return confusion_matrix(y, y_pred)

    def get_roc_curve(self, model, X, y) -> tuple:
        """Get ROC curve data (fpr, tpr, thresholds)."""
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = model.decision_function(X)
        return roc_curve(y, y_proba)


def train_logistic_regression(
    X, y, random_state=SEED
) -> LogisticRegression:
    """
    Train baseline logistic regression.

    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed

    Returns:
        Fitted LogisticRegression model
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def train_lasso_logistic(
    X, y, cv_splits=5, random_state=SEED
) -> LogisticRegressionCV:
    """
    Train LASSO-regularized logistic regression (L1 penalty via elastic net).

    Args:
        X: Feature matrix
        y: Target vector
        cv_splits: Number of CV folds for lambda tuning
        random_state: Random seed

    Returns:
        Fitted LogisticRegressionCV model with L1 penalty

    Note:
        Cs grid: 10 log-spaced values over [10^-3, 10^3], matching report
        Appendix A. Uses solver="liblinear" with penalty="l1" (equivalent to
        elastic net at l1_ratio=1, but far faster/more stable on this sparse
        5000-feature matrix than saga -- saga repeatedly failed to converge
        within max_iter and crashed with no traceback in testing).
    """
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=cv_splits,
        l1_ratios=(1,),
        solver="liblinear",
        random_state=random_state,
        max_iter=2000,
        class_weight="balanced",
        scoring="roc_auc",
        use_legacy_attributes=False,
    )
    model.fit(X, y)
    return model


def train_ridge_logistic(
    X, y, cv_splits=5, random_state=SEED
) -> LogisticRegressionCV:
    """
    Train Ridge-regularized logistic regression (L2 penalty via elastic net).

    Args:
        X: Feature matrix
        y: Target vector
        cv_splits: Number of CV folds for lambda tuning
        random_state: Random seed

    Returns:
        Fitted LogisticRegressionCV model with L2 penalty

    Note:
        Cs grid: 10 log-spaced values over [10^-3, 10^3] (see train_lasso_logistic).
        Uses solver="liblinear" with penalty="l2".
    """
    model = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 10),
        cv=cv_splits,
        l1_ratios=(0,),
        solver="liblinear",
        random_state=random_state,
        max_iter=2000,
        class_weight="balanced",
        scoring="roc_auc",
        use_legacy_attributes=False,
    )
    model.fit(X, y)
    return model


def train_random_forest(
    X, y, n_estimators=200, max_depth=None, random_state=SEED
) -> RandomForestClassifier:
    """
    Train random forest classifier.

    Args:
        X: Feature matrix
        y: Target vector
        n_estimators: Number of trees
        max_depth: Max tree depth (None = unlimited)
        random_state: Random seed

    Returns:
        Fitted RandomForestClassifier model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features="sqrt",
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def build_logistic_pipeline(random_state=SEED) -> Pipeline:
    """
    Build an UNFIT Logistic Regression pipeline: TF-IDF -> classifier.

    Fitting this as a whole inside cross-validation (e.g. via
    ModelEvaluator.evaluate_classifier or cross_val_predict) refits the TF-IDF
    vectorizer fresh on each training fold, eliminating the global-vocabulary
    leakage that comes from fitting TfidfVectorizer once on the full corpus
    before splitting. Pass raw text (not a precomputed matrix) as X.
    """
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])


def build_lasso_pipeline(random_state=SEED) -> Pipeline:
    """
    Build an UNFIT LASSO logistic regression pipeline: TF-IDF -> classifier.

    See build_logistic_pipeline for why this must be fit inside CV on raw text.
    LogisticRegressionCV's internal cv=5 hyperparameter tuning stays leak-free
    inside a Pipeline: it only ever sees the outer fold's training rows, since
    the whole pipeline (including the TF-IDF step) is fit once per outer fold.
    """
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            cv=5,
            l1_ratios=(1,),
            solver="liblinear",
            random_state=random_state,
            max_iter=2000,
            class_weight="balanced",
            scoring="roc_auc",
            use_legacy_attributes=False,
        )),
    ])


def build_ridge_pipeline(random_state=SEED) -> Pipeline:
    """Build an UNFIT Ridge logistic regression pipeline: TF-IDF -> classifier."""
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 10),
            cv=5,
            l1_ratios=(0,),
            solver="liblinear",
            random_state=random_state,
            max_iter=2000,
            class_weight="balanced",
            scoring="roc_auc",
            use_legacy_attributes=False,
        )),
    ])


def build_random_forest_pipeline(n_estimators=200, random_state=SEED) -> Pipeline:
    """
    Build an UNFIT Random Forest pipeline: TF-IDF -> classifier.

    RandomForestClassifier accepts sparse input directly, so no .toarray()
    densification step is needed inside the pipeline.
    """
    return Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def get_top_features_lasso_from_pipeline(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract top LASSO-selected features from an already-fitted pipeline."""
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    return get_top_features_lasso(pipeline.named_steps["clf"], feature_names, top_n=top_n)


def get_top_features_rf_from_pipeline(pipeline: Pipeline, top_n: int = 30) -> pd.DataFrame:
    """Extract top Random Forest feature importances from an already-fitted pipeline."""
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    return get_top_features_rf(pipeline.named_steps["clf"], feature_names, top_n=top_n)


def get_top_features_lasso(
    model,
    feature_names: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract top LASSO-selected features.

    Args:
        model: Fitted LogisticRegressionCV with L1 penalty
        feature_names: Array of feature names
        top_n: Number of top features to return (by absolute coefficient magnitude)

    Returns:
        DataFrame with columns: feature, coefficient
    """
    # Get coefficients — flatten to 1D regardless of sklearn version
    # Binary classification: coef_ is (1, n_features); squeeze to (n_features,)
    coefs = np.asarray(model.coef_).squeeze()

    # Get indices of top features by absolute value
    top_indices = np.argsort(np.abs(coefs))[-top_n:][::-1]

    top_features_df = pd.DataFrame({
        "feature": feature_names[top_indices],
        "coefficient": coefs[top_indices],
    })

    return top_features_df


def get_top_features_rf(
    model,
    feature_names: np.ndarray,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Extract top Random Forest features by importance.

    Args:
        model: Fitted RandomForestClassifier
        feature_names: Array of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with columns: feature, importance
    """
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_n:][::-1]

    top_features_df = pd.DataFrame({
        "feature": feature_names[top_indices],
        "importance": importances[top_indices],
    })

    return top_features_df


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Compare multiple model evaluation results.

    Args:
        results_list: List of dicts from ModelEvaluator.evaluate_classifier

    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(results_list)
    return df
