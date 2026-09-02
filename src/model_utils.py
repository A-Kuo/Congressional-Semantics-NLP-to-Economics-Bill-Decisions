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
import numpy as np


SEED = 42


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
        Cs grid: 20 log-spaced values over [10^-4, 10^4]. Report Appendix A
        documents a narrower [10^-3, 10^3] / 10-value grid from an earlier
        run; this wider grid is a superset and does not change which C nested
        CV selects. Keep Appendix A's grid description or this code in sync
        if either changes.
    """
    # l1_ratios=(1,) is pure L1 (LASSO); saga solver required for elastic net
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=cv_splits,
        l1_ratios=(1,),
        solver="saga",
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
        Cs grid: 20 log-spaced values over [10^-4, 10^4] (see train_lasso_logistic).
    """
    # l1_ratios=(0,) is pure L2 (Ridge); saga solver required for elastic net
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20),
        cv=cv_splits,
        l1_ratios=(0,),
        solver="saga",
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
