"""Unit tests for model_utils module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from src import model_utils


class TestModelTraining:
    """Test model training functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=100,
            n_informative=20,
            random_state=42
        )
        return X, y

    def test_train_logistic_regression_returns_model(self, sample_data):
        """Test that logistic regression training returns a fitted model."""
        X, y = sample_data
        model = model_utils.train_logistic_regression(X, y, random_state=42)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_logistic_regression_can_predict(self, sample_data):
        """Test that trained model can make predictions."""
        X, y = sample_data
        model = model_utils.train_logistic_regression(X, y, random_state=42)

        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})

    def test_train_lasso_logistic_returns_model(self, sample_data):
        """Test that LASSO model returns a fitted model."""
        X, y = sample_data
        model = model_utils.train_lasso_logistic(X, y, cv_splits=5, random_state=42)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")

    def test_train_lasso_creates_sparsity(self, sample_data):
        """Test that LASSO creates sparse coefficients."""
        X, y = sample_data
        model = model_utils.train_lasso_logistic(X, y, cv_splits=5, random_state=42)

        coefs = model.coef_[0]

        # Some coefficients should be exactly zero
        num_zero = np.sum(coefs == 0)
        assert num_zero > 0

    def test_train_ridge_logistic_returns_model(self, sample_data):
        """Test that Ridge model returns a fitted model."""
        X, y = sample_data
        model = model_utils.train_ridge_logistic(X, y, cv_splits=5, random_state=42)

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "coef_")

    def test_train_ridge_keeps_features(self, sample_data):
        """Test that Ridge doesn't zero out coefficients."""
        X, y = sample_data
        model = model_utils.train_ridge_logistic(X, y, cv_splits=5, random_state=42)

        coefs = model.coef_[0]

        # Few (or no) coefficients should be exactly zero
        num_zero = np.sum(coefs == 0)
        assert num_zero < len(coefs) * 0.1  # Less than 10% zeros

    def test_train_random_forest_returns_model(self, sample_data):
        """Test that Random Forest training returns a fitted model."""
        X, y = sample_data
        model = model_utils.train_random_forest(
            X, y, n_estimators=10, random_state=42
        )

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "feature_importances_")

    def test_train_random_forest_feature_importances(self, sample_data):
        """Test that feature importances sum to 1."""
        X, y = sample_data
        model = model_utils.train_random_forest(
            X, y, n_estimators=10, random_state=42
        )

        importances = model.feature_importances_
        assert np.isclose(importances.sum(), 1.0)


class TestModelEvaluation:
    """Test model evaluation functions."""

    @pytest.fixture
    def setup_evaluator_and_data(self):
        """Setup evaluator with sample data."""
        evaluator = model_utils.ModelEvaluator(random_state=42)

        X, y = make_classification(
            n_samples=200,
            n_features=50,
            n_informative=20,
            random_state=42
        )

        model = model_utils.train_logistic_regression(X, y, random_state=42)

        return evaluator, model, X, y

    def test_evaluate_classifier_returns_dict(self, setup_evaluator_and_data):
        """Test that evaluation returns a dictionary with metrics."""
        evaluator, model, X, y = setup_evaluator_and_data

        results = evaluator.evaluate_classifier(model, X, y, model_name="TestModel")

        assert isinstance(results, dict)
        assert "model" in results
        assert "accuracy_mean" in results
        assert "auc_roc_mean" in results
        assert "f1_mean" in results

    def test_evaluate_classifier_metric_ranges(self, setup_evaluator_and_data):
        """Test that metrics are in valid ranges."""
        evaluator, model, X, y = setup_evaluator_and_data

        results = evaluator.evaluate_classifier(model, X, y, model_name="TestModel")

        # All metrics should be between 0 and 1
        for key in ["accuracy_mean", "auc_roc_mean", "precision_mean", "recall_mean", "f1_mean"]:
            assert 0 <= results[key] <= 1, f"{key} out of range: {results[key]}"

    def test_evaluate_classifier_std_dev_exists(self, setup_evaluator_and_data):
        """Test that standard deviations are computed."""
        evaluator, model, X, y = setup_evaluator_and_data

        results = evaluator.evaluate_classifier(model, X, y, model_name="TestModel", cv_splits=5)

        assert "accuracy_std" in results
        assert "auc_roc_std" in results
        assert results["accuracy_std"] >= 0

    def test_get_confusion_matrix_shape(self, setup_evaluator_and_data):
        """Test that confusion matrix has correct shape."""
        evaluator, model, X, y = setup_evaluator_and_data

        cm = evaluator.get_confusion_matrix(model, X, y)

        assert cm.shape == (2, 2)
        assert cm.sum() == len(y)

    def test_get_roc_curve_returns_three_arrays(self, setup_evaluator_and_data):
        """Test that ROC curve returns FPR, TPR, thresholds."""
        evaluator, model, X, y = setup_evaluator_and_data

        fpr, tpr, thresholds = evaluator.get_roc_curve(model, X, y)

        assert len(fpr) > 1
        assert len(tpr) > 1
        assert len(thresholds) > 1
        assert len(fpr) == len(tpr)


class TestFeatureExtraction:
    """Test feature extraction from models."""

    @pytest.fixture
    def setup_models_and_features(self):
        """Setup models and feature data."""
        X, y = make_classification(
            n_samples=200,
            n_features=50,
            n_informative=20,
            random_state=42
        )

        lasso_model = model_utils.train_lasso_logistic(X, y, cv_splits=5, random_state=42)
        rf_model = model_utils.train_random_forest(X, y, n_estimators=10, random_state=42)

        feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])

        return lasso_model, rf_model, feature_names

    def test_get_top_features_lasso_returns_dataframe(self, setup_models_and_features):
        """Test that LASSO feature extraction returns DataFrame."""
        lasso_model, _, feature_names = setup_models_and_features

        result = model_utils.get_top_features_lasso(lasso_model, feature_names, top_n=10)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "coefficient" in result.columns

    def test_get_top_features_lasso_respects_top_n(self, setup_models_and_features):
        """Test that top_n parameter is respected."""
        lasso_model, _, feature_names = setup_models_and_features

        for top_n in [5, 10, 20]:
            result = model_utils.get_top_features_lasso(lasso_model, feature_names, top_n=top_n)
            assert len(result) <= top_n

    def test_get_top_features_lasso_sorted_by_magnitude(self, setup_models_and_features):
        """Test that features are sorted by coefficient magnitude."""
        lasso_model, _, feature_names = setup_models_and_features

        result = model_utils.get_top_features_lasso(lasso_model, feature_names, top_n=20)

        magnitudes = result["coefficient"].abs().values
        # Should be sorted in descending order
        assert np.all(magnitudes[:-1] >= magnitudes[1:])

    def test_get_top_features_rf_returns_dataframe(self, setup_models_and_features):
        """Test that RF feature extraction returns DataFrame."""
        _, rf_model, feature_names = setup_models_and_features

        result = model_utils.get_top_features_rf(rf_model, feature_names, top_n=10)

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_get_top_features_rf_respects_top_n(self, setup_models_and_features):
        """Test that top_n parameter is respected."""
        _, rf_model, feature_names = setup_models_and_features

        for top_n in [5, 10, 20]:
            result = model_utils.get_top_features_rf(rf_model, feature_names, top_n=top_n)
            assert len(result) <= top_n

    def test_get_top_features_rf_importances_positive(self, setup_models_and_features):
        """Test that importances are non-negative."""
        _, rf_model, feature_names = setup_models_and_features

        result = model_utils.get_top_features_rf(rf_model, feature_names, top_n=20)

        assert (result["importance"] >= 0).all()
        assert (result["importance"] <= 1).all()


class TestModelComparison:
    """Test model comparison functions."""

    def test_compare_models_returns_dataframe(self):
        """Test that model comparison returns DataFrame."""
        results_list = [
            {"model": "Model A", "accuracy_mean": 0.85, "auc_roc_mean": 0.88},
            {"model": "Model B", "accuracy_mean": 0.82, "auc_roc_mean": 0.90},
        ]

        result = model_utils.compare_models(results_list)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_compare_models_preserves_data(self):
        """Test that comparison preserves all data."""
        results_list = [
            {"model": "Model A", "accuracy_mean": 0.85, "auc_roc_mean": 0.88},
            {"model": "Model B", "accuracy_mean": 0.82, "auc_roc_mean": 0.90},
        ]

        result = model_utils.compare_models(results_list)

        assert result.loc[0, "model"] == "Model A"
        assert result.loc[0, "accuracy_mean"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
