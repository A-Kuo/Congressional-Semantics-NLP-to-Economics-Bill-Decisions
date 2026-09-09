"""Unit tests for model_utils module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src import model_utils


def _make_text_classification_data(n_samples=120, seed=42):
    """Small synthetic text + label dataset for pipeline tests (avoids needing
    real Congressional data in unit tests; large enough for min_df=5 with a
    permissive vectorizer config override)."""
    rng = np.random.RandomState(seed)
    passed_vocab = ["consent", "unanimous", "suspend", "motion", "senate"]
    failed_vocab = ["controversial", "partisan", "reckless", "unfunded", "deficit"]
    filler = ["the", "bill", "act", "congress", "committee", "vote", "floor"]

    texts, labels = [], []
    for i in range(n_samples):
        is_passed = i % 2 == 0
        vocab = passed_vocab if is_passed else failed_vocab
        words = rng.choice(vocab, size=15).tolist() + rng.choice(filler, size=15).tolist()
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(1 if is_passed else 0)
    return np.array(texts), np.array(labels)


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


class TestPipelineBuilders:
    """Test leak-free TF-IDF -> classifier pipeline builders.

    These pipelines exist so cross_validate/cross_val_predict can refit the
    TfidfVectorizer fresh on each training fold, instead of fitting it once on
    the full corpus before splitting (the leakage bug fixed in this pass).
    Tests fit on raw text directly, mirroring real usage.
    """

    @pytest.fixture
    def text_data(self):
        return _make_text_classification_data()

    def test_build_logistic_pipeline_is_unfit_pipeline(self):
        pipeline = model_utils.build_logistic_pipeline(random_state=42)
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps.keys()) == ["tfidf", "clf"]
        assert not hasattr(pipeline.named_steps["clf"], "coef_")

    def test_logistic_pipeline_fits_and_predicts_on_text(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_logistic_pipeline(random_state=42)
        pipeline.fit(X_text, y)

        preds = pipeline.predict(X_text[:10])
        proba = pipeline.predict_proba(X_text[:10])
        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
        assert proba.shape == (10, 2)

    def test_lasso_pipeline_fits_and_has_coefficients(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_lasso_pipeline(random_state=42)
        pipeline.fit(X_text, y)

        assert hasattr(pipeline.named_steps["clf"], "coef_")
        assert isinstance(pipeline.named_steps["tfidf"].get_feature_names_out(), np.ndarray)

    def test_ridge_pipeline_fits_and_has_coefficients(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_ridge_pipeline(random_state=42)
        pipeline.fit(X_text, y)

        assert hasattr(pipeline.named_steps["clf"], "coef_")

    def test_random_forest_pipeline_fits_on_sparse_tfidf_output(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_random_forest_pipeline(n_estimators=10, random_state=42)
        pipeline.fit(X_text, y)

        assert hasattr(pipeline.named_steps["clf"], "feature_importances_")
        preds = pipeline.predict(X_text[:10])
        assert len(preds) == 10

    def test_get_top_features_lasso_from_pipeline(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_lasso_pipeline(random_state=42)
        pipeline.fit(X_text, y)

        result = model_utils.get_top_features_lasso_from_pipeline(pipeline, top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "coefficient" in result.columns
        assert len(result) <= 5

    def test_get_top_features_rf_from_pipeline(self, text_data):
        X_text, y = text_data
        pipeline = model_utils.build_random_forest_pipeline(n_estimators=10, random_state=42)
        pipeline.fit(X_text, y)

        result = model_utils.get_top_features_rf_from_pipeline(pipeline, top_n=5)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_pipeline_refits_vectorizer_per_call_no_state_bleed(self, text_data):
        """A fresh build_*_pipeline() call must not share vectorizer state with
        a previously-fit pipeline instance (guards against accidentally reusing
        one fitted pipeline object across CV/interpretation call sites)."""
        X_text, y = text_data
        pipeline_a = model_utils.build_logistic_pipeline(random_state=42)
        pipeline_a.fit(X_text, y)

        pipeline_b = model_utils.build_logistic_pipeline(random_state=42)
        assert not hasattr(pipeline_b.named_steps["clf"], "coef_")
        pipeline_b.fit(X_text[:60], y[:60])

        vocab_a = pipeline_a.named_steps["tfidf"].vocabulary_
        vocab_b = pipeline_b.named_steps["tfidf"].vocabulary_
        assert vocab_a is not vocab_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
