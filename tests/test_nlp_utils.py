"""Unit tests for nlp_utils module."""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from src import nlp_utils


class TestTextPreprocessing:
    """Test text preprocessing functions."""

    def test_preprocess_text_lowercase(self):
        """Test that text is lowercased."""
        text = "The QUICK Brown Fox"
        processed = nlp_utils.preprocess_text(text)
        assert processed == processed.lower()

    def test_preprocess_text_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com and http://google.com for info"
        processed = nlp_utils.preprocess_text(text)
        assert "https" not in processed
        assert "http" not in processed

    def test_preprocess_text_removes_emails(self):
        """Test that emails are removed."""
        text = "Contact me at john@example.com or jane@company.org"
        processed = nlp_utils.preprocess_text(text)
        assert "@" not in processed

    def test_preprocess_text_removes_punctuation(self):
        """Test that punctuation is removed."""
        text = "Hello, world! How are you?"
        processed = nlp_utils.preprocess_text(text)
        assert "," not in processed
        assert "!" not in processed
        assert "?" not in processed

    def test_preprocess_text_preserves_words(self):
        """Test that words are preserved."""
        text = "budget deficit taxation spending"
        processed = nlp_utils.preprocess_text(text)
        for word in ["budget", "deficit", "taxation", "spending"]:
            assert word in processed


class TestEconomicFiltering:
    """Test economic speech filtering."""

    def test_filter_economic_speeches_keeps_economic(self):
        """Test that speeches with economic keywords are kept."""
        df = pd.DataFrame({
            "speeches_combined": [
                "This bill is about taxation and budget policy",
                "Discussion of trade tariffs and labor costs",
                "Sports and entertainment regulations"  # No economic keywords
            ],
            "passed": [1, 1, 0]
        })

        filtered = nlp_utils.filter_economic_speeches(df)

        # Should keep only first 2 rows (have economic keywords)
        assert len(filtered) == 2
        assert filtered.index.tolist() == [0, 1]

    def test_filter_economic_speeches_handles_nulls(self):
        """Test that null values don't cause errors."""
        df = pd.DataFrame({
            "speeches_combined": [
                "tax policy",
                None,
                "budget proposal"
            ],
            "passed": [1, 0, 1]
        })

        filtered = nlp_utils.filter_economic_speeches(df)

        # Should not include null row
        assert len(filtered) <= 2
        assert filtered["speeches_combined"].isnull().sum() == 0


class TestTFIDFVectorization:
    """Test TF-IDF feature creation."""

    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for TF-IDF testing."""
        return [
            "budget deficit tax policy economic growth",
            "taxation employment wage labor supply",
            "trade tariff export import commerce",
            "congressional amendment bill legislative process",
            "appropriation spending federal government"
        ]

    def test_create_tfidf_features_shape(self, sample_texts):
        """Test that TF-IDF matrix has correct shape."""
        X, vectorizer, features = nlp_utils.create_tfidf_features(
            sample_texts,
            max_features=10,
            min_df=1,
            max_df=1.0
        )

        assert X.shape[0] == len(sample_texts)  # n_documents
        assert X.shape[1] <= 10  # max_features
        assert len(features) == X.shape[1]

    def test_create_tfidf_features_sparsity(self, sample_texts):
        """Test that TF-IDF matrix is sparse."""
        X, _, _ = nlp_utils.create_tfidf_features(
            sample_texts,
            max_features=100,
            min_df=1,
            max_df=1.0
        )

        # Check that matrix is sparse (most values are 0)
        sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
        assert sparsity > 0.7  # At least 70% sparse

    def test_create_tfidf_features_min_df_filter(self):
        """Test that min_df removes words appearing in fewer than min_df docs."""
        # "frequent" appears in both docs (survives min_df=2)
        # "alpha"/"beta" appear in only 1 doc each (filtered by min_df=2)
        texts = ["frequent alpha policy", "frequent beta policy"]
        X, vectorizer, features = nlp_utils.create_tfidf_features(
            texts,
            max_features=100,
            min_df=2,  # Require at least 2 documents
            max_df=1.0
        )

        # "alpha" and "beta" appear in only 1 doc — should be filtered
        assert "alpha" not in features
        assert "beta" not in features
        # "frequent" appears in both — should survive
        assert "frequent" in features

    def test_create_tfidf_features_max_df_filter(self):
        """Test that max_df removes very common words."""
        texts = [
            "the the the word",
            "the the the word",
            "the the the word",
            "unique"
        ]
        X, vectorizer, features = nlp_utils.create_tfidf_features(
            texts,
            max_features=100,
            min_df=1,
            max_df=0.5  # Max 50% of documents
        )

        # "the" appears in 3/4 (75%) docs, should be filtered
        assert "the" not in features


class TestTopFeatureExtraction:
    """Test feature importance extraction."""

    @pytest.fixture
    def sample_tfidf_data(self):
        """Create sample TF-IDF data."""
        X = csr_matrix(np.array([
            [0.5, 0.3, 0.1, 0.0],
            [0.2, 0.4, 0.2, 0.1],
            [0.1, 0.1, 0.6, 0.2],
        ]))
        features = np.array(["tax", "budget", "deficit", "spending"])
        return X, features

    def test_get_top_tfidf_words_class_1(self, sample_tfidf_data):
        """Test extracting top TF-IDF words for class 1."""
        X, features = sample_tfidf_data
        y = np.array([1, 1, 0])

        # Create a mock vectorizer
        class MockVectorizer:
            def get_feature_names_out(self):
                return features

        vectorizer = MockVectorizer()

        top = nlp_utils.get_top_tfidf_words(vectorizer, X, y, class_label=1, top_n=2)

        # Should return 2 features
        assert len(top) == 2

        # Check format: (word, score) tuples
        for word, score in top:
            assert isinstance(word, str)
            assert isinstance(score, (float, np.floating))
            assert score >= 0

    def test_get_top_tfidf_words_respects_top_n(self, sample_tfidf_data):
        """Test that top_n parameter is respected."""
        X, features = sample_tfidf_data
        y = np.array([1, 1, 0])

        class MockVectorizer:
            def get_feature_names_out(self):
                return features

        for top_n in [1, 2, 3]:
            top = nlp_utils.get_top_tfidf_words(
                MockVectorizer(), X, y, class_label=1, top_n=top_n
            )
            assert len(top) <= top_n


class TestTextFeatureExtraction:
    """Test derived text feature extraction."""

    def test_extract_text_features_creates_columns(self):
        """Test that all text features are created."""
        df = pd.DataFrame({
            "speeches_combined": [
                "This is a short text.",
                "This is a longer text with more words in it.",
                "Yet another example with different statistics."
            ]
        })

        df_feat = nlp_utils.extract_text_features(df)

        assert "text_length" in df_feat.columns
        assert "text_word_count" in df_feat.columns
        assert "sentence_count" in df_feat.columns
        assert "avg_word_length" in df_feat.columns

    def test_extract_text_features_values_reasonable(self):
        """Test that extracted features have reasonable values."""
        df = pd.DataFrame({
            "speeches_combined": [
                "Short.",
                "This is a longer sentence with multiple words.",
            ]
        })

        df_feat = nlp_utils.extract_text_features(df)

        # First text should be shorter
        assert df_feat.iloc[0]["text_length"] < df_feat.iloc[1]["text_length"]
        assert df_feat.iloc[0]["text_word_count"] < df_feat.iloc[1]["text_word_count"]

        # All features should be positive
        assert (df_feat["text_length"] >= 0).all()
        assert (df_feat["text_word_count"] >= 0).all()
        assert (df_feat["sentence_count"] >= 0).all()
        assert (df_feat["avg_word_length"] >= 0).all()

    def test_extract_text_features_handles_nulls(self):
        """Test that null values are handled gracefully."""
        df = pd.DataFrame({
            "speeches_combined": [
                "Normal text here.",
                None,
                "Another text."
            ]
        })

        df_feat = nlp_utils.extract_text_features(df)

        # No NaN in extracted features
        assert df_feat[["text_length", "text_word_count"]].isnull().sum().sum() == 0

        # Null input should result in 0 values
        assert df_feat.iloc[1]["text_length"] == 0
        assert df_feat.iloc[1]["text_word_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
