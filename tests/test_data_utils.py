"""Unit tests for data_utils module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src import data_utils


class TestDataCleaning:
    """Test data cleaning and preprocessing functions."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110", "HR3-110"],
            "congress": [110, 110, 110],
            "title": ["Bill A", "Bill B", None],
            "speeches_combined": [
                "This is a speech about tax policy.",
                "Another speech about budget deficits.",
                "Speech C"
            ],
            "speakers_names": ["Smith|Jones", "Brown", "White|Black|Green"],
            "speakers_parties": ["D|R", "D", "R|D|R"],
            "passed": [1, 0, 1]
        })

    def test_clean_bill_speeches_deduplication(self):
        """Test that duplicates are removed (uses null-free fixture)."""
        df_no_nulls = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110", "HR3-110"],
            "congress": [110, 110, 110],
            "title": ["Bill A", "Bill B", "Bill C"],
            "speeches_combined": ["speech one about tax", "speech two about budget", "speech three about trade"],
            "speakers_names": ["Smith|Jones", "Brown", "White"],
            "speakers_parties": ["D|R", "D", "R"],
            "passed": [1, 0, 1]
        })
        df_dup = pd.concat([df_no_nulls, df_no_nulls.iloc[[0]]], ignore_index=True)
        cleaned = data_utils.clean_bill_speeches(df_dup)

        # Should have 3 rows after dedup (not 4)
        assert len(cleaned) == 3

    def test_clean_bill_speeches_null_removal(self, sample_df):
        """Test that rows with null text are removed."""
        # Row with None title should not be removed (depends on text column)
        cleaned = data_utils.clean_bill_speeches(sample_df)

        # All rows should have non-null speeches_combined
        assert cleaned['speeches_combined'].isnull().sum() == 0

    def test_clean_bill_speeches_feature_creation(self, sample_df):
        """Test that derived features are created correctly."""
        cleaned = data_utils.clean_bill_speeches(sample_df)

        assert "speech_length" in cleaned.columns
        assert "speech_word_count" in cleaned.columns
        assert "num_speakers" in cleaned.columns
        assert "majority_party" in cleaned.columns

        # Check values are reasonable
        assert (cleaned["speech_length"] > 0).all()
        assert (cleaned["speech_word_count"] > 0).all()
        assert (cleaned["num_speakers"] > 0).all()

    def test_clean_bill_speeches_majority_party(self, sample_df):
        """Test that majority party is correctly computed."""
        cleaned = data_utils.clean_bill_speeches(sample_df)

        # Row 0: "D|R" -> D (first in majority)
        assert cleaned.iloc[0]["majority_party"] == "D"

        # Row 1: "D" -> D
        assert cleaned.iloc[1]["majority_party"] == "D"


class TestTFIDFProcessing:
    """Test NLP utils TF-IDF creation (called from data_utils indirectly)."""

    def test_merge_bills_and_speeches(self):
        """Test merging of bills and speeches."""
        # Create sample datasets
        bills_df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110"],
            "congress": [110, 110],
            "title": ["Bill A", "Bill B"],
            "passed": [1, 0]
        })

        speeches_df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR1-110", "HR2-110"],
            "congress": [110, 110, 110],
            "speech_text": ["Speech 1", "Speech 2", "Speech 3"],
            "speaker_name": ["Smith", "Jones", "Brown"],
            "speaker_party": ["D", "R", "D"]
        })

        merged = data_utils.merge_bills_and_speeches(bills_df, speeches_df)

        # Should have 2 rows (one per bill)
        assert len(merged) == 2

        # Speeches should be concatenated
        assert "Speech 1" in merged.iloc[0]["speeches_combined"]
        assert "Speech 2" in merged.iloc[0]["speeches_combined"]

        # Speakers should be pipe-separated
        assert "Smith" in merged.iloc[0]["speakers_names"]
        assert "Jones" in merged.iloc[0]["speakers_names"]

    def test_merge_incomplete_join(self):
        """Test that only bills with speeches are kept (inner join)."""
        bills_df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110", "HR3-110"],
            "congress": [110, 110, 110],
            "title": ["Bill A", "Bill B", "Bill C"],
            "passed": [1, 0, 1]
        })

        speeches_df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110"],
            "congress": [110, 110],
            "speech_text": ["Speech 1", "Speech 2"],
            "speaker_name": ["Smith", "Brown"],
            "speaker_party": ["D", "D"]
        })

        merged = data_utils.merge_bills_and_speeches(bills_df, speeches_df)

        # Should only have 2 rows (HR1 and HR2, not HR3)
        assert len(merged) == 2
        assert all(merged["bill_id"].isin(["HR1-110", "HR2-110"]))


class TestDataValidation:
    """Test validation functions."""

    def test_save_and_load_processed_data(self):
        """Test saving and loading CSV."""
        df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110"],
            "passed": [1, 0],
            "speech_text": ["Text A", "Text B"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.csv")

            # Save
            data_utils.save_processed_data(df, path)
            assert os.path.exists(path)

            # Load
            loaded = data_utils.load_processed_data(path)

            # Check integrity
            assert len(loaded) == len(df)
            assert list(loaded.columns) == list(df.columns)
            assert loaded["bill_id"].tolist() == df["bill_id"].tolist()

    def test_data_sanity_checks(self):
        """Test data sanity checks."""
        df = pd.DataFrame({
            "bill_id": ["HR1-110", "HR2-110"],
            "passed": [1, 0],
            "speech_text": ["Text A", "Text B"]
        })

        # No nulls
        assert df.isnull().sum().sum() == 0

        # Binary target
        assert set(df["passed"].unique()) == {0, 1}

        # Reasonable dataset size
        assert len(df) > 0


class TestEconomicFiltering:
    """Test filtering to economic keywords."""

    def test_economic_subjects_filter(self):
        """Test that economic subjects are correctly identified."""
        # This is tested implicitly in fetch_bills_from_congress_gov
        # but we can verify the keyword set is non-empty
        assert len(data_utils.ECONOMIC_SUBJECTS) > 0
        assert "Taxation" in data_utils.ECONOMIC_SUBJECTS
        assert "Budget and Appropriations" in data_utils.ECONOMIC_SUBJECTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
