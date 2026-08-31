"""NLP utilities: tokenization, TF-IDF, preprocessing."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from typing import Tuple, List


# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


ECONOMIC_KEYWORDS = {
    "tax", "taxation", "budget", "deficit", "trade", "tariff",
    "wage", "labor", "employment", "unemployment", "budget",
    "appropriation", "spending", "revenue", "income", "corporation",
    "business", "commerce", "industry", "economic", "economy",
    "fiscal", "financial", "bank", "credit", "debt",
}

STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Preprocess raw speech text.

    Args:
        text: Raw speech text

    Returns:
        Cleaned text: lowercase, punctuation removed, stopwords removed
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters but keep spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_economic_speeches(df: pd.DataFrame, text_column: str = "speeches_combined") -> pd.DataFrame:
    """
    Filter speeches to those containing economic keywords.

    Args:
        df: DataFrame with speech text
        text_column: Column name containing speech text

    Returns:
        Filtered DataFrame
    """
    def has_economic_keywords(text):
        if pd.isna(text):
            return False
        words = set(text.lower().split())
        return bool(words & ECONOMIC_KEYWORDS)

    mask = df[text_column].apply(has_economic_keywords)
    filtered = df[mask].copy()

    print(f"Filtered {len(df)} to {len(filtered)} bills with economic keywords")

    return filtered


def create_tfidf_features(
    texts: List[str],
    max_features: int = 5000,
    min_df: int = 5,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = (1, 1),
    preprocessor=None,
) -> Tuple[np.ndarray, TfidfVectorizer, List[str]]:
    """
    Create TF-IDF feature matrix from texts.

    Args:
        texts: List of document texts
        max_features: Max number of features to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency (as fraction)
        ngram_range: N-gram range, e.g., (1, 1) for unigrams, (1, 2) for uni+bigrams
        preprocessor: Custom preprocessing function (default: None, use sklearn's)

    Returns:
        (tfidf_matrix, vectorizer, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words="english",
        preprocessor=preprocessor,
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",  # Only alphabetic tokens
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    print(f"Created TF-IDF matrix: {tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features")

    return tfidf_matrix, vectorizer, feature_names


def get_top_tfidf_words(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    y: np.ndarray,
    class_label: int = 1,
    top_n: int = 30,
) -> List[Tuple[str, float]]:
    """
    Get top TF-IDF words for a specific class.

    Args:
        vectorizer: Fitted TfidfVectorizer
        tfidf_matrix: Sparse matrix from vectorizer
        y: Binary target vector
        class_label: Class to extract words for (0 or 1)
        top_n: Number of top words to return

    Returns:
        List of (word, mean_tfidf_score) tuples, sorted by score descending
    """
    feature_names = vectorizer.get_feature_names_out()
    class_mask = y == class_label

    # Calculate mean TF-IDF for class
    class_matrix = tfidf_matrix[class_mask]
    mean_tfidf = np.asarray(class_matrix.mean(axis=0)).flatten()

    # Get indices of top words
    top_indices = np.argsort(mean_tfidf)[-top_n:][::-1]

    top_words = [
        (feature_names[i], mean_tfidf[i])
        for i in top_indices
    ]

    return top_words


def extract_text_features(df: pd.DataFrame, text_column: str = "speeches_combined") -> pd.DataFrame:
    """
    Extract additional text features from speeches.

    Args:
        df: DataFrame with speech text
        text_column: Column name containing speech text

    Returns:
        DataFrame with new columns: text_length, text_word_count, sentence_count, avg_word_length
    """
    df = df.copy()

    df["text_length"] = df[text_column].fillna("").str.len()
    df["text_word_count"] = df[text_column].fillna("").str.split().str.len()

    # Sentence count (approximate)
    df["sentence_count"] = df[text_column].fillna("").str.count(r"[.!?]") + 1

    # Average word length
    def avg_word_length(text):
        if pd.isna(text) or len(text) == 0:
            return 0
        words = text.split()
        return np.mean([len(w) for w in words]) if len(words) > 0 else 0

    df["avg_word_length"] = df[text_column].fillna("").apply(avg_word_length)

    return df
