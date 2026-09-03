"""Plotting and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids plt.show() hanging/crashing headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from wordcloud import WordCloud
from typing import List, Tuple, Dict


def set_style():
    """Set consistent seaborn and matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def plot_class_balance(y, output_path: str = None):
    """
    Plot class balance (passed vs failed bills).

    Args:
        y: Binary target vector
        output_path: Path to save figure (optional)
    """
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    class_counts = pd.Series(y).value_counts().sort_index()
    class_names = ["Failed", "Passed"]
    colors = ["#d62728", "#2ca02c"]

    bars = ax.bar(class_names, class_counts.values, color=colors, alpha=0.7, edgecolor="black")

    ax.set_ylabel("Number of Bills")
    ax.set_title("Class Balance: Bill Passage Outcomes")
    ax.set_ylim(0, max(class_counts.values) * 1.1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"{int(height)}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_wordcloud(texts: List[str], output_path: str = None, title: str = "Word Cloud"):
    """
    Plot word cloud from text list.

    Args:
        texts: List of text documents
        output_path: Path to save figure (optional)
        title: Title for the word cloud
    """
    set_style()

    combined_text = " ".join(texts)

    wordcloud = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        max_words=100,
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_feature_coefficients(
    features_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Feature Coefficients",
    max_features: int = 20
):
    """
    Plot horizontal bar chart of feature coefficients (e.g., from LASSO).

    Args:
        features_df: DataFrame with columns 'feature' and 'coefficient'
        output_path: Path to save figure (optional)
        title: Title for the plot
        max_features: Number of features to display
    """
    set_style()

    df = features_df.head(max_features).copy()
    df = df.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    colors = ["#d62728" if x < 0 else "#2ca02c" for x in df["coefficient"]]
    ax.barh(df["feature"], df["coefficient"], color=colors, alpha=0.7, edgecolor="black")

    ax.set_xlabel("Coefficient Value")
    ax.set_title(title, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_feature_importance(
    features_df: pd.DataFrame,
    output_path: str = None,
    title: str = "Feature Importance",
    max_features: int = 30
):
    """
    Plot horizontal bar chart of feature importances (e.g., from Random Forest).

    Args:
        features_df: DataFrame with columns 'feature' and 'importance'
        output_path: Path to save figure (optional)
        title: Title for the plot
        max_features: Number of features to display
    """
    set_style()

    df = features_df.head(max_features).copy()
    df = df.sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))

    ax.barh(df["feature"], df["importance"], color="#1f77b4", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_roc_curves(
    models_data: List[Tuple[str, np.ndarray, np.ndarray]],
    output_path: str = None,
):
    """
    Plot overlaid ROC curves for multiple models.

    Args:
        models_data: List of (model_name, y_true, y_score) tuples
        output_path: Path to save figure (optional)
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, y_true, y_score in models_data:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", linewidth=2)

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_confusion_matrices(
    models_data: List[Tuple[str, np.ndarray]],
    output_path: str = None,
):
    """
    Plot grid of confusion matrices for multiple models.

    Args:
        models_data: List of (model_name, confusion_matrix) tuples
        output_path: Path to save figure (optional)
    """
    set_style()

    n_models = len(models_data)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, cm) in enumerate(models_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Failed", "Passed"],
                    yticklabels=["Failed", "Passed"],
                    cbar=False)

        ax.set_title(f"{model_name}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()


def plot_cv_lambda_search(
    Cs_values: np.ndarray,
    cv_scores_mean: np.ndarray,
    cv_scores_std: np.ndarray,
    output_path: str = None,
    title: str = "Cross-Validation Lambda Search",
):
    """
    Plot CV error vs lambda (or C) for regularization tuning.

    Args:
        Cs_values: Array of C values (inverse lambda)
        cv_scores_mean: Mean CV scores
        cv_scores_std: Std dev of CV scores
        output_path: Path to save figure (optional)
        title: Title for the plot
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(np.log10(Cs_values), cv_scores_mean, yerr=cv_scores_std,
                marker="o", capsize=5, capthick=2, linewidth=2, label="CV Score")

    ax.set_xlabel("log10(C) [Inverse Lambda]")
    ax.set_ylabel("AUC-ROC Score")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")

    plt.show()
