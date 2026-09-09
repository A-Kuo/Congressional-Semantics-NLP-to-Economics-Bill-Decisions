"""
Leave-One-Congress-Out (LOCO) generalization test.

The pooled 5-fold StratifiedKFold used elsewhere in this repo shuffles all
1,120 bills together before splitting, so every fold's test set still contains
bills from the SAME five Congressional sessions the model was tuned on. This
script instead trains on four Congresses and tests on the one fully unseen
Congress, repeated once per Congress -- a stricter, real-label-backed test of
whether the model generalizes beyond the specific sessions it has seen, not
just beyond the specific bills.

Uses the same leak-free TF-IDF -> classifier pipelines as run_pipeline.py
(src/model_utils.build_*_pipeline), fit fresh per held-out Congress.

Caveat printed explicitly in the output: Congress 114 has only 6 positive
bills out of 129, so its held-out AUC is high-variance, not a precise
estimate -- read the per-Congress numbers as directional, and the pooled
mean/std across all five as the more stable summary.

Usage:
    python scripts/loco_generalization.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, ".")
from src import data_utils, model_utils

SEED = 42
PREPROCESSED_PATH = "data/processed/bills_speeches_preprocessed.csv"
POOLED_COMPARISON_PATH = "results/tables/model_comparison.csv"
TABLES_DIR = "results/tables"
FIGURES_DIR = "results/figures"

MODEL_BUILDERS = {
    "Logistic": model_utils.build_logistic_pipeline,
    "LASSO": model_utils.build_lasso_pipeline,
    "Ridge": model_utils.build_ridge_pipeline,
    "Random Forest": lambda random_state: model_utils.build_random_forest_pipeline(
        n_estimators=200, random_state=random_state),
}

# Maps this script's model names to run_pipeline.py's model_comparison.csv names,
# for the pooled-vs-LOCO comparison figure.
POOLED_NAME_MAP = {
    "Logistic": "Logistic Regression (Baseline)",
    "LASSO": "LASSO",
    "Ridge": "Ridge",
    "Random Forest": "Random Forest",
}


def main():
    df = data_utils.load_processed_data(PREPROCESSED_PATH)
    congresses = sorted(df["congress"].unique())
    print(f"Loaded {len(df)} bills across Congresses {congresses}")
    for c in congresses:
        sub = df[df["congress"] == c]
        print(f"  Congress {c}: {len(sub)} bills, {sub['passed'].sum()} passed ({sub['passed'].mean():.1%})")

    rows = []
    for held_out in congresses:
        train_df = df[df["congress"] != held_out]
        test_df = df[df["congress"] == held_out]
        X_train, y_train = train_df["speeches_combined"].values, train_df["passed"].values
        X_test, y_test = test_df["speeches_combined"].values, test_df["passed"].values

        n_test_positive = int(y_test.sum())
        if n_test_positive == 0 or n_test_positive == len(y_test):
            print(f"\nCongress {held_out}: skipping AUC (only one class present, "
                  f"{n_test_positive}/{len(y_test)} positive)")
            auc_note = "undefined (single class in held-out set)"
        else:
            auc_note = None

        print(f"\n=== Held out: Congress {held_out} "
              f"(train n={len(train_df)}, test n={len(test_df)}, "
              f"test positives={n_test_positive}) ===")

        for name, builder in MODEL_BUILDERS.items():
            pipeline = builder(random_state=SEED)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            auc = roc_auc_score(y_test, proba) if auc_note is None else np.nan
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)

            print(f"  {name:15s}  AUC={auc if auc_note is None else 'n/a':>6}  "
                  f"acc={acc:.3f}  f1={f1:.3f}")

            rows.append({
                "model": name,
                "held_out_congress": held_out,
                "n_test": len(test_df),
                "n_test_positive": n_test_positive,
                "auc": auc,
                "accuracy": acc,
                "f1": f1,
            })

    results_df = pd.DataFrame(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    out_csv = f"{TABLES_DIR}/loco_generalization.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Per-model summary across held-out Congresses (AUC mean/std, ignoring
    # any Congress where AUC was undefined due to a single-class test set).
    summary = (
        results_df.dropna(subset=["auc"])
        .groupby("model")["auc"]
        .agg(loco_mean_auc="mean", loco_std_auc="std")
        .reset_index()
    )
    print("\n=== LOCO Summary (mean/std AUC across held-out Congresses) ===")
    print(summary.to_string(index=False))

    # Compare against the pooled 5-fold CV AUC from run_pipeline.py's output, if available.
    if os.path.exists(POOLED_COMPARISON_PATH):
        pooled = pd.read_csv(POOLED_COMPARISON_PATH)
        pooled_map = pooled.set_index("model")["auc_roc_mean"].to_dict()
        summary["pooled_cv_auc"] = summary["model"].map(
            lambda m: pooled_map.get(POOLED_NAME_MAP[m], np.nan))
        summary["drop_vs_pooled"] = summary["pooled_cv_auc"] - summary["loco_mean_auc"]
        print("\n=== Pooled CV AUC vs. LOCO mean AUC ===")
        print(summary.to_string(index=False))
        summary.to_csv(f"{TABLES_DIR}/loco_vs_pooled_summary.csv", index=False)

        # Grouped bar chart: pooled CV AUC vs. LOCO mean AUC per model.
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        models = summary["model"].tolist()
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, summary["pooled_cv_auc"], width, label="Pooled 5-fold CV", color="#4C72B0")
        ax.bar(x + width/2, summary["loco_mean_auc"], width, label="LOCO mean", color="#DD8452",
               yerr=summary["loco_std_auc"], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title("Pooled Cross-Validation vs. Leave-One-Congress-Out AUC")
        ax.legend()
        plt.tight_layout()
        out_fig = f"{FIGURES_DIR}/loco_vs_pooled_auc.png"
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out_fig}")
    else:
        print(f"\nNOTE: {POOLED_COMPARISON_PATH} not found -- run run_pipeline.py first "
              f"for the pooled-vs-LOCO comparison.")

    print("\nCaveat: Congress 114 has few positive bills, so its held-out AUC is "
          "high-variance -- read per-Congress rows as directional; the LOCO "
          "mean/std across all Congresses is the more stable summary.")


if __name__ == "__main__":
    main()
