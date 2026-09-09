"""
Permutation / negative-control test: is the reported AUC distinguishable from
chance, or could it plausibly arise from a model with this many features and
this little data even with no real signal?

For each of the four leak-free pipelines (src/model_utils.build_*_pipeline),
runs sklearn.model_selection.permutation_test_score: the TRUE AUC is computed
via normal cross-validation, then labels are shuffled N times and the SAME
cross-validation procedure is repeated on each shuffle to build a null
distribution of what AUC looks like with no real relationship between text and
outcome. If the true AUC sits far outside that null distribution, the score
reflects real signal, not an artifact of leakage or overfitting to this
sample's noise.

Uses real text (not a precomputed TF-IDF matrix), so TF-IDF is refit inside
every single permutation's cross-validation folds -- same leak-free design as
the rest of the pipeline.

Compute budget: LASSO/Ridge's internal LogisticRegressionCV nested tuning
makes each permutation notably more expensive than Logistic/RF, so they use
fewer permutations (200 vs 1000). See plan/README for reasoning.

Usage:
    python scripts/permutation_test.py
    python scripts/permutation_test.py --pilot   # quick 20-permutation timing check first
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, permutation_test_score

sys.path.insert(0, ".")
from src import data_utils, model_utils

SEED = 42
PREPROCESSED_PATH = "data/processed/bills_speeches_preprocessed.csv"
TABLES_DIR = "results/statistical_analysis"
FIGURES_DIR = "results/figures"

MODEL_BUILDERS = {
    "Logistic": (model_utils.build_logistic_pipeline, 300),
    "LASSO": (model_utils.build_lasso_pipeline, 200),
    "Ridge": (model_utils.build_ridge_pipeline, 200),
    "Random Forest": (model_utils.build_random_forest_pipeline, 300),
}
# Permutation counts picked from a 20-permutation timing pilot (--pilot flag):
# Logistic ~1.4s/perm, LASSO ~5.0s/perm (nested CV), Ridge ~2.4s/perm, RF ~3.0s/perm.
# At these counts total runtime is ~45-50 min. The pilot already showed true AUCs
# 10+ null-std-devs above the null mean for all four models, so p-value precision
# beyond ~1/300 buys little -- the effect size, not permutation count, is what's
# doing the work here.


def run_one(name, builder, n_permutations, X_text, y, cv):
    print(f"\n=== {name} ({n_permutations} permutations) ===")
    pipeline = builder(random_state=SEED)
    t0 = time.time()
    true_auc, null_aucs, p_value = permutation_test_score(
        pipeline, X_text, y,
        cv=cv,
        scoring="roc_auc",
        n_permutations=n_permutations,
        random_state=SEED,
        n_jobs=-1,
    )
    elapsed = time.time() - t0
    print(f"  True AUC:      {true_auc:.4f}")
    print(f"  Null mean AUC: {null_aucs.mean():.4f} +/- {null_aucs.std():.4f}")
    print(f"  p-value:       {p_value:.4f}  (fraction of null AUCs >= true AUC)")
    print(f"  Elapsed: {elapsed:.1f}s")
    return {
        "model": name,
        "true_auc": true_auc,
        "null_mean_auc": null_aucs.mean(),
        "null_std_auc": null_aucs.std(),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "elapsed_sec": elapsed,
    }, null_aucs, true_auc


def main(pilot: bool):
    df = data_utils.load_processed_data(PREPROCESSED_PATH)
    X_text = df["speeches_combined"].values
    y = df["passed"].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"Loaded {len(df)} bills, pass rate {y.mean():.1%}")

    if pilot:
        print("\n--- PILOT MODE: 20 permutations per model, timing only ---")
        for name, (builder, _) in MODEL_BUILDERS.items():
            run_one(name, builder, 20, X_text, y, cv)
        print("\nPilot complete. Re-run without --pilot for full results.")
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows = []
    null_distributions = {}
    true_aucs = {}
    for name, (builder, n_perm) in MODEL_BUILDERS.items():
        row, null_aucs, true_auc = run_one(name, builder, n_perm, X_text, y, cv)
        rows.append(row)
        null_distributions[name] = null_aucs
        true_aucs[name] = true_auc

    results_df = pd.DataFrame(rows)
    out_csv = f"{TABLES_DIR}/permutation_test_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Null distribution histogram, one subplot per model, true AUC marked.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.flat, MODEL_BUILDERS.keys()):
        ax.hist(null_distributions[name], bins=20, color="#888", alpha=0.7,
                label="Null (shuffled labels)")
        ax.axvline(true_aucs[name], color="crimson", linewidth=2, label="True AUC")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.5)")
        p_val = results_df.loc[results_df["model"] == name, "p_value"].iloc[0]
        ax.set_title(f"{name}  (p={p_val:.4f})")
        ax.set_xlabel("AUC-ROC")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_fig = f"{FIGURES_DIR}/permutation_null_distribution.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_fig}")

    print("\n=== SUMMARY ===")
    print(results_df[["model", "true_auc", "null_mean_auc", "null_std_auc", "p_value", "n_permutations"]]
          .to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Quick 20-permutation timing pilot before committing to full run")
    args = parser.parse_args()
    main(pilot=args.pilot)
