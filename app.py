"""
Streamlit dashboard for the Congressional Bill Passage Prediction project.

Presents the real-data pipeline's results (dataset, model performance, feature
interpretation) alongside the leakage audit and generalization tests (permutation
test, leave-one-Congress-out) that back this repo's "is this real signal, not
leakage" answer, plus a live prediction demo using the fitted pipelines.

Run with:
    streamlit run app.py

Reads only from results/ and data/processed/ -- run the pipeline scripts first
(see README.md "Real-data pipeline order") if these don't exist yet:
    python scripts/fetch_real_bills.py
    python scripts/link_speeches_to_bills.py
    python scripts/build_merged_dataset.py
    python run_pipeline.py
    python statistical_rigor_analysis.py
    python scripts/permutation_test.py
    python scripts/loco_generalization.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

TABLES_DIR = "results/tables"
FIGURES_DIR = "results/figures"
STATS_DIR = "results/statistical_analysis"
CHECKPOINT_PATH = "results/checkpoint_pipeline.pkl"
PREPROCESSED_PATH = "data/processed/bills_speeches_preprocessed.csv"

MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "lasso": "LASSO Logistic",
    "ridge": "Ridge Logistic",
    "rf": "Random Forest",
}

st.set_page_config(
    page_title="Congressional Bill Passage Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_dataset():
    if not os.path.exists(PREPROCESSED_PATH):
        return None
    return pd.read_csv(PREPROCESSED_PATH)


@st.cache_resource
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    with open(CHECKPOINT_PATH, "rb") as f:
        return pickle.load(f)


def missing_file_notice(path):
    st.info(
        f"`{path}` not found yet. Run the corresponding pipeline script first "
        f"(see the module docstring at the top of `app.py` or `README.md`)."
    )


def show_figure(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        missing_file_notice(path)


# ── Sidebar navigation ──────────────────────────────────────────────────────

PAGES = [
    "Overview",
    "Dataset",
    "Model Performance",
    "Feature Interpretation",
    "Leakage Audit & Generalization",
    "Try It Yourself",
]
page = st.sidebar.radio("Section", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Congressional Bill Passage Prediction**\n\n"
    "Predicting whether U.S. economic legislation becomes law from the language "
    "of its floor-speech debate. Real data: Congress.gov + Stanford Congressional "
    "Record, 110th–114th Congress."
)

dataset = load_dataset()
checkpoint = load_checkpoint()


# ── Overview ─────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Congressional Bill Passage Prediction")
    st.subheader("Can floor-speech language predict whether a bill becomes law?")

    st.markdown(
        """
This project links real U.S. Congressional floor speeches (Stanford Congressional
Record corpus) to real bill outcomes (Congress.gov API) for economic legislation
in the 110th–114th Congress (2007–2016), and asks whether the language used
while a bill is debated predicts whether it becomes law.

**Type:** Predictive, not causal. **Y:** bill passed / failed.
**X:** TF-IDF word frequencies from aggregated floor speeches.
        """
    )

    if dataset is not None:
        n = len(dataset)
        n_passed = int(dataset["passed"].sum())
        pass_rate = dataset["passed"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Bills", f"{n:,}")
        col2.metric("Passed", f"{n_passed}")
        col3.metric("Pass rate", f"{pass_rate:.1%}")
    else:
        missing_file_notice(PREPROCESSED_PATH)

    st.markdown(
        """
### Headline result
The best model (Random Forest) reaches **AUC ≈ 0.955** in cross-validation, and
its top predictive features are **procedural** floor-action vocabulary
(*motion, suspend, unanimous, consent, senate*) rather than substantive economic
terms — language describing *how* a bill is being moved through the chamber,
not *what* it's about.

An AUC this high invites a natural question: is this leakage, or real signal?
See **Leakage Audit & Generalization** for a permutation test and a
leave-one-Congress-out test that both say: real signal, not leakage — every
model still scores AUC > 0.85 on a Congressional session it never trained on.
        """
    )

# ── Dataset ──────────────────────────────────────────────────────────────────

elif page == "Dataset":
    st.title("Dataset")

    if dataset is None:
        missing_file_notice(PREPROCESSED_PATH)
    else:
        st.markdown(
            f"**{len(dataset):,} bills** from the 110th–114th Congress, each with "
            "≥1 linked floor speech (regex-matched explicit \"H.R. ####\" / "
            "\"S. ####\" citations against the real Stanford Congressional Record text)."
        )

        by_congress = (
            dataset.groupby("congress")["passed"]
            .agg(bills="count", passed="sum")
            .reset_index()
        )
        by_congress["pass_rate"] = by_congress["passed"] / by_congress["bills"]
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**By Congress**")
            st.dataframe(
                by_congress.style.format({"pass_rate": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )
        with col2:
            show_figure(f"{FIGURES_DIR}/class_balance.png", "Class balance: passed vs. failed")

        st.markdown("**Most frequent terms, by outcome** (word clouds from raw floor-speech text)")
        wc1, wc2 = st.columns(2)
        with wc1:
            show_figure(f"{FIGURES_DIR}/tfidf_wordcloud_passed.png", "Passed bills")
        with wc2:
            show_figure(f"{FIGURES_DIR}/tfidf_wordcloud_failed.png", "Failed bills")

        with st.expander("Browse sample bills"):
            cols = [c for c in ["bill_id", "congress", "title", "passed"] if c in dataset.columns]
            st.dataframe(dataset[cols].sample(min(20, len(dataset)), random_state=42),
                         use_container_width=True, hide_index=True)

# ── Model Performance ────────────────────────────────────────────────────────

elif page == "Model Performance":
    st.title("Model Performance")
    st.caption(
        "All four models are TF-IDF → classifier sklearn Pipelines, cross-validated "
        "on raw text (5-fold stratified) so the vectorizer refits fresh on each "
        "training fold — no global-vocabulary leakage."
    )

    comparison = load_csv(f"{TABLES_DIR}/model_comparison.csv")
    if comparison is None:
        missing_file_notice(f"{TABLES_DIR}/model_comparison.csv")
    else:
        display_cols = ["model", "accuracy_mean", "auc_roc_mean", "precision_mean",
                         "recall_mean", "f1_mean"]
        display_cols = [c for c in display_cols if c in comparison.columns]
        st.dataframe(
            comparison[display_cols].sort_values("auc_roc_mean", ascending=False)
            .style.format({c: "{:.3f}" for c in display_cols if c != "model"})
            .background_gradient(subset=["auc_roc_mean"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        show_figure(f"{FIGURES_DIR}/roc_curves.png", "ROC curves (5-fold CV)")
    with col2:
        show_figure(f"{FIGURES_DIR}/confusion_matrices.png", "Confusion matrices (threshold 0.5)")

    bias_var = load_csv(f"{STATS_DIR}/bias_variance_analysis.csv")
    if bias_var is not None:
        st.markdown("**Train vs. test AUC (overfitting check)**")
        st.dataframe(bias_var, use_container_width=True, hide_index=True)

# ── Feature Interpretation ───────────────────────────────────────────────────

elif page == "Feature Interpretation":
    st.title("Feature Interpretation")
    st.caption(
        "Full-sample fits, used for interpretation only — not the held-out "
        "performance numbers shown on the Model Performance page."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("LASSO: top coefficients")
        lasso_features = load_csv(f"{TABLES_DIR}/lasso_top_features.csv")
        if lasso_features is not None:
            st.dataframe(
                lasso_features.style.format({"coefficient": "{:+.2f}"})
                .background_gradient(subset=["coefficient"], cmap="RdBu", vmin=-30, vmax=30),
                use_container_width=True, hide_index=True, height=400,
            )
        show_figure(f"{FIGURES_DIR}/lasso_coefficients.png")

    with col2:
        st.subheader("Random Forest: top importances")
        rf_features = load_csv(f"{TABLES_DIR}/rf_top_features.csv")
        if rf_features is not None:
            st.dataframe(
                rf_features.style.format({"importance": "{:.4f}"})
                .background_gradient(subset=["importance"], cmap="Greens"),
                use_container_width=True, hide_index=True, height=400,
            )
        show_figure(f"{FIGURES_DIR}/rf_importance.png")

    st.markdown(
        """
### Reading these tables
Random Forest's top features are dominated by **procedural** floor-action
vocabulary (*motion, suspend, unanimous, consent, senate, pass, rules*) —
language describing how a bill is being moved through the chamber, not what
it's about. LASSO's top coefficients are more mixed: alongside procedural terms
it assigns large weights to topically narrow, idiosyncratic words (e.g. names of
specific bills' subject matter) that plausibly reflect the small (76-bill)
positive class rather than a generalizable signal — see the p-values in
`results/statistical_analysis/STATISTICAL_SUMMARY.txt` for a direct,
quantified version of that caveat.
        """
    )

# ── Leakage Audit & Generalization ──────────────────────────────────────────

elif page == "Leakage Audit & Generalization":
    st.title("Is This Leakage or Real Signal?")
    st.markdown(
        """
AUCs of 0.92–0.96 are high enough to warrant checking for leakage or
overfitting before treating them as reproducible.

**Known limitation (fixed):** the pipeline originally fit `TfidfVectorizer` once
on the full corpus *before* cross-validation, leaking held-out test-fold
document-frequency statistics into training features. Fixed by wrapping every
model as a TF-IDF → classifier `Pipeline` cross-validated on raw text (vectorizer
refits per training fold). **Effect: AUCs moved by at most 0.008** — this
particular leakage path was not the main driver of the scores.

Two further, label-backed tests check genuine generalization:
        """
    )

    st.subheader("1. Permutation test")
    st.caption(
        "Shuffle the passed/failed labels and repeat cross-validation to build a "
        "null AUC distribution per model. If the true AUC sits far outside it, "
        "the score reflects real signal, not an artifact of the feature space's "
        "size relative to the sample."
    )
    perm_results = load_csv(f"{STATS_DIR}/permutation_test_results.csv")
    if perm_results is not None:
        display_cols = ["model", "true_auc", "null_mean_auc", "null_std_auc", "p_value", "n_permutations"]
        display_cols = [c for c in display_cols if c in perm_results.columns]
        st.dataframe(
            perm_results[display_cols]
            .style.format({c: "{:.4f}" for c in display_cols if c not in ("model", "n_permutations")}),
            use_container_width=True, hide_index=True,
        )
        missing_models = {"Logistic", "LASSO", "Ridge", "Random Forest"} - set(perm_results["model"])
        if missing_models:
            st.caption(f"Still running / not yet completed: {', '.join(sorted(missing_models))}. "
                       f"Re-run `python scripts/permutation_test.py` to resume.")
    else:
        missing_file_notice(f"{STATS_DIR}/permutation_test_results.csv")
    show_figure(f"{FIGURES_DIR}/permutation_null_distribution.png",
                "Null AUC distributions (gray) vs. true AUC (red line) per model")

    st.subheader("2. Leave-one-Congress-out (LOCO)")
    st.caption(
        "Train on 4 Congresses, test on the 5th entirely unseen one, repeated per "
        "Congress — a stricter test than pooled cross-validation, which still "
        "draws every fold's test set from sessions the model has seen elsewhere "
        "in training."
    )
    loco_summary = load_csv(f"{TABLES_DIR}/loco_vs_pooled_summary.csv")
    if loco_summary is not None:
        st.dataframe(
            loco_summary.sort_values("loco_mean_auc", ascending=False)
            .style.format({c: "{:.3f}" for c in loco_summary.columns if c != "model"}),
            use_container_width=True, hide_index=True,
        )
    else:
        missing_file_notice(f"{TABLES_DIR}/loco_vs_pooled_summary.csv")
    show_figure(f"{FIGURES_DIR}/loco_vs_pooled_auc.png", "Pooled cross-validation vs. LOCO mean AUC")

    with st.expander("Per-held-out-Congress detail"):
        loco_detail = load_csv(f"{TABLES_DIR}/loco_generalization.csv")
        if loco_detail is not None:
            st.dataframe(loco_detail, use_container_width=True, hide_index=True)
            st.caption(
                "Congress 114 has only 6 of 129 bills passed, so its held-out AUC "
                "is high-variance — read per-Congress rows as directional; the "
                "LOCO mean/std across all Congresses is the more stable summary."
            )

# ── Try It Yourself ──────────────────────────────────────────────────────────

elif page == "Try It Yourself":
    st.title("Try It Yourself")
    st.caption(
        "Paste floor-speech-style text and see what the fitted models predict. "
        "Pipelines are fit on the full 1,120-bill dataset (not held out) — this "
        "is a demo of the model's behavior, not a validation exercise."
    )

    if checkpoint is None:
        missing_file_notice(CHECKPOINT_PATH)
    else:
        example = (
            "Mr. Speaker, I move to suspend the rules and pass the bill, as amended. "
            "This measure has the unanimous consent of both parties and I ask for its "
            "immediate consideration by the House."
        )
        text = st.text_area("Floor-speech text", value=example, height=150)
        model_key = st.selectbox(
            "Model",
            options=list(checkpoint["pipelines"].keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES.get(k, k),
        )

        if st.button("Predict", type="primary") and text.strip():
            pipeline = checkpoint["pipelines"][model_key]
            proba = pipeline.predict_proba([text.lower()])[0, 1]
            st.metric("Predicted probability of passage", f"{proba:.1%}")

            tfidf = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]
            vocab = tfidf.vocabulary_
            input_tokens = set(tfidf.build_analyzer()(text.lower()))
            present_features = [w for w in input_tokens if w in vocab]

            if present_features:
                if hasattr(clf, "coef_"):
                    coefs = np.asarray(clf.coef_).squeeze()
                    rows = [(w, coefs[vocab[w]]) for w in present_features]
                    rows.sort(key=lambda r: abs(r[1]), reverse=True)
                    st.markdown("**Recognized words from this text, by coefficient:**")
                    st.dataframe(
                        pd.DataFrame(rows[:15], columns=["word", "coefficient"])
                        .style.format({"coefficient": "{:+.2f}"})
                        .background_gradient(subset=["coefficient"], cmap="RdBu", vmin=-30, vmax=30),
                        use_container_width=True, hide_index=True,
                    )
                elif hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                    rows = [(w, importances[vocab[w]]) for w in present_features]
                    rows.sort(key=lambda r: r[1], reverse=True)
                    st.markdown("**Recognized words from this text, by importance:**")
                    st.dataframe(
                        pd.DataFrame(rows[:15], columns=["word", "importance"])
                        .style.format({"importance": "{:.4f}"})
                        .background_gradient(subset=["importance"], cmap="Greens"),
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.caption(
                    "None of this text's words are in the model's vocabulary "
                    "(after min_df/max_df filtering) — try more procedural or "
                    "policy-specific language."
                )

        st.markdown("---")
        st.caption(
            "Try replacing the procedural language above with purely substantive "
            "economic language (e.g. removing \"suspend the rules\" / \"unanimous "
            "consent\") to see how much the prediction shifts — this is the same "
            "procedural-vs-substantive contrast discussed on the Feature "
            "Interpretation page."
        )
