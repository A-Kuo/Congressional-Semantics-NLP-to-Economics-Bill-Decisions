"""
Statistical Rigor Analysis: Phase 1 Enhancement
Implements missing statistical tests:
1. P-values for coefficients
2. Confidence intervals (95%)
3. Paired t-test for model comparison
4. Per-fold breakdown
5. MAE, RMSE, R²
6. Bias-variance analysis
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: avoids plt.show() hanging/crashing headless runs
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

print("="*80)
print("STATISTICAL RIGOR ANALYSIS: Congressional Bill Passage Prediction")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA...")
DATA_PATH = 'data/processed/bills_speeches_merged.csv'
df = pd.read_csv(DATA_PATH)

X_text = df['speeches_combined'].values
y = df['passed'].values

print(f"   Loaded {len(df)} bills")
print(f"   Pass rate: {y.mean():.1%}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n2. FEATURE ENGINEERING (TF-IDF)...")
# Matches the report's documented config (Appendix A) and src/nlp_utils.create_tfidf_features
vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.95,
                             stop_words="english", lowercase=True)
X_tfidf = vectorizer.fit_transform(X_text)
feature_names = np.array(vectorizer.get_feature_names_out())

print(f"   TF-IDF shape: {X_tfidf.shape}")
print(f"   Sparsity: {1 - (X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.1%}")

# ============================================================================
# 3. CROSS-VALIDATION SETUP
# ============================================================================
print("\n3. SETTING UP CROSS-VALIDATION...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ============================================================================
# HELPER FUNCTION 1: Compute Standard Errors for Coefficients (Logistic)
# ============================================================================
def compute_logistic_standard_errors(X, y, coefs):
    """
    Compute standard errors for logistic regression coefficients.
    Uses the observed information matrix approach.
    Clips probabilities to prevent numerical issues with near-perfect predictions.
    """
    from scipy.special import expit  # sigmoid function

    X_dense = X.toarray() if hasattr(X, 'toarray') else X
    proba = expit(X_dense @ coefs.T)
    # Clip to avoid p*(1-p) → 0 which makes XtWX singular
    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    # Weights for each observation (p * (1-p))
    weights = proba * (1 - proba)

    # Weighted design matrix using efficient diagonal weighting
    W_sqrt = np.sqrt(weights.ravel())
    XW = X_dense * W_sqrt[:, np.newaxis]
    XtWX = XW.T @ XW

    try:
        # Use pseudoinverse for near-singular matrices
        cov_matrix = np.linalg.pinv(XtWX)
        diag_vals = np.diag(cov_matrix)
        # Clamp any negative values (numerical artifact) before sqrt
        se = np.sqrt(np.maximum(diag_vals, 1e-15))
    except np.linalg.LinAlgError:
        se = np.full_like(coefs, np.nan)

    return se

# ============================================================================
# HELPER FUNCTION 2: Compute All Metrics
# ============================================================================
def compute_all_metrics(y_true, y_pred_prob, y_pred_binary=None):
    """Compute all evaluation metrics"""
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    return {
        'auc_roc': roc_auc_score(y_true, y_pred_prob),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'mae': mean_absolute_error(y_true, y_pred_prob),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred_prob)),
        'r2': r2_score(y_true, y_pred_prob),
    }

# ============================================================================
# 4. TRAIN MODELS & COMPUTE STATISTICS
# ============================================================================
print("\n4. TRAINING MODELS & COMPUTING STATISTICS...\n")

results_by_fold = {
    'Logistic': {'folds': []},
    'LASSO': {'folds': []},
    'Ridge': {'folds': []},
    'RF': {'folds': []},
}

results_summary = []

fold_idx = 0
for train_idx, test_idx in cv.split(X_tfidf, y):
    fold_idx += 1
    print(f"   Fold {fold_idx}/5...")

    X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ========== LOGISTIC REGRESSION ==========
    lr = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)

    y_train_pred_prob = lr.predict_proba(X_train)[:, 1]
    y_test_pred_prob = lr.predict_proba(X_test)[:, 1]

    train_metrics = compute_all_metrics(y_train, y_train_pred_prob)
    test_metrics = compute_all_metrics(y_test, y_test_pred_prob)

    results_by_fold['Logistic']['folds'].append({
        'train_auc': train_metrics['auc_roc'],
        'test_auc': test_metrics['auc_roc'],
        'metrics': test_metrics,
    })

    # ========== LASSO ==========
    lasso = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=5, l1_ratios=(1,),
                                solver='liblinear', max_iter=2000, class_weight='balanced',
                                scoring='roc_auc', random_state=SEED, use_legacy_attributes=False)
    lasso.fit(X_train, y_train)

    y_train_pred_prob = lasso.predict_proba(X_train)[:, 1]
    y_test_pred_prob = lasso.predict_proba(X_test)[:, 1]

    train_metrics = compute_all_metrics(y_train, y_train_pred_prob)
    test_metrics = compute_all_metrics(y_test, y_test_pred_prob)

    results_by_fold['LASSO']['folds'].append({
        'train_auc': train_metrics['auc_roc'],
        'test_auc': test_metrics['auc_roc'],
        'metrics': test_metrics,
        'model': lasso,
        'coefs': lasso.coef_[0],
    })

    # ========== RIDGE ==========
    ridge = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=5, l1_ratios=(0,),
                                solver='liblinear', max_iter=2000, class_weight='balanced',
                                scoring='roc_auc', random_state=SEED, use_legacy_attributes=False)
    ridge.fit(X_train, y_train)

    y_train_pred_prob = ridge.predict_proba(X_train)[:, 1]
    y_test_pred_prob = ridge.predict_proba(X_test)[:, 1]

    train_metrics = compute_all_metrics(y_train, y_train_pred_prob)
    test_metrics = compute_all_metrics(y_test, y_test_pred_prob)

    results_by_fold['Ridge']['folds'].append({
        'train_auc': train_metrics['auc_roc'],
        'test_auc': test_metrics['auc_roc'],
        'metrics': test_metrics,
    })

    # ========== RANDOM FOREST ==========
    X_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    rf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                               random_state=SEED, class_weight='balanced', n_jobs=-1)
    rf.fit(X_dense, y_train)

    y_train_pred_prob = rf.predict_proba(X_dense)[:, 1]
    y_test_pred_prob = rf.predict_proba(X_test_dense)[:, 1]

    train_metrics = compute_all_metrics(y_train, y_train_pred_prob)
    test_metrics = compute_all_metrics(y_test, y_test_pred_prob)

    results_by_fold['RF']['folds'].append({
        'train_auc': train_metrics['auc_roc'],
        'test_auc': test_metrics['auc_roc'],
        'metrics': test_metrics,
    })

print("\n✓ Training complete")

# ============================================================================
# 5. AGGREGATE RESULTS: PER-FOLD BREAKDOWN (TEST 4)
# ============================================================================
print("\n5. COMPUTING PER-FOLD BREAKDOWN...")

for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
    fold_aucs = [fold['test_auc'] for fold in results_by_fold[model_name]['folds']]

    print(f"\n   {model_name}:")
    for i, auc in enumerate(fold_aucs, 1):
        print(f"      Fold {i}: AUC = {auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"      Mean:  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    results_by_fold[model_name]['mean_auc'] = mean_auc
    results_by_fold[model_name]['std_auc'] = std_auc
    results_by_fold[model_name]['fold_aucs'] = fold_aucs

# ============================================================================
# 6. BIAS-VARIANCE ANALYSIS (TEST 6)
# ============================================================================
print("\n6. BIAS-VARIANCE ANALYSIS (Train vs Test)...")

bias_var_data = []
for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
    train_aucs = [fold['train_auc'] for fold in results_by_fold[model_name]['folds']]
    test_aucs = [fold['test_auc'] for fold in results_by_fold[model_name]['folds']]

    mean_train = np.mean(train_aucs)
    mean_test = np.mean(test_aucs)
    overfitting = mean_train - mean_test

    bias_var_data.append({
        'Model': model_name,
        'Train AUC': f"{mean_train:.4f}",
        'Test AUC': f"{mean_test:.4f}",
        'Overfitting': f"{overfitting:.4f}",
        'Status': "HIGH" if overfitting > 0.030 else "MILD" if overfitting > 0.010 else "MINIMAL"
    })

    print(f"\n   {model_name}:")
    print(f"      Train AUC: {mean_train:.4f}")
    print(f"      Test AUC:  {mean_test:.4f}")
    print(f"      Overfitting (train - test): {overfitting:.4f} ({bias_var_data[-1]['Status']})")

bias_var_df = pd.DataFrame(bias_var_data)

# ============================================================================
# 7. COMPREHENSIVE METRICS TABLE (TESTS 1, 2, 3, 5)
# ============================================================================
print("\n7. COMPUTING COMPREHENSIVE METRICS TABLE...")

model_results = []

# Train final models on full data for metrics
X_tfidf_dense = X_tfidf.toarray()

lr_final = LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')
lr_final.fit(X_tfidf, y)
lr_final_proba = lr_final.predict_proba(X_tfidf)[:, 1]

lasso_final = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=5, l1_ratios=(1,),
                                   solver='liblinear', max_iter=2000, class_weight='balanced',
                                   scoring='roc_auc', random_state=SEED, use_legacy_attributes=False)
lasso_final.fit(X_tfidf, y)
lasso_final_proba = lasso_final.predict_proba(X_tfidf)[:, 1]

ridge_final = LogisticRegressionCV(Cs=np.logspace(-3, 3, 10), cv=5, l1_ratios=(0,),
                                  solver='liblinear', max_iter=2000, class_weight='balanced',
                                  scoring='roc_auc', random_state=SEED, use_legacy_attributes=False)
ridge_final.fit(X_tfidf, y)
ridge_final_proba = ridge_final.predict_proba(X_tfidf)[:, 1]

rf_final = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                 random_state=SEED, class_weight='balanced', n_jobs=-1)
rf_final.fit(X_tfidf_dense, y)
rf_final_proba = rf_final.predict_proba(X_tfidf_dense)[:, 1]

# ========== TEST 1: P-VALUES FOR LOGISTIC ==========
print("\n   Computing p-values for Logistic Regression...")
lr_coefs = lr_final.coef_[0]
lr_se = compute_logistic_standard_errors(X_tfidf, y, lr_coefs)
lr_z_scores = lr_coefs / lr_se
lr_p_values = 2 * (1 - stats.norm.cdf(np.abs(lr_z_scores)))
lr_significant = np.sum(lr_p_values < 0.05)

print(f"      Logistic: {lr_significant}/{len(lr_coefs)} coefficients significant (p<0.05)")

# ========== TEST 1: P-VALUES FOR LASSO ==========
print("   Computing p-values for LASSO...")
lasso_coefs = lasso_final.coef_[0]
non_zero_idx = lasso_coefs != 0
non_zero_count = np.sum(non_zero_idx)

if non_zero_count > 0:
    X_tfidf_nz = X_tfidf[:, non_zero_idx]
    lasso_se = compute_logistic_standard_errors(X_tfidf_nz, y, lasso_coefs[non_zero_idx])
    lasso_z_scores = lasso_coefs[non_zero_idx] / lasso_se
    lasso_p_values = 2 * (1 - stats.norm.cdf(np.abs(lasso_z_scores)))
    lasso_significant = np.sum(lasso_p_values < 0.05)
    print(f"      LASSO: {lasso_significant}/{non_zero_count} non-zero coefficients significant (p<0.05)")
else:
    lasso_significant = 0

# ========== TEST 1: P-VALUES FOR RIDGE ==========
print("   Computing p-values for Ridge...")
ridge_coefs = ridge_final.coef_[0]
ridge_se = compute_logistic_standard_errors(X_tfidf, y, ridge_coefs)
ridge_z_scores = ridge_coefs / ridge_se
ridge_p_values = 2 * (1 - stats.norm.cdf(np.abs(ridge_z_scores)))
ridge_significant = np.sum(ridge_p_values < 0.05)

print(f"      Ridge: {ridge_significant}/{len(ridge_coefs)} coefficients significant (p<0.05)")

# Print top significant features (research question: which words predict passage?)
print("\n   Top significant LASSO features by coefficient magnitude:")
if non_zero_count > 0:
    lasso_coef_all = lasso_final.coef_[0]
    nz_names = feature_names[non_zero_idx]
    nz_coefs = lasso_coef_all[non_zero_idx]
    sort_idx = np.argsort(np.abs(nz_coefs))[::-1][:20]
    for rank, i in enumerate(sort_idx, 1):
        direction = "PASS" if nz_coefs[i] > 0 else "FAIL"
        sig_marker = "***" if lasso_p_values[i] < 0.001 else "**" if lasso_p_values[i] < 0.01 else "*" if lasso_p_values[i] < 0.05 else ""
        print(f"      {rank:2d}. {nz_names[i]:20s}  coef={nz_coefs[i]:+.4f}  p={lasso_p_values[i]:.4f} {sig_marker}  → predicts {direction}")

# RF top features
rf_importances = rf_final.feature_importances_
rf_top_idx = np.argsort(rf_importances)[::-1][:10]
print("\n   Top RF feature importances:")
for rank, i in enumerate(rf_top_idx, 1):
    print(f"      {rank:2d}. {feature_names[i]:20s}  importance={rf_importances[i]:.4f}")

# ========== TEST 2: CONFIDENCE INTERVALS FOR AUC ==========
print("\n   Computing 95% confidence intervals for AUC...")

for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
    fold_aucs = results_by_fold[model_name]['fold_aucs']
    mean_auc = np.mean(fold_aucs)
    se_auc = np.std(fold_aucs) / np.sqrt(len(fold_aucs))
    ci_lower = mean_auc - 1.96 * se_auc
    ci_upper = mean_auc + 1.96 * se_auc

    results_by_fold[model_name]['auc_ci'] = (ci_lower, ci_upper)
    print(f"      {model_name}: AUC = {mean_auc:.4f} [95% CI: {ci_lower:.4f}-{ci_upper:.4f}]")

# ========== TEST 5: REGRESSION METRICS (MAE, RMSE, R²) ==========
print("\n   Computing regression metrics (MAE, RMSE, R²)...")

models_for_metrics = [
    ('Logistic', lr_final_proba, 'cv'),
    ('LASSO', lasso_final_proba, 'cv'),
    ('Ridge', ridge_final_proba, 'cv'),
    ('RF', rf_final_proba, 'cv'),
]

for model_name, y_pred_prob, eval_type in models_for_metrics:
    mae = mean_absolute_error(y, y_pred_prob)
    rmse = np.sqrt(mean_squared_error(y, y_pred_prob))
    r2 = r2_score(y, y_pred_prob)

    results_by_fold[model_name]['mae'] = mae
    results_by_fold[model_name]['rmse'] = rmse
    results_by_fold[model_name]['r2'] = r2

    print(f"      {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# ========== COMPILE COMPREHENSIVE TABLE ==========
comprehensive_results = []

for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
    fold_aucs = results_by_fold[model_name]['fold_aucs']
    auc_ci = results_by_fold[model_name]['auc_ci']

    comprehensive_results.append({
        'Model': model_name,
        'AUC-ROC': f"{np.mean(fold_aucs):.4f}",
        'AUC 95% CI': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
        'Accuracy': f"{results_by_fold[model_name]['folds'][-1]['metrics']['accuracy']:.4f}",
        'Precision': f"{results_by_fold[model_name]['folds'][-1]['metrics']['precision']:.4f}",
        'Recall': f"{results_by_fold[model_name]['folds'][-1]['metrics']['recall']:.4f}",
        'F1': f"{results_by_fold[model_name]['folds'][-1]['metrics']['f1']:.4f}",
        'MAE': f"{results_by_fold[model_name]['mae']:.4f}",
        'RMSE': f"{results_by_fold[model_name]['rmse']:.4f}",
        'R²': f"{results_by_fold[model_name]['r2']:.4f}",
    })

comp_df = pd.DataFrame(comprehensive_results)

# ============================================================================
# 8. MODEL COMPARISON: PAIRED T-TEST (TEST 3)
# ============================================================================
print("\n8. PAIRED T-TEST FOR MODEL COMPARISON...")

model_pairs = [
    ('RF', 'Logistic'),
    ('RF', 'LASSO'),
    ('RF', 'Ridge'),
]

t_test_results = []

for m1, m2 in model_pairs:
    auc_m1 = np.array(results_by_fold[m1]['fold_aucs'])
    auc_m2 = np.array(results_by_fold[m2]['fold_aucs'])

    diff = auc_m1 - auc_m2
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))

    t_stat = mean_diff / se_diff if se_diff > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=4))  # 5 folds = df=4

    sig = "✓✓✓" if p_val < 0.001 else "✓✓" if p_val < 0.01 else "✓" if p_val < 0.05 else "✗"

    t_test_results.append({
        'Comparison': f"{m1} vs {m2}",
        'Mean Diff AUC': f"{mean_diff:.4f}",
        'SE': f"{se_diff:.4f}",
        'T-Stat': f"{t_stat:.2f}",
        'P-Value': f"{p_val:.4f}",
        'Significant': sig,
    })

    print(f"\n   {m1} vs {m2}:")
    print(f"      Mean difference: {mean_diff:.4f}")
    print(f"      T-statistic: {t_stat:.2f}")
    print(f"      P-value: {p_val:.4f} {sig}")

t_test_df = pd.DataFrame(t_test_results)

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n9. SAVING RESULTS...")

os.makedirs('results/statistical_analysis', exist_ok=True)

comp_df.to_csv('results/statistical_analysis/comprehensive_metrics_table.csv', index=False)
bias_var_df.to_csv('results/statistical_analysis/bias_variance_analysis.csv', index=False)
t_test_df.to_csv('results/statistical_analysis/model_comparison_t_tests.csv', index=False)

print("   ✓ Saved comprehensive_metrics_table.csv")
print("   ✓ Saved bias_variance_analysis.csv")
print("   ✓ Saved model_comparison_t_tests.csv")

# ============================================================================
# 10. PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: STATISTICAL RIGOR TESTS COMPLETED")
print("="*80)

print("\n✓ TEST 1: P-VALUES FOR COEFFICIENTS (Linear Models Only)")
print(f"   Logistic: {lr_significant}/2000 significant coefficients")
print(f"   LASSO: {lasso_significant}/{non_zero_count} significant non-zero coefficients")
print(f"   Ridge: {ridge_significant}/2000 significant coefficients")

print("\n✓ TEST 2: 95% CONFIDENCE INTERVALS FOR AUC")
print("   (See comprehensive_metrics_table.csv)")

print("\n✓ TEST 3: PAIRED T-TEST FOR MODEL COMPARISON")
for _, row in t_test_df.iterrows():
    print(f"   {row['Comparison']}: t={row['T-Stat']}, p={row['P-Value']} {row['Significant']}")

print("\n✓ TEST 4: PER-FOLD BREAKDOWN (Mean ± Std)")
for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
    mean = results_by_fold[model_name]['mean_auc']
    std = results_by_fold[model_name]['std_auc']
    print(f"   {model_name}: AUC = {mean:.4f} ± {std:.4f}")

print("\n✓ TEST 5: REGRESSION METRICS (MAE, RMSE, R²)")
print("   (See comprehensive_metrics_table.csv)")

print("\n✓ TEST 6: BIAS-VARIANCE ANALYSIS (Train vs Test)")
for _, row in bias_var_df.iterrows():
    print(f"   {row['Model']}: Train={row['Train AUC']}, Test={row['Test AUC']}, Status={row['Status']}")

print("\n" + "="*80)
print("All statistical tests completed successfully!")
print("="*80)

# Save summary as text
with open('results/statistical_analysis/STATISTICAL_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("STATISTICAL RIGOR ANALYSIS SUMMARY\n")
    f.write("Congressional Bill Passage Prediction - Phase 1 Enhancement\n")
    f.write("="*80 + "\n\n")

    f.write("TEST 1: P-VALUES FOR COEFFICIENTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Logistic: {lr_significant}/2000 significant coefficients (p<0.05)\n")
    f.write(f"LASSO: {lasso_significant}/{non_zero_count} significant non-zero coefficients (p<0.05)\n")
    f.write(f"Ridge: {ridge_significant}/2000 significant coefficients (p<0.05)\n\n")

    f.write("TEST 2: 95% CONFIDENCE INTERVALS FOR AUC\n")
    f.write("-"*80 + "\n")
    for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
        auc_ci = results_by_fold[model_name]['auc_ci']
        mean = results_by_fold[model_name]['mean_auc']
        f.write(f"{model_name}: AUC = {mean:.4f} [95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f}]\n")
    f.write("\n")

    f.write("TEST 3: PAIRED T-TEST FOR MODEL COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(t_test_df.to_string(index=False))
    f.write("\n\n")

    f.write("TEST 4: PER-FOLD BREAKDOWN\n")
    f.write("-"*80 + "\n")
    for model_name in ['Logistic', 'LASSO', 'Ridge', 'RF']:
        fold_aucs = results_by_fold[model_name]['fold_aucs']
        f.write(f"\n{model_name}:\n")
        for i, auc in enumerate(fold_aucs, 1):
            f.write(f"   Fold {i}: {auc:.4f}\n")
        mean = np.mean(fold_aucs)
        std = np.std(fold_aucs)
        f.write(f"   Mean:   {mean:.4f} ± {std:.4f}\n")
    f.write("\n")

    f.write("TEST 5: REGRESSION METRICS\n")
    f.write("-"*80 + "\n")
    f.write(comp_df.to_string(index=False))
    f.write("\n\n")

    f.write("TEST 6: BIAS-VARIANCE ANALYSIS\n")
    f.write("-"*80 + "\n")
    f.write(bias_var_df.to_string(index=False))
    f.write("\n\n")

    f.write("FEATURE ANALYSIS: TOP SIGNIFICANT PREDICTORS\n")
    f.write("-"*80 + "\n")
    f.write("Research Question: Which words predict bill passage/failure?\n\n")
    if non_zero_count > 0:
        f.write("Top LASSO features (by coefficient magnitude, with p-values):\n")
        nz_names = feature_names[non_zero_idx]
        nz_coefs = lasso_final.coef_[0][non_zero_idx]
        sort_idx = np.argsort(np.abs(nz_coefs))[::-1][:20]
        for rank, i in enumerate(sort_idx, 1):
            direction = "PASS" if nz_coefs[i] > 0 else "FAIL"
            sig = "***" if lasso_p_values[i] < 0.001 else "**" if lasso_p_values[i] < 0.01 else "*" if lasso_p_values[i] < 0.05 else "n.s."
            f.write(f"  {rank:2d}. {nz_names[i]:20s}  coef={nz_coefs[i]:+.4f}  p={lasso_p_values[i]:.4f} {sig:4s}  → predicts {direction}\n")
    f.write("\nTop RF Feature Importances:\n")
    rf_importances = rf_final.feature_importances_
    rf_top_idx = np.argsort(rf_importances)[::-1][:10]
    for rank, i in enumerate(rf_top_idx, 1):
        f.write(f"  {rank:2d}. {feature_names[i]:20s}  importance={rf_importances[i]:.4f}\n")

print("✓ Saved STATISTICAL_SUMMARY.txt")
