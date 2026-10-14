"""
CloudGuard-Water: Phase 4 — Detailed Evaluation, Validation & Model Selection
Author: CloudGuard Production Engineering
Course: Machine Learning for Cloud Services

This script conducts rigorous 5-fold Stratified Cross-Validation, Dummy baseline comparison,
investigation of near-perfect scores, error boundary analysis, class weighting evaluation,
and produces validation reports and publication-quality figures.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

# ---------------------------------------------------------------------------
# Global Configuration & Styling
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid", palette="muted")

FEATURE_COLS = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
TARGET_COL = 'Label'
TARGET_MAP = {'Safe': 0, 'Unsafe': 1}

# ---------------------------------------------------------------------------
# Helper: Locate Dataset
# ---------------------------------------------------------------------------
def find_dataset() -> Path:
    candidates = [
        DATA_DIR / "water_quality_dataset1.csv",
        DATA_DIR / "water_quality_dataset1_original.csv",
        Path("data/water_quality_dataset1.csv"),
        Path("../data/water_quality_dataset1.csv")
    ]
    for p in candidates:
        if p.exists():
            return p
    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        for name in ["water_quality_dataset1.csv", "water_quality_dataset1_original.csv"]:
            target = parent / "data" / name
            if target.exists():
                return target
    raise FileNotFoundError("Could not find water_quality_dataset1.csv")

# ---------------------------------------------------------------------------
# 1. 5-Fold Stratified Cross-Validation
# ---------------------------------------------------------------------------
def run_stratified_cv(X, y):
    print("\n" + "="*70)
    print("1. RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, criterion='gini', random_state=42, n_jobs=-1),
        'Dummy Classifier (Most Frequent)': DummyClassifier(strategy='most_frequent')
    }
    
    cv_records = []
    
    for name, model in models.items():
        print(f"  Evaluating: {name} across 5 folds...")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocessing fits strictly on the training fold
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            
            acc = float(accuracy_score(y_val, y_pred))
            rec_safe = float(recall_score(y_val, y_pred, pos_label=0))
            rec_unsafe = float(recall_score(y_val, y_pred, pos_label=1))
            prec_safe = float(precision_score(y_val, y_pred, pos_label=0))
            prec_unsafe = float(precision_score(y_val, y_pred, pos_label=1))
            f1_s = float(f1_score(y_val, y_pred, pos_label=0))
            f1_u = float(f1_score(y_val, y_pred, pos_label=1))
            macro_f1 = float(f1_score(y_val, y_pred, average='macro'))
            
            fold_metrics.append({
                'Fold': fold,
                'Accuracy': acc,
                'Macro_F1': macro_f1,
                'Recall_Safe': rec_safe,
                'Recall_Unsafe': rec_unsafe,
                'Precision_Safe': prec_safe,
                'Precision_Unsafe': prec_unsafe,
                'F1_Safe': f1_s,
                'F1_Unsafe': f1_u
            })
            
        fold_df = pd.DataFrame(fold_metrics)
        mean_s = fold_df.mean(numeric_only=True)
        std_s = fold_df.std(numeric_only=True)
        
        cv_records.append({
            'Model': name,
            'Accuracy_Mean': mean_s['Accuracy'],
            'Accuracy_Std': std_s['Accuracy'],
            'Macro_F1_Mean': mean_s['Macro_F1'],
            'Macro_F1_Std': std_s['Macro_F1'],
            'Recall_Safe_Mean': mean_s['Recall_Safe'],
            'Recall_Safe_Std': std_s['Recall_Safe'],
            'Recall_Unsafe_Mean': mean_s['Recall_Unsafe'],
            'Recall_Unsafe_Std': std_s['Recall_Unsafe'],
            'Precision_Safe_Mean': mean_s['Precision_Safe'],
            'Precision_Safe_Std': std_s['Precision_Safe'],
            'Precision_Unsafe_Mean': mean_s['Precision_Unsafe'],
            'Precision_Unsafe_Std': std_s['Precision_Unsafe']
        })
        
        print(f"    Mean Accuracy : {mean_s['Accuracy']:.6f} (+/- {std_s['Accuracy']:.6f})")
        print(f"    Mean Macro F1 : {mean_s['Macro_F1']:.6f} (+/- {std_s['Macro_F1']:.6f})")
        print(f"    Safe Recall   : {mean_s['Recall_Safe']:.6f} (+/- {std_s['Recall_Safe']:.6f})")
        print(f"    Unsafe Recall : {mean_s['Recall_Unsafe']:.6f} (+/- {std_s['Recall_Unsafe']:.6f})")
        
    return pd.DataFrame(cv_records)

# ---------------------------------------------------------------------------
# 2. Test Set Evaluation & Error Breakdown
# ---------------------------------------------------------------------------
def evaluate_test_set_and_errors(X_train, X_test, y_train, y_test):
    print("\n" + "="*70)
    print("2. TEST SET EVALUATION, CONFUSION MATRICES & ERROR BREAKDOWN")
    print("="*70)
    
    models = {
        'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, criterion='gini', random_state=42, n_jobs=-1),
        'Logistic Regression (Balanced)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]),
        'Decision Tree (Balanced)': DecisionTreeClassifier(max_depth=5, criterion='gini', class_weight='balanced', random_state=42),
        'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, max_depth=8, criterion='gini', class_weight='balanced', random_state=42, n_jobs=-1)
    }
    
    test_results = []
    trained_dict = {}
    y_test_list = list(y_test)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_dict[name] = model
        
        y_pred = model.predict(X_test)
        y_pred_list = [int(p) for p in y_pred]
        
        acc = float(accuracy_score(y_test, y_pred))
        rec_safe = float(recall_score(y_test, y_pred, pos_label=0))
        rec_unsafe = float(recall_score(y_test, y_pred, pos_label=1))
        prec_safe = float(precision_score(y_test, y_pred, pos_label=0))
        prec_unsafe = float(precision_score(y_test, y_pred, pos_label=1))
        macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
        weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
        
        cm = confusion_matrix(y_test_list, y_pred_list)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        
        test_results.append({
            'Model': name,
            'Accuracy': acc,
            'Macro_F1': macro_f1,
            'Weighted_F1': weighted_f1,
            'Safe_Precision': prec_safe,
            'Safe_Recall': rec_safe,
            'Unsafe_Precision': prec_unsafe,
            'Unsafe_Recall': rec_unsafe,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })
        
        print(f"  {name:<32}: Acc={acc:.6f} | Macro F1={macro_f1:.6f} | TN={tn:>5}, FP={fp:>2}, FN={fn:>2}, TP={tp:>5}")
        
    test_df = pd.DataFrame(test_results)
    return test_df, trained_dict

# ---------------------------------------------------------------------------
# 3. Investigation into Near-Perfect Scores
# ---------------------------------------------------------------------------
def investigate_near_perfect_scores(df: pd.DataFrame):
    print("\n" + "="*70)
    print("3. SCIENTIFIC INVESTIGATION INTO NEAR-PERFECT ACCURACY (~99.99%)")
    print("="*70)
    
    # Check physical threshold rule
    # Safe water: TDS <= 211.0 ppm and Turbidity <= 4.45 NTU and 6.5 <= pH <= 7.8
    safe_ground_truth = (df['Label'] == 'Safe')
    rule_predicted_safe = (
        (df['TDS_ppm'] <= 211.0) & 
        (df['Turbidity_NTU'] <= 4.45) & 
        (df['pH'] >= 6.5) & 
        (df['pH'] <= 7.8)
    )
    
    exact_matches = int((safe_ground_truth == rule_predicted_safe).sum())
    mismatches = int((safe_ground_truth != rule_predicted_safe).sum())
    match_pct = (exact_matches / len(df)) * 100.0
    
    print(f"  Total Observations in Dataset : {len(df):,}")
    print(f"  Exact Matches with Simple Rule: {exact_matches:,} ({match_pct:.4f}%)")
    print(f"  Boundary Edge Mismatches       : {mismatches} ({100.0 - match_pct:.4f}%)")
    
    edge_df = df[safe_ground_truth != rule_predicted_safe].copy()
    print("\n  Summary of Boundary Mismatches:")
    print(edge_df[['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C', 'Label']].to_string())
    
    return edge_df

# ---------------------------------------------------------------------------
# 4. Generate Publication-Quality Figures
# ---------------------------------------------------------------------------
def generate_validation_plots(cv_df, test_df, trained_dict, X_test, y_test, df):
    print("\n" + "="*70)
    print("4. GENERATING PUBLICATION-QUALITY VALIDATION FIGURES")
    print("="*70)
    
    # -----------------------------------------------------------------------
    # Figure 1: 5-Fold Stratified Cross-Validation Stability
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    main_cv = cv_df[cv_df['Model'] != 'Dummy Classifier (Most Frequent)'].copy()
    models_list = list(main_cv['Model'])
    x_pos = np.arange(len(models_list))
    width = 0.22
    
    # Subplot A: Accuracy & Macro F1
    ax1 = axes[0]
    ax1.bar(x_pos - width/2, list(main_cv['Accuracy_Mean']), width, yerr=list(main_cv['Accuracy_Std']),
            capsize=5, label='Mean Accuracy', color='#1c7ed6', edgecolor='#333')
    ax1.bar(x_pos + width/2, list(main_cv['Macro_F1_Mean']), width, yerr=list(main_cv['Macro_F1_Std']),
            capsize=5, label='Mean Macro F1', color='#37b24d', edgecolor='#333')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_list, fontsize=10, fontweight='bold')
    ax1.set_ylim(0.998, 1.0002)
    ax1.set_ylabel("Metric Score", fontsize=11)
    ax1.set_title("Cross-Validation Stability: Accuracy & Macro F1 (5-Fold)", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', frameon=True)
    
    # Subplot B: Safe Recall vs Unsafe Recall
    ax2 = axes[1]
    ax2.bar(x_pos - width/2, list(main_cv['Recall_Safe_Mean']), width, yerr=list(main_cv['Recall_Safe_Std']),
            capsize=5, label='Safe Recall (Class 0)', color='#4263eb', edgecolor='#333')
    ax2.bar(x_pos + width/2, list(main_cv['Recall_Unsafe_Mean']), width, yerr=list(main_cv['Recall_Unsafe_Std']),
            capsize=5, label='Unsafe Recall (Class 1)', color='#f03e3e', edgecolor='#333')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_list, fontsize=10, fontweight='bold')
    ax2.set_ylim(0.998, 1.0002)
    ax2.set_ylabel("Recall Score", fontsize=11)
    ax2.set_title("Cross-Validation Stability: Class-Specific Recalls", fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', frameon=True)
    
    plt.tight_layout()
    cv_fig_path = FIGURES_DIR / "cv_stability_comparison.png"
    plt.savefig(cv_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cv_fig_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Dummy Baseline vs Trained Models
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    eval_models = ['Dummy (Most Frequent)', 'Logistic Regression', 'Decision Tree', 'Random Forest']
    sub_test = test_df[test_df['Model'].isin(eval_models)].copy()
    
    x_indices = np.arange(len(eval_models))
    w = 0.22
    
    ax.bar(x_indices - w, list(sub_test['Accuracy']), w, label='Accuracy', color='#1c7ed6', edgecolor='#333')
    ax.bar(x_indices, list(sub_test['Macro_F1']), w, label='Macro F1', color='#37b24d', edgecolor='#333')
    ax.bar(x_indices + w, list(sub_test['Safe_Recall']), w, label='Safe Recall (Potable)', color='#f59f00', edgecolor='#333')
    
    ax.set_xticks(x_indices)
    ax.set_xticklabels(eval_models, fontsize=10, fontweight='bold')
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("ML Discriminative Superiority Over Majority-Class Baseline (20,160 Test Samples)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc='upper left', frameon=True)
    
    # Add text annotations on Dummy bars
    ax.text(0 - w, 0.72, "70.0%", ha='center', fontsize=9, fontweight='bold')
    ax.text(0, 0.43, "41.2%", ha='center', fontsize=9, fontweight='bold')
    ax.text(0 + w, 0.03, "0.0%!", ha='center', color='red', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    dummy_fig_path = FIGURES_DIR / "dummy_vs_models_comparison.png"
    plt.savefig(dummy_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dummy_fig_path}")

    # -----------------------------------------------------------------------
    # Figure 3: ROC & Precision-Recall Curves
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot A: ROC Curve
    ax_roc = axes[0]
    ax_pr = axes[1]
    
    curves_models = {
        'Logistic Regression': (trained_dict['Logistic Regression'], '#1c7ed6'),
        'Decision Tree': (trained_dict['Decision Tree'], '#2b8a3e'),
        'Random Forest': (trained_dict['Random Forest'], '#d9480f')
    }
    
    for name, (m, col) in curves_models.items():
        if hasattr(m, "predict_proba"):
            probs = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.5f})", color=col, linewidth=2)
            
            prec, rec, _ = precision_recall_curve(y_test, probs)
            ap = average_precision_score(y_test, probs)
            ax_pr.plot(rec, prec, label=f"{name} (AP = {ap:.5f})", color=col, linewidth=2)
            
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC=0.50)')
    ax_roc.set_xlim([-0.001, 0.05])
    ax_roc.set_ylim([0.995, 1.001])
    ax_roc.set_xlabel("False Positive Rate (Safe predicted as Unsafe)", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Unsafe Sensitivity)", fontsize=10)
    ax_roc.set_title("ROC Curves (Zoomed Inset: FPR [0, 0.05])", fontsize=12, fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=True)
    
    ax_pr.set_xlim([0.995, 1.001])
    ax_pr.set_ylim([0.995, 1.001])
    ax_pr.set_xlabel("Recall (Unsafe Sensitivity)", fontsize=10)
    ax_pr.set_ylabel("Precision (Unsafe Detection)", fontsize=10)
    ax_pr.set_title("Precision-Recall Curves (Zoomed Inset)", fontsize=12, fontweight='bold')
    ax_pr.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    roc_fig_path = FIGURES_DIR / "model_roc_pr_curves.png"
    plt.savefig(roc_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {roc_fig_path}")

    # -----------------------------------------------------------------------
    # Figure 4: Error Boundary Scatter (TDS vs Turbidity with Error Cases)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    sample_df = df.sample(n=3000, random_state=42)
    safe_pts = sample_df[sample_df['Label'] == 'Safe']
    unsafe_pts = sample_df[sample_df['Label'] == 'Unsafe']
    
    ax.scatter(safe_pts['TDS_ppm'], safe_pts['Turbidity_NTU'], 
               color='#2b8a3e', alpha=0.35, s=15, label='Safe Ground Truth (Sample)')
    ax.scatter(unsafe_pts['TDS_ppm'], unsafe_pts['Turbidity_NTU'], 
               color='#c92a2a', alpha=0.25, s=15, label='Unsafe Ground Truth (Sample)')
    
    # Highlight the test set misclassifications
    dt_model = trained_dict['Decision Tree']
    y_test_pred = dt_model.predict(X_test)
    mis_idx = X_test[y_test != y_test_pred].index
    
    if len(mis_idx) > 0:
        mis_pts = df.loc[mis_idx]
        ax.scatter(mis_pts['TDS_ppm'], mis_pts['Turbidity_NTU'],
                   color='#ffd43b', edgecolor='black', s=120, marker='*',
                   label='Test Misclassifications (Borderline)', zorder=5)
        for _, row in mis_pts.iterrows():
            ax.annotate(f"{row['Label']} (TDS={row['TDS_ppm']:.1f}, Turb={row['Turbidity_NTU']:.2f})",
                        (row['TDS_ppm'], row['Turbidity_NTU']),
                        xytext=(15, 10), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff9db", ec="#f59f00", lw=1),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#f59f00"))

    # Plot decision boundary lines
    ax.axvline(211.0, color='#1c7ed6', linestyle='--', linewidth=1.5, label='TDS Threshold (~211 ppm)')
    ax.axhline(4.45, color='#e03131', linestyle='--', linewidth=1.5, label='Turbidity Threshold (~4.45 NTU)')
    
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.5, 15)
    ax.set_xlabel("TDS (ppm)", fontsize=11)
    ax.set_ylabel("Turbidity (NTU)", fontsize=11)
    ax.set_title("Physical Separation & Boundary Misclassifications (TDS vs Turbidity)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True, fontsize=8.5)
    
    plt.tight_layout()
    scatter_fig_path = FIGURES_DIR / "error_boundary_scatter.png"
    plt.savefig(scatter_fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {scatter_fig_path}")

# ---------------------------------------------------------------------------
# 5. Save Validation Results to CSV
# ---------------------------------------------------------------------------
def save_validation_results(cv_df: pd.DataFrame, test_df: pd.DataFrame):
    out_csv = REPORTS_DIR / "phase4_validation_results.csv"
    
    # Merge test metrics with CV metrics where applicable
    merged = test_df.copy()
    merged['CV_Accuracy_Mean'] = np.nan
    merged['CV_Accuracy_Std'] = np.nan
    merged['CV_Macro_F1_Mean'] = np.nan
    merged['CV_Macro_F1_Std'] = np.nan
    
    cv_map = cv_df.set_index('Model')
    for idx, row in merged.iterrows():
        m_name = str(row['Model'])
        if m_name in cv_map.index:
            merged.at[idx, 'CV_Accuracy_Mean'] = cv_map.loc[m_name, 'Accuracy_Mean']
            merged.at[idx, 'CV_Accuracy_Std'] = cv_map.loc[m_name, 'Accuracy_Std']
            merged.at[idx, 'CV_Macro_F1_Mean'] = cv_map.loc[m_name, 'Macro_F1_Mean']
            merged.at[idx, 'CV_Macro_F1_Std'] = cv_map.loc[m_name, 'Macro_F1_Std']
            
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved combined validation results: {out_csv}")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    data_file = find_dataset()
    print(f"Loading authoritative dataset from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].map(TARGET_MAP)
    
    # Partition matching Phase 3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 1. Cross Validation
    cv_df = run_stratified_cv(X, y)
    
    # 2. Test Set Evaluation & Error Analysis
    test_df, trained_dict = evaluate_test_set_and_errors(X_train, X_test, y_train, y_test)
    
    # 3. Investigation
    edge_df = investigate_near_perfect_scores(df)
    
    # 4. Plots
    generate_validation_plots(cv_df, test_df, trained_dict, X_test, y_test, df)
    
    # 5. Save Results CSV
    save_validation_results(cv_df, test_df)
    
    print("\n" + "="*70)
    print("PHASE 4 DATA EXPERIMENTS & VALIDATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
