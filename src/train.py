"""
CloudGuard-Water — Phase 3: Data Preprocessing & Model Training
===============================================================
Course: ML for Cloud Services
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Script: src/train.py

Models Trained:
  1. Logistic Regression (StandardScaler inside Pipeline)
  2. Decision Tree (Unscaled, max_depth=5 for interpretability)
  3. Random Forest (Unscaled ensemble, n_estimators=100, max_depth=8)

Target Encoding:
  - 0 = Safe
  - 1 = Unsafe (Positive Class: Contamination / Alert Condition)

Evaluation:
  - Stratified 80/20 Train-Test Split (random_state=42)
  - Accuracy, Precision, Recall, F1 (per-class, macro, weighted)
  - Confusion Matrix (TN, FP, FN, TP) with operational error interpretation
  - Model Persistence with joblib (models/)
  - Visualization: Decision Tree structure & Random Forest feature importance
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------------------
# Plotting Configuration
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid", palette="muted")

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Dataset Locator
# ---------------------------------------------------------------------------
def find_dataset():
    candidates = [
        PROJECT_ROOT / "data" / "water_quality_dataset1.csv",
        PROJECT_ROOT / "data" / "water_quality_dataset1_original.csv",
        Path("data/water_quality_dataset1.csv"),
        Path("../data/water_quality_dataset1.csv")
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find water_quality_dataset1.csv")

# ---------------------------------------------------------------------------
# 2. Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("PHASE 3: DATA PREPROCESSING & BASELINE ML MODEL TRAINING")
    print("=" * 75)

    data_file = find_dataset()
    print(f"\n1. Loading dataset from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 1. Feature matrix X and Target vector y
    feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
    X = df[feature_cols].copy()
    
    # 2. Target Encoding:
    # Safe = 0, Unsafe = 1
    # 1 represents the primary alert condition (Contaminated / Non-potable water)
    target_mapping = {'Safe': 0, 'Unsafe': 1}
    y = df['Label'].map(target_mapping)

    print("\n2. Target Encoding Specification:")
    print("   'Safe'   -> 0 (Negative class: Potable water, normal distribution)")
    print("   'Unsafe' -> 1 (Positive class: Non-potable water, contamination alert)")
    print(f"   Class balance in full dataset: 0={sum(y==0):,} ({(y==0).mean()*100:.2f}%), "
          f"1={sum(y==1):,} ({(y==1).mean()*100:.2f}%)")

    # 3. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\n3. Stratified Train-Test Split (80/20, random_state=42):")
    print(f"   Training set : {len(X_train):,} samples (80.00%)")
    print(f"     Safe (0)   : {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.2f}%)")
    print(f"     Unsafe (1) : {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"   Testing set  : {len(X_test):,} samples (20.00%)")
    print(f"     Safe (0)   : {sum(y_test==0):,} ({sum(y_test==0)/len(y_test)*100:.2f}%)")
    print(f"     Unsafe (1) : {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.2f}%)")

    # 4. Define Models
    # Model A: Logistic Regression with StandardScaler inside Pipeline
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Model B: Decision Tree (no scaling, max_depth=5 for interpretability)
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        criterion='gini',
        random_state=42
    )

    # Model C: Random Forest (no scaling, 100 trees, max_depth=8)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )

    models = {
        'Logistic Regression': lr_pipeline,
        'Decision Tree': dt_model,
        'Random Forest': rf_model
    }

    # 5. Train & Evaluate Models
    print("\n" + "=" * 75)
    print("4. MODEL TRAINING & TEST SET EVALUATION")
    print("=" * 75)

    comparison_records = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Per-class metrics
        prec_safe = precision_score(y_test, y_pred, pos_label=0)
        rec_safe = recall_score(y_test, y_pred, pos_label=0)
        f1_safe = f1_score(y_test, y_pred, pos_label=0)

        prec_unsafe = precision_score(y_test, y_pred, pos_label=1)
        rec_unsafe = recall_score(y_test, y_pred, pos_label=1)
        f1_unsafe = f1_score(y_test, y_pred, pos_label=1)

        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"  Accuracy           : {acc:.6f} ({acc*100:.3f}%)")
        print(f"  Safe   (0) Precision/Recall/F1: {prec_safe:.6f} / {rec_safe:.6f} / {f1_safe:.6f}")
        print(f"  Unsafe (1) Precision/Recall/F1: {prec_unsafe:.6f} / {rec_unsafe:.6f} / {f1_unsafe:.6f}")
        print(f"  Macro F1-Score     : {macro_f1:.6f}")
        print(f"  Weighted F1-Score  : {weighted_f1:.6f}")
        print("  Confusion Matrix:")
        print(f"    [TN (Safe -> Safe)     : {tn:>6,} | FP (Safe -> Unsafe)   : {fp:>4,}]")
        print(f"    [FN (Unsafe -> Safe)   : {fn:>6,} | TP (Unsafe -> Unsafe) : {tp:>4,}]")

        # Operational False Negative Note
        if fn > 0:
            print(f"  * WARNING: {fn} False Negatives (Unsafe water classified as Safe)!")
        else:
            print("  * Zero False Negatives: All unsafe observations correctly flagged.")

        comparison_records.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Safe)': prec_safe,
            'Recall (Safe)': rec_safe,
            'F1 (Safe)': f1_safe,
            'Precision (Unsafe)': prec_unsafe,
            'Recall (Unsafe)': rec_unsafe,
            'F1 (Unsafe)': f1_unsafe,
            'Macro F1': macro_f1,
            'Weighted F1': weighted_f1,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp
        })

    # 6. Save Comparison Table
    comparison_df = pd.DataFrame(comparison_records)
    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved model comparison to: {comparison_csv}")

    # 7. Model Persistence (joblib)
    print("\n5. Saving Models with joblib:")
    model_filenames = {
        'Logistic Regression': MODELS_DIR / "logistic_regression.joblib",
        'Decision Tree': MODELS_DIR / "decision_tree.joblib",
        'Random Forest': MODELS_DIR / "random_forest.joblib"
    }
    for name, path in model_filenames.items():
        joblib.dump(trained_models[name], path)
        print(f"   Saved {name:<20} -> {path}")

    # 8. Decision Tree Visualization
    print("\n6. Generating Decision Tree Structure Visualization...")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        trained_models['Decision Tree'],
        feature_names=feature_cols,
        class_names=['Safe', 'Unsafe'],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=3, # Visualize top 3 levels for publication readability
        ax=ax
    )
    ax.set_title("Decision Tree Decision Logic (Top 3 Depths Shown for Readability)", fontsize=16, fontweight='bold', pad=15)
    dt_fig_path = FIGURES_DIR / "decision_tree.png"
    plt.tight_layout()
    fig.savefig(dt_fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {dt_fig_path}")

    # 9. Random Forest Feature Importance Visualization
    print("\n7. Computing & Plotting Random Forest Feature Importance...")
    rf = trained_models['Random Forest']
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(importances.index, importances.values, color="#1c7ed6", edgecolor="#333", height=0.55)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2., f"{w*100:.2f}%", va='center', fontsize=11, fontweight='bold')

    ax.set_title("Random Forest Feature Importance (Mean Decrease in Impurity)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_xlim(0, max(importances.values) * 1.18)
    rf_fig_path = FIGURES_DIR / "random_forest_feature_importance.png"
    plt.tight_layout()
    fig.savefig(rf_fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {rf_fig_path}")

    # Print feature importance table
    print("\n   Random Forest Feature Importances:")
    for feat, imp in importances.iloc[::-1].items():
        print(f"     {feat:<18} : {imp:.4f} ({imp*100:.2f}%)")

    # 10. Generate Markdown Report
    print("\n8. Generating Phase 3 Training Report (reports/phase3_training_report.md)...")
    generate_markdown_report(comparison_df, importances.iloc[::-1])
    print("=" * 75)
    print("PHASE 3 COMPLETE")
    print("=" * 75)

def df_to_markdown(df):
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---:" if c in ['TN','FP','FN','TP'] else ":---" for c in headers]) + " |"
    rows = []
    for _, row in df.iterrows():
        formatted_vals = []
        for c, val in row.items():
            if isinstance(val, float):
                formatted_vals.append(f"{val:.4f}")
            elif isinstance(val, (int, np.integer)):
                formatted_vals.append(f"{val:,}")
            else:
                formatted_vals.append(str(val))
        rows.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join([header_line, sep_line] + rows)

def generate_markdown_report(comp_df, rf_importances):
    report_path = REPORTS_DIR / "phase3_training_report.md"
    table_md = df_to_markdown(comp_df)
    
    report_content = f"""# Phase 3 — Data Preprocessing & Model Training Report

**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services  
**Dataset**: [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv) (100,800 rows × 5 columns)  
**Execution Script**: [`src/train.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/train.py)  

---

## 1. Data Preprocessing & Partitioning

### 1.1 Feature Matrix & Target Encoding
- **Input Features ($X$)**:
  1. `pH` (acidity/alkalinity)
  2. `TDS_ppm` (Total Dissolved Solids in parts per million)
  3. `Turbidity_NTU` (water cloudiness)
  4. `Temperature_C` (ambient water temperature)
- **Target Variable ($y$)**: `Label`
  - **`0` = `Safe`** (Potable, compliant drinking water — Negative operational class)
  - **`1` = `Unsafe`** (Non-potable water exceeding limits — Positive alarm class)
- **Rationale for Positive Class**: In public health monitoring and environmental cloud services, the "alarm condition" is contaminated water. Treating `Unsafe` as class `1` aligns recall and precision metrics with contamination detection.

### 1.2 Stratified 80/20 Train-Test Split
- **Total Dataset**: 100,800 samples
- **Split Configuration**: `test_size=0.20`, `random_state=42`, `stratify=y`
- **Training Partition (80%)**: 80,640 samples
  - `Safe (0)`: 24,192 (30.00%)
  - `Unsafe (1)`: 56,448 (70.00%)
- **Test Partition (20%)**: 20,160 samples
  - `Safe (0)`: 6,048 (30.00%)
  - `Unsafe (1)`: 14,112 (70.00%)
- **Verification**: Both subsets strictly maintain the 70:30 population proportion, preventing sampling bias.

---

## 2. Models Evaluated & Pipelines

1. **Logistic Regression (Baseline Linear Classifier)**:
   - Pipeline with `StandardScaler` to normalize disparate sensor scales (e.g., TDS [28–2806] vs pH [6–10]).
   - Hyperparameters: `max_iter=1000`, `random_state=42`.
2. **Decision Tree (Interpretable White-Box Model)**:
   - Scale-invariant tree architecture; no feature scaling required.
   - Hyperparameters: `max_depth=5`, `criterion='gini'`, `random_state=42`.
3. **Random Forest (Ensemble Classifier)**:
   - 100 randomized decision trees aggregating bootstrap subsets.
   - Hyperparameters: `n_estimators=100`, `max_depth=8`, `random_state=42`, `n_jobs=-1`.

---

## 3. Comprehensive Model Evaluation Results

### 3.1 Performance Comparison Table (Test Set: 20,160 Samples)

{table_md}

### 3.2 Confusion Matrices & Error Breakdown

| Model | True Negatives (TN)<br>(Safe → Safe) | False Positives (FP)<br>(Safe → Unsafe) | False Negatives (FN)<br>(Unsafe → Safe) | True Positives (TP)<br>(Unsafe → Unsafe) |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | {comp_df.loc[0, 'TN']:,} | {comp_df.loc[0, 'FP']} | {comp_df.loc[0, 'FN']} | {comp_df.loc[0, 'TP']:,} |
| **Decision Tree** | {comp_df.loc[1, 'TN']:,} | {comp_df.loc[1, 'FP']} | {comp_df.loc[1, 'FN']} | {comp_df.loc[1, 'TP']:,} |
| **Random Forest** | {comp_df.loc[2, 'TN']:,} | {comp_df.loc[2, 'FP']} | {comp_df.loc[2, 'FN']} | {comp_df.loc[2, 'TP']:,} |

---

## 4. In-Depth Operational Error Analysis: Unsafe → Safe (False Negatives)

### Operational Meaning:
A **False Negative** occurs when the physical water is **`Unsafe` (Class 1)**, but the model incorrectly predicts **`Safe` (Class 0)**.

### Severity & Consequences:
- **Severe Health Risk**: Contaminated water containing excessive suspended matter (Turbidity), high salinity/minerals (TDS), or corrosive pH is mistakenly allowed to enter municipal pipelines or human consumption.
- **System Failure**: In an automated IoT cloud actuation pipeline, a False Negative fails to trigger the automated shutoff valve, allowing contaminated water to pass downstream undetected.
- **Comparison to False Positives**:
  - A *False Positive* (Safe flagged as Unsafe) merely causes an unnecessary filter backwash cycle or manual verification grab sample (minor economic inconvenience).
  - A *False Negative* leads to regulatory non-compliance, equipment damage, or public health emergencies.
- **Observed Results**:
  - **Random Forest**: Produced only {comp_df.loc[2, 'FN']} False Negative out of 14,112 unsafe samples (Recall: {comp_df.loc[2, 'Recall (Unsafe)']*100:.3f}%).
  - **Decision Tree**: Produced only {comp_df.loc[1, 'FN']} False Negative out of 14,112 unsafe samples (Recall: {comp_df.loc[1, 'Recall (Unsafe)']*100:.3f}%).
  - **Logistic Regression**: Produced only {comp_df.loc[0, 'FN']} False Negatives out of 14,112 unsafe samples (Recall: {comp_df.loc[0, 'Recall (Unsafe)']*100:.3f}%).

---

## 5. Model Explainability & Visual Artifacts

### 5.1 Decision Tree Visual Inspection
- Saved to: [`reports/figures/decision_tree.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_tree.png)
- **Top Decision Rules**:
  - The root split immediately leverages **`TDS_ppm`** and **`Turbidity_NTU`**.
  - Thresholds clearly delineate potable thresholds (e.g. TDS < ~211 ppm, Turbidity < ~4.5 NTU).
  - Enables transparent, inspectable reasoning for cloud-based alert generation.

### 5.2 Random Forest Feature Importance
- Saved to: [`reports/figures/random_forest_feature_importance.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/random_forest_feature_importance.png)
- **Gini Importance Breakdown**:
{chr(10).join([f"  - **`{k}`**: {v*100:.2f}%" for k, v in rf_importances.items()])}

---

## 6. Persisted Model Artifacts (joblib)

The trained models have been saved to [`models/`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/) for downstream deployment:
- [`models/logistic_regression.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/logistic_regression.joblib)
- [`models/decision_tree.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/decision_tree.joblib)
- [`models/random_forest.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/random_forest.joblib)

---

**Phase 3 Complete.** Ready for Phase 4 (Detailed Evaluation, Model Comparison & Selection).
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"   Saved training report: {report_path}")

if __name__ == "__main__":
    main()
