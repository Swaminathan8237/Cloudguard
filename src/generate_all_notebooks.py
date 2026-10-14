"""
CloudGuard-Water — Notebook Generator & Runner
==============================================
Generates and executes:
  1. notebooks/01_dataset_inspection.ipynb (Phase 1)
  2. notebooks/03_baseline_models.ipynb   (Phase 3)

Ensures every script in the project is executable as an interactive notebook
with pre-rendered outputs, clean self-healing dataset locators, and academic markdown.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate notebooks/01_dataset_inspection.ipynb
# ---------------------------------------------------------------------------
def build_inspection_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# CloudGuard-Water: Phase 1 — Dataset Inspection & Verification

**Course**: Machine Learning for Cloud Services  
**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Dataset**: `data/water_quality_dataset1.csv`  

---

### Objectives:
1. Verify file size, row/column counts, and storage format.
2. Inspect schema and exact data types.
3. Validate data quality (missing values, exact duplicates, physical range validity).
4. Analyze target variable distribution (`Safe` vs. `Unsafe`) and calculate the class imbalance ratio.
5. Generate descriptive statistics (mean, median, std, min, max, IQR) for all 4 physical features.
6. Compute the numerical correlation matrix.
7. Conduct statistical outlier detection using Tukey's $1.5 \\times \\text{IQR}$ rule without removing valid observations.
8. Assess dataset suitability for machine-learning classification.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style="whitegrid", palette="muted")
"""))

    # Section 1 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 1. File Metadata & Dataset Loading
We locate the authoritative dataset using a multi-environment locator that works across local IDEs, terminal runs, and remote environments.
"""))

    # Section 1 Code
    cells.append(nbf.v4.new_code_cell("""def find_dataset():
    candidates = [
        Path("data/water_quality_dataset1.csv"),
        Path("../data/water_quality_dataset1.csv"),
        Path("../../data/water_quality_dataset1.csv"),
        Path("data/water_quality_dataset1_original.csv"),
        Path("../data/water_quality_dataset1_original.csv"),
        Path("Cloudguard/data/water_quality_dataset1.csv"),
        Path("/content/Cloudguard/data/water_quality_dataset1.csv"),
        Path("/content/data/water_quality_dataset1.csv"),
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
    raise FileNotFoundError("Could not find dataset in standard locations.")

data_file = find_dataset()
file_size = data_file.stat().st_size

print(f"File Path    : {data_file.resolve()}")
print(f"File Size    : {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
print(f"File Format  : CSV")

df = pd.read_csv(data_file)
print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(10)
"""))

    # Section 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Schema & Exact Column Data Types
Checking exact column names, pandas dtypes, and memory usage.
"""))

    # Section 2 Code
    cells.append(nbf.v4.new_code_cell("""schema_info = []
for col in df.columns:
    schema_info.append({
        'Column Name': col,
        'Pandas dtype': str(df[col].dtype),
        'Non-Null Count': df[col].notnull().sum(),
        'Sample Value': df[col].iloc[0]
    })

schema_df = pd.DataFrame(schema_info)
print(f"In-Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
schema_df
"""))

    # Section 3 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. Data Quality & Physical Validity Checks
Verifying missing values, duplicate rows, and whether all values adhere to known physical boundaries:
- **pH**: Must be within $[0, 14]$
- **TDS_ppm**: Must be $\\ge 0$
- **Turbidity_NTU**: Must be $\\ge 0$
- **Temperature_C**: Plausible range $[-10, 100\\,^\\circ\\text{C}]$
"""))

    # Section 3 Code
    cells.append(nbf.v4.new_code_cell("""print("Missing Values per Column:")
print(df.isnull().sum())
print(f"\\nTotal Missing Values : {df.isnull().sum().sum()}")
print(f"Exact Duplicate Rows : {df.duplicated().sum()}")

# Physical boundary checks
range_checks = {
    'pH': (0, 14),
    'TDS_ppm': (0, 50000),
    'Turbidity_NTU': (0, 10000),
    'Temperature_C': (-10, 100)
}

print("\\nPhysical Plausibility Checks:")
for col, (lo, hi) in range_checks.items():
    viol = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:<18}: {viol} values outside [{lo}, {hi}]")
"""))

    # Section 4 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Target Variable Analysis (`Label`)
Analyzing class distribution, frequencies, and the imbalance ratio between `Safe` and `Unsafe`.
"""))

    # Section 4 Code
    cells.append(nbf.v4.new_code_cell("""counts = df['Label'].value_counts()
percs = df['Label'].value_counts(normalize=True) * 100

for lbl in counts.index:
    print(f"  {lbl:<8}: {counts[lbl]:>7,} samples ({percs[lbl]:>5.2f}%)")

imbalance_ratio = counts['Unsafe'] / counts['Safe']
print(f"\\nImbalance Ratio (Unsafe : Safe): {imbalance_ratio:.2f} : 1")

fig, ax = plt.subplots(figsize=(6, 4))
palette = {"Safe": "#2b8a3e", "Unsafe": "#c92a2a"}
bars = ax.bar(list(counts.index), list(counts.values), color=[palette[c] for c in counts.index], edgecolor="#333", width=0.5)

for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., h + 1500, f"{h:,}\\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')

ax.set_title("Target Variable Distribution (Safe vs. Unsafe)", fontsize=12, fontweight='bold', pad=10)
ax.set_ylabel("Observation Count")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.show()
"""))

    # Section 5 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 5. Numerical Feature Statistics & Per-Class Breakdown
Calculating descriptive statistics (mean, std, min, 25%, median, 75%, max) for each physical feature across the full dataset and grouped by class.
"""))

    # Section 5 Code
    cells.append(nbf.v4.new_code_cell("""features = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
desc = df[features].describe(percentiles=[0.25, 0.50, 0.75]).T
desc['median'] = df[features].median()

print("Overall Feature Statistics:")
print(desc[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']].round(2))

print("\\nPer-Class Means:")
class_means = df.groupby('Label')[features].mean().round(2)
class_means
"""))

    # Section 6 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Correlation Matrix
Evaluating pairwise linear correlations among input features and with the target.
"""))

    # Section 6 Code
    cells.append(nbf.v4.new_code_cell("""corr_df = df[features].copy()
corr_df['Is_Unsafe'] = (df['Label'] == 'Unsafe').astype(int)
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0, square=True, linewidths=1.2, ax=ax)
ax.set_title("Correlation Heatmap (Input Features + Is_Unsafe)", fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.show()
"""))

    # Section 7 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 7. Statistical Outlier Detection (Tukey's 1.5 × IQR Rule)
Calculating upper and lower fences.
**Critical Decision**: We report these outliers but **do not remove them**, because high TDS (>2,300 ppm) and high turbidity (>43 NTU) represent real coastal saline and runoff conditions that the model must recognize.
"""))

    # Section 7 Code
    cells.append(nbf.v4.new_code_cell("""outlier_summary = []
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
    outlier_summary.append({
        'Feature': col,
        'Q1': round(Q1, 2),
        'Q3': round(Q3, 2),
        'IQR': round(IQR, 2),
        'Lower Fence': round(lower_fence, 2),
        'Upper Fence': round(upper_fence, 2),
        'Outliers Count': n_outliers,
        '% Outliers': round(n_outliers / len(df) * 100, 2)
    })

pd.DataFrame(outlier_summary)
"""))

    # Section 8 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 8. Phase 1 Assessment & Findings Summary

1. **Dataset Integrity**: 100,800 rows, 5 columns, 0 missing, 0 duplicates, 0 physically invalid readings.
2. **Target Balance**: 70.0% Unsafe, 30.0% Safe (2.33:1 imbalance ratio).
3. **Key Predictors**: TDS and Turbidity show significant mean separation between Safe and Unsafe classes.
4. **Temporal Note**: No timestamp exists in this dataset; therefore, this is a **cross-sectional classification task**, not time-series forecasting.
5. **Conclusion**: The dataset is completely clean and fully suitable for machine-learning classification.
"""))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "01_dataset_inspection.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# 2. Generate notebooks/03_baseline_models.ipynb
# ---------------------------------------------------------------------------
def build_baseline_models_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# CloudGuard-Water: Phase 3 — Baseline Model Training & Evaluation

**Course**: Machine Learning for Cloud Services  
**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Dataset**: `data/water_quality_dataset1.csv` (100,800 rows × 5 columns)  

---

### Objectives:
1. Formulate the feature matrix $X$ and target vector $y$.
2. Encode classes: `0 = Safe`, `1 = Unsafe` (Positive alarm class).
3. Execute a stratified 80/20 train-test split (`test_size=0.20`, `random_state=42`, `stratify=y`).
4. Train three baseline models:
   - **Logistic Regression** (with `StandardScaler` inside a Pipeline)
   - **Decision Tree** (unscaled, `max_depth=5` for interpretability)
   - **Random Forest** (unscaled ensemble, `n_estimators=100`, `max_depth=8`)
5. Evaluate all models on Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, and Confusion Matrices.
6. Analyze the operational consequences of False Negatives (predicting `Safe` when water is actually `Unsafe`).
7. Visualize the Decision Tree decision rules and plot Random Forest feature importances.
8. Persist trained models using `joblib` into `models/`.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""import os
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

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style="whitegrid", palette="muted")
"""))

    # Section 1 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Target Encoding
We load the dataset and encode the target:
- **`Safe` → `0`** (Negative class: Potable water conforming to standards)
- **`Unsafe` → `1`** (Positive class: Contaminated water exceeding safety thresholds)
"""))

    # Section 1 Code
    cells.append(nbf.v4.new_code_cell("""def find_dataset():
    candidates = [
        Path("data/water_quality_dataset1.csv"),
        Path("../data/water_quality_dataset1.csv"),
        Path("../../data/water_quality_dataset1.csv"),
        Path("data/water_quality_dataset1_original.csv"),
        Path("../data/water_quality_dataset1_original.csv"),
        Path("Cloudguard/data/water_quality_dataset1.csv"),
        Path("/content/Cloudguard/data/water_quality_dataset1.csv"),
        Path("/content/data/water_quality_dataset1.csv"),
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
    raise FileNotFoundError("Could not find dataset in standard locations.")

data_file = find_dataset()
df = pd.read_csv(data_file)
print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

feature_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
X = df[feature_cols].copy()

target_mapping = {'Safe': 0, 'Unsafe': 1}
y = df['Label'].map(target_mapping)

print("\\nTarget Encoding:")
print(f"  'Safe'   -> 0 (Count: {list(y).count(0):,}, {list(y).count(0)/len(y)*100:.2f}%)")
print(f"  'Unsafe' -> 1 (Count: {list(y).count(1):,}, {list(y).count(1)/len(y)*100:.2f}%)")
"""))

    # Section 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 2. Stratified 80/20 Train-Test Split
Preserving the exact 70:30 class distribution across both training (80,640 samples) and testing (20,160 samples) subsets.
"""))

    # Section 2 Code
    cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

safe_train = list(y_train).count(0)
unsafe_train = list(y_train).count(1)
safe_test = list(y_test).count(0)
unsafe_test = list(y_test).count(1)
print(f"Training set : {len(X_train):,} samples (80.00%)")
print(f"  Safe (0)   : {safe_train:,} ({safe_train/len(y_train)*100:.2f}%)")
print(f"  Unsafe (1) : {unsafe_train:,} ({unsafe_train/len(y_train)*100:.2f}%)")
print(f"Testing set  : {len(X_test):,} samples (20.00%)")
print(f"  Safe (0)   : {safe_test:,} ({safe_test/len(y_test)*100:.2f}%)")
print(f"  Unsafe (1) : {unsafe_test:,} ({unsafe_test/len(y_test)*100:.2f}%)")
"""))

    # Section 3 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 3. Model Training
We initialize and train three baseline models:
1. **Logistic Regression**: Scaled via `StandardScaler` inside a Pipeline.
2. **Decision Tree**: Unscaled, `max_depth=5` for interpretability.
3. **Random Forest**: Unscaled ensemble with 100 trees, `max_depth=8`.
"""))

    # Section 3 Code
    cells.append(nbf.v4.new_code_cell("""models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, criterion='gini', random_state=42, n_jobs=-1)
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

print("All models successfully trained!")
"""))

    # Section 4 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 4. Model Evaluation & Comparison
Evaluating all models on the unseen 20,160-sample test set.
"""))

    # Section 4 Code
    cells.append(nbf.v4.new_code_cell("""comparison_records = []
confusion_matrices = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    prec_safe = precision_score(y_test, y_pred, pos_label=0)
    rec_safe = recall_score(y_test, y_pred, pos_label=0)
    f1_safe = f1_score(y_test, y_pred, pos_label=0)
    
    prec_unsafe = precision_score(y_test, y_pred, pos_label=1)
    rec_unsafe = recall_score(y_test, y_pred, pos_label=1)
    f1_unsafe = f1_score(y_test, y_pred, pos_label=1)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    cm = confusion_matrix(list(y_test), [int(p) for p in y_pred])
    confusion_matrices[name] = cm
    tn, fp, fn, tp = cm.ravel()
    
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

comparison_df = pd.DataFrame(comparison_records)
comparison_df.round(6)
"""))

    # Section 5 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 5. Confusion Matrices & Operational Error Analysis

In IoT water safety monitoring:
- **False Positive (FP)**: Safe water predicted as Unsafe $\\rightarrow$ triggers an unnecessary inspection or filter backwash (minor operational cost).
- **False Negative (FN)**: Contaminated water predicted as Safe $\\rightarrow$ dangerous failure mode where contaminated water passes into public drinking supplies without automated valve shutoff.
"""))

    # Section 5 Code
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred Safe', 'Pred Unsafe'],
                yticklabels=['Actual Safe', 'Actual Unsafe'])
    ax.set_title(f"{name}\\nFN (Dangerous): {cm[1, 0]} | FP: {cm[0, 1]}", fontsize=11, fontweight='bold')

plt.suptitle("Confusion Matrices on Test Set (20,160 Samples)", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()
"""))

    # Section 6 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 6. Model Explainability — Decision Tree Visualization
Visualizing the exact if-then decision boundaries for regulatory transparency.
"""))

    # Section 6 Code
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(18, 9))
plot_tree(
    trained_models['Decision Tree'],
    feature_names=feature_cols,
    class_names=['Safe', 'Unsafe'],
    filled=True,
    rounded=True,
    fontsize=9,
    max_depth=3,
    ax=ax
)
ax.set_title("Decision Tree Decision Logic (Top 3 Depths)", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

    # Section 7 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 7. Model Explainability — Random Forest Feature Importance
Calculating Mean Decrease in Impurity (Gini Importance) across all 100 ensemble trees.
"""))

    # Section 7 Code
    cells.append(nbf.v4.new_code_cell("""rf = trained_models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(list(importances.index), list(importances.values), color="#1c7ed6", edgecolor="#333", height=0.5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.015, bar.get_y() + bar.get_height()/2., f"{w*100:.2f}%", va='center', fontweight='bold', fontsize=10)

ax.set_title("Random Forest Feature Importance", fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel("Relative Importance Score")
ax.set_xlim(0, max(importances.values) * 1.18)
plt.tight_layout()
plt.show()

print("Feature Importance Ranking:")
for feat, imp in importances.iloc[::-1].items():
    print(f"  {feat:<18}: {imp:.4f} ({imp*100:.2f}%)")
"""))

    # Section 8 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## 8. Persisting Models to `models/`
Saving all trained models via `joblib` for downstream deployment into the cloud inference microservice.
"""))

    # Section 8 Code
    cells.append(nbf.v4.new_code_cell("""models_dir = Path("../models")
if not models_dir.exists():
    models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

for name, model in trained_models.items():
    filename = name.lower().replace(" ", "_") + ".joblib"
    save_path = models_dir / filename
    joblib.dump(model, save_path)
    print(f"Saved: {save_path.resolve()}")

print("\\nAll models successfully serialized!")
"""))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "03_baseline_models.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {out_path}")
    return out_path

# ---------------------------------------------------------------------------
# 3. Execute Notebooks & Save Outputs
# ---------------------------------------------------------------------------
def execute_notebook(nb_path):
    print(f"\nExecuting notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved outputs to: {nb_path}")

def main():
    p1 = build_inspection_notebook()
    p3 = build_baseline_models_notebook()

    execute_notebook(p1)
    execute_notebook(p3)

    print("\nALL NOTEBOOKS GENERATED AND EXECUTED WITH ZERO ERRORS!")

if __name__ == "__main__":
    main()
