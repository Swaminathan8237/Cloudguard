"""
Script to create notebooks/02_eda.ipynb for Phase 2: Exploratory Data Analysis.
"""

from pathlib import Path
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    }

    cells = []

    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell("""# CloudGuard-Water: Exploratory Data Analysis (Phase 2)

**Course**: Machine Learning for Cloud Services  
**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Dataset**: `data/water_quality_dataset1.csv` (100,800 rows × 5 columns)  

---

### Objectives of this Notebook:
1. Verify and visualize the class distribution (`Safe` vs `Unsafe`) and understand the operational impact of class imbalance.
2. Analyze the individual distributions, ranges, and skewness of the four sensor features: `pH`, `TDS_ppm`, `Turbidity_NTU`, `Temperature_C`.
3. Visually compare Safe vs. Unsafe samples using box plots and violin plots to evaluate feature discriminative power.
4. Calculate and visualize the correlation matrix between numerical input features (without target leakage).
5. Explore pairwise relationships using a stratified representative sample.
6. Identify and visualize IQR-based statistical outliers and document why they must be retained for realistic water quality monitoring.
7. Synthesize key findings and actionable recommendations for ML modeling in Phase 3/4.
"""))

    # Cell 2: Imports & Configuration
    cells.append(nbf.v4.new_code_cell("""import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visual styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
sns.set_theme(style="whitegrid", palette="muted")

CLASS_PALETTE = {"Safe": "#2b8a3e", "Unsafe": "#c92a2a"}
NUMERIC_FEATURES = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
TARGET_COL = 'Label'
"""))

    # Cell 3: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Integrity Verification
Load the authoritative dataset and verify that it contains exactly 100,800 rows and 5 columns with zero missing values.
"""))

    # Cell 4: Data loading code
    cells.append(nbf.v4.new_code_cell("""# Locate dataset relative to notebook directory
data_path = Path("../data/water_quality_dataset1.csv")
if not data_path.exists():
    data_path = Path("data/water_quality_dataset1.csv")

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
df.head(10)
"""))

    # Cell 5: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Target Class Distribution (`Label`)

In safety-critical classification, understanding the class balance is paramount:
- **Accuracy Paradox**: A naive model that always predicts `Unsafe` would achieve 70.0% accuracy while failing 100% of potable water assessments.
- **Cost of Errors**: False Negatives (predicting `Safe` when water is actually `Unsafe`) have severe health and regulatory consequences.
"""))

    # Cell 6: Target distribution code & plot
    cells.append(nbf.v4.new_code_cell("""counts = df[TARGET_COL].value_counts()
percentages = df[TARGET_COL].value_counts(normalize=True) * 100

print(f"Safe:   {counts['Safe']:,} samples ({percentages['Safe']:.2f}%)")
print(f"Unsafe: {counts['Unsafe']:,} samples ({percentages['Unsafe']:.2f}%)")
print(f"Imbalance Ratio (Unsafe : Safe): {counts['Unsafe'] / counts['Safe']:.2f} : 1")

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(counts.index, counts.values, color=[CLASS_PALETTE[c] for c in counts.index], edgecolor="#333", width=0.5)

for bar in bars:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., h + 1500, f"{h:,}\\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')

ax.set_title("Class Distribution: Safe vs. Unsafe", fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel("Observation Count", fontsize=11)
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.show()
"""))

    # Cell 7: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Individual Feature Distributions

Examine the histogram, Kernel Density Estimate (KDE), range, and skewness for each physical water quality indicator:
- `pH`: Dimensionless acidity/alkalinity scale (WHO potability benchmark: 6.5–8.5).
- `TDS_ppm`: Dissolved mineral/salt concentration (desirable drinking limit: <300–500 ppm).
- `Turbidity_NTU`: Suspended particulate matter and cloudiness (potable limit: <5.0 NTU).
- `Temperature_C`: Ambient water temperature (°C).
"""))

    # Cell 8: Feature distributions code & plot
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

titles = {
    'pH': "pH Distribution (WHO: 6.5 - 8.5)",
    'TDS_ppm': "Total Dissolved Solids (TDS in ppm)",
    'Turbidity_NTU': "Turbidity (NTU, WHO Limit: < 5 NTU)",
    'Temperature_C': "Water Temperature (°C)"
}

for idx, col in enumerate(NUMERIC_FEATURES):
    ax = axes[idx]
    skew = df[col].skew()
    sns.histplot(data=df, x=col, hue=TARGET_COL, palette=CLASS_PALETTE, kde=True, ax=ax, bins=40, stat="density", common_norm=False, alpha=0.45)
    ax.set_title(titles[col], fontsize=11, fontweight='bold')
    ax.text(0.04, 0.90, f"Skewness: {skew:+.2f}", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle("Feature Distributions by Water Safety Class", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
"""))

    # Cell 9: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. Safe vs. Unsafe Comparison (Boxplots)

Visualizing the central tendencies (median, interquartile range) and spread between `Safe` and `Unsafe` samples provides direct visual evidence of feature discriminative power.
"""))

    # Cell 10: Boxplots code
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

units = {'pH': 'pH', 'TDS_ppm': 'ppm', 'Turbidity_NTU': 'NTU', 'Temperature_C': '°C'}

for idx, col in enumerate(NUMERIC_FEATURES):
    ax = axes[idx]
    sns.boxplot(data=df, x=TARGET_COL, y=col, hue=TARGET_COL, palette=CLASS_PALETTE, ax=ax, width=0.45, fliersize=1.5, legend=False)
    safe_m = df[df[TARGET_COL] == 'Safe'][col].mean()
    unsafe_m = df[df[TARGET_COL] == 'Unsafe'][col].mean()
    ax.set_title(f"{col}\\nSafe: {safe_m:.2f} | Unsafe: {unsafe_m:.2f}", fontsize=10, fontweight='bold')
    ax.set_ylabel(units[col])

plt.suptitle("Safe vs. Unsafe Feature Comparison", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()
"""))

    # Cell 11: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Input Feature Correlation Matrix

We examine pairwise linear correlations among the **four input features only**.
*Note*: As emphasized in rigorous data science practice, **correlation does not imply causation**. Environmental processes (e.g., runoff containing both dissolved minerals and suspended sediment) cause co-movement among variables.
"""))

    # Cell 12: Correlation code
    cells.append(nbf.v4.new_code_cell("""corr = df[NUMERIC_FEATURES].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0, square=True, linewidths=1.2, ax=ax)
ax.set_title("Input Feature Pearson Correlation Heatmap", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()

print("Correlation Matrix:")
print(corr.round(4))
"""))

    # Cell 13: Markdown Section 6
    cells.append(nbf.v4.new_markdown_cell("""## 6. Pairwise Relationships (Representative Stratified Sample)

Because rendering 100,800 points in a 4×4 pairplot is computationally intensive and causes severe visual overplotting, we use a **5% stratified representative sample (5,040 rows)** solely for visualization.
"""))

    # Cell 14: Pairplot code
    cells.append(nbf.v4.new_code_cell("""sample_df = df.groupby(TARGET_COL, group_keys=False).apply(
    lambda x: x.sample(n=int(len(x) * 0.05), random_state=42),
    include_groups=True
)

g = sns.pairplot(
    sample_df, 
    vars=NUMERIC_FEATURES, 
    hue=TARGET_COL, 
    palette=CLASS_PALETTE, 
    diag_kind="kde",
    plot_kws={"alpha": 0.35, "s": 12, "edgecolor": "none"}
)
g.fig.subplots_adjust(top=0.94)
g.fig.suptitle("Pairwise Feature Scatter & KDE Distributions (5% Stratified Sample)", fontsize=12, fontweight='bold')
plt.show()
"""))

    # Cell 15: Markdown Section 7
    cells.append(nbf.v4.new_markdown_cell("""## 7. Statistical Outlier Inspection (Tukey's IQR Rule)

In Phase 1, the following IQR-based outliers were identified:
- `pH`: 423 points (>9.73)
- `TDS_ppm`: 3,096 points (>2,313.8 ppm)
- `Turbidity_NTU`: 118 points (>43.47 NTU)
- `Temperature_C`: 72 points (<17.33 °C)

**Why Outliers Must NOT Be Removed**:
1. High TDS values (>2,000 ppm) represent real, highly saline marine and coastal water samples.
2. High turbidity (>40 NTU) represents real storm runoff events.
3. These extreme values are **real-world environmental extremes**, not measurement corruptions or typos. Removing them would blind our safety classifier to the most severe contamination events.
"""))

    # Cell 16: Outlier boxplot code
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))

for idx, col in enumerate(NUMERIC_FEATURES):
    ax = axes[idx]
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = (n_outliers / len(df)) * 100
    
    sns.boxplot(data=df, y=col, ax=ax, color="#74c0fc", width=0.35, fliersize=2)
    if lower > df[col].min():
        ax.axhline(lower, color="#d9480f", linestyle="--", label=f"Lower ({lower:.1f})")
    if upper < df[col].max():
        ax.axhline(upper, color="#d9480f", linestyle="--", label=f"Upper ({upper:.1f})")
    ax.set_title(f"{col}\\n{n_outliers:,} Outliers ({pct:.2f}%)", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc="upper right")

plt.suptitle("Statistical Outlier Inspection (1.5 × IQR Rule)", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()
"""))

    # Cell 17: Markdown Section 8
    cells.append(nbf.v4.new_markdown_cell("""## 8. Statistical Summary Table

Synthesizing all descriptive statistics and per-class metrics into a concise reference table:
"""))

    # Cell 18: Summary table code
    cells.append(nbf.v4.new_code_cell("""summary_records = []
for col in NUMERIC_FEATURES:
    s_mean = df[df[TARGET_COL] == 'Safe'][col].mean()
    u_mean = df[df[TARGET_COL] == 'Unsafe'][col].mean()
    summary_records.append({
        'Feature': col,
        'Overall Mean': df[col].mean(),
        'Overall Median': df[col].median(),
        'Overall Std': df[col].std(),
        'Min': df[col].min(),
        'Max': df[col].max(),
        'Safe Mean': s_mean,
        'Unsafe Mean': u_mean,
        'Mean Difference (Unsafe - Safe)': u_mean - s_mean
    })

summary_table = pd.DataFrame(summary_records)
summary_table.round(2)
"""))

    # Cell 19: Markdown Section 9
    cells.append(nbf.v4.new_markdown_cell("""## 9. Key Findings for ML Modeling (Transition to Phase 3/4)

### 1. Discriminative Power
- **TDS_ppm** and **Turbidity_NTU** exhibit distinct separation between Safe and Unsafe distributions.
- **pH** and **Temperature_C** show meaningful but overlapping shifts.

### 2. Class Imbalance (70% Unsafe : 30% Safe)
- Must employ **Stratified K-Fold** or stratified train/test splits.
- Primary evaluation must focus on **Recall on Unsafe** and **Macro F1-Score**.

### 3. Feature Scaling
- Required for **Logistic Regression** (due to vastly differing physical scales, e.g., TDS [28–2806] vs pH [6–10]).
- Not required for **Decision Tree** and **Random Forest**.

### 4. Preservation of Outliers
- All IQR outliers represent valid physical phenomena; retaining them ensures realistic generalization.

### 5. Multicollinearity
- High positive correlation between pH and TDS (+0.804) may affect Logistic Regression coefficients; tree-based models are naturally invariant.
"""))

    nb.cells = cells
    
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "02_eda.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {out_path}")

if __name__ == "__main__":
    create_notebook()
