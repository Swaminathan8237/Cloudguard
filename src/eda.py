"""
CloudGuard-Water — Phase 2: Exploratory Data Analysis (EDA)
=============================================================
Course: ML for Cloud Services
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Script: src/eda.py

Purpose:
  1. Calculate verified descriptive statistics for all 4 features.
  2. Compute per-class statistics and mean differences.
  3. Calculate feature correlation matrix (input features only).
  4. Generate and save 6 publication-ready figures to reports/figures/:
     - class_distribution.png
     - feature_distributions.png
     - safe_vs_unsafe_boxplots.png
     - correlation_heatmap.png
     - feature_pairplot.png (stratified representative sample for performance)
     - outlier_analysis.png
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Plotting Configuration (Clean, academic, accessible)
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid", palette="muted")

# Palette definition: Safe = Teal/Green, Unsafe = Crimson/Red
CLASS_PALETTE = {"Safe": "#2b8a3e", "Unsafe": "#c92a2a"}
COLOR_LIST = ["#228be6", "#fab005", "#12b886", "#fa5252"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "water_quality_dataset1.csv"
REPORTS_FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
TARGET_COL = 'Label'

def load_data():
    if not DATA_FILE.exists():
        print(f"Error: Dataset not found at {DATA_FILE}")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    return df

# ---------------------------------------------------------------------------
# 1. Class Distribution Analysis & Plot
# ---------------------------------------------------------------------------
def plot_class_distribution(df):
    print("\n--- 1. Class Distribution Analysis ---")
    counts = df[TARGET_COL].value_counts()
    percs = df[TARGET_COL].value_counts(normalize=True) * 100
    
    for cls in ['Safe', 'Unsafe']:
        print(f"  {cls}: {counts[cls]:,} samples, {percs[cls]:.2f}%")
    
    imbalance_ratio = counts['Unsafe'] / counts['Safe']
    print(f"  Class Imbalance Ratio (Unsafe : Safe): {imbalance_ratio:.2f} : 1")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        counts.index, 
        counts.values, 
        color=[CLASS_PALETTE[c] for c in counts.index],
        edgecolor="#333333",
        linewidth=1.2,
        width=0.55
    )
    
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2., 
            h + 1200, 
            f'{h:,}\n({pct:.1f}%)', 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontweight='bold',
            color='#212529'
        )
        
    ax.set_title("Class Distribution (Safe vs. Unsafe)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Water Quality Classification (Ground Truth)", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Observations", fontsize=12, labelpad=10)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Text annotation explaining imbalance
    ax.text(
        0.5, 0.88,
        f"Total Samples: {len(df):,}\n70.0% Unsafe  |  30.0% Safe (2.33:1 Imbalance)",
        transform=ax.transAxes,
        fontsize=10,
        ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da")
    )
    
    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "class_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 2. Feature Distributions (Overall & by Class)
# ---------------------------------------------------------------------------
def plot_feature_distributions(df):
    print("\n--- 2. Feature Distributions ---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    
    titles = {
        'pH': "pH Distribution (WHO Potable: 6.5 - 8.5)",
        'TDS_ppm': "Total Dissolved Solids (TDS in ppm)",
        'Turbidity_NTU': "Turbidity (NTU, WHO Limit: < 5 NTU)",
        'Temperature_C': "Water Temperature (°C)"
    }
    
    for idx, col in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        skew = df[col].skew()
        print(f"  {col:<18} -> Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean: {df[col].mean():.2f}, Skewness: {skew:.2f}")
        
        sns.histplot(
            data=df, 
            x=col, 
            hue=TARGET_COL, 
            palette=CLASS_PALETTE, 
            kde=True, 
            ax=ax, 
            bins=40,
            stat="density",
            common_norm=False,
            alpha=0.45,
            edgecolor=None
        )
        ax.set_title(titles[col], fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(title="Class", labels=["Unsafe", "Safe"], loc="upper right")
        
        # Add skewness annotation
        ax.text(
            0.04, 0.90,
            f"Skew: {skew:+.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", alpha=0.8, edgecolor="#cccccc")
        )
        
    plt.suptitle("Physicochemical Feature Distributions by Water Safety Class", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "feature_distributions.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 3. Safe vs Unsafe Comparison (Boxplots)
# ---------------------------------------------------------------------------
def plot_safe_vs_unsafe_boxplots(df):
    print("\n--- 3. Safe vs Unsafe Boxplot Comparison ---")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    units = {
        'pH': 'Dimensionless pH Units',
        'TDS_ppm': 'ppm (mg/L)',
        'Turbidity_NTU': 'NTU',
        'Temperature_C': '°C'
    }
    
    for idx, col in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        sns.boxplot(
            data=df, 
            x=TARGET_COL, 
            y=col, 
            hue=TARGET_COL,
            palette=CLASS_PALETTE, 
            ax=ax, 
            width=0.45,
            fliersize=1.5,
            legend=False
        )
        safe_m = df[df[TARGET_COL] == 'Safe'][col].mean()
        unsafe_m = df[df[TARGET_COL] == 'Unsafe'][col].mean()
        
        ax.set_title(f"{col}\nSafe: {safe_m:.2f} | Unsafe: {unsafe_m:.2f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Water Quality Class", fontsize=10)
        ax.set_ylabel(units[col], fontsize=10)
        
    plt.suptitle("Feature Variations & Class Separation Across Safe vs. Unsafe Observations", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "safe_vs_unsafe_boxplots.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 4. Correlation Analysis (4 Input Features Only)
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    print("\n--- 4. Feature Correlation Matrix (Input Features Only) ---")
    corr = df[NUMERIC_FEATURES].corr()
    print(corr.to_string(float_format=lambda x: f"{x:+.4f}"))
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".3f", 
        cmap="coolwarm", 
        vmin=-1.0, 
        vmax=1.0, 
        center=0.0,
        square=True, 
        linewidths=1.2, 
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        ax=ax
    )
    
    ax.set_title("Pearson Correlation Heatmap of Input Features", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "correlation_heatmap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 5. Pairwise Relationships (Representative Stratified Sample)
# ---------------------------------------------------------------------------
def plot_pairwise_relationships(df):
    print("\n--- 5. Pairwise Relationships (Representative Stratified Sample) ---")
    # Take a stratified sample of 6,000 observations (4,200 Unsafe, 1,800 Safe)
    sample_df = df.groupby(TARGET_COL, group_keys=False).apply(
        lambda x: x.sample(n=int(len(x) * 0.05), random_state=42)
    )
    print(f"  Sampled {len(sample_df):,} rows ({len(sample_df[sample_df[TARGET_COL]=='Safe']):,} Safe, "
          f"{len(sample_df[sample_df[TARGET_COL]=='Unsafe']):,} Unsafe) for clean pairwise visualization.")
    
    g = sns.pairplot(
        sample_df, 
        vars=NUMERIC_FEATURES, 
        hue=TARGET_COL, 
        palette=CLASS_PALETTE, 
        diag_kind="kde",
        plot_kws={"alpha": 0.35, "s": 12, "edgecolor": "none"},
        diag_kws={"fill": True, "alpha": 0.45},
        corner=False
    )
    g.figure.subplots_adjust(top=0.94)
    g.figure.suptitle("Pairwise Sensor Feature Scatter & KDE Distributions by Class (5% Stratified Sample)", fontsize=13, fontweight='bold')
    
    out_path = REPORTS_FIG_DIR / "feature_pairplot.png"
    g.savefig(out_path, dpi=250)
    plt.close()
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 6. Outlier Visualization (IQR Boxplots with Fences)
# ---------------------------------------------------------------------------
def plot_outlier_analysis(df):
    print("\n--- 6. Outlier Visualization ---")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    for idx, col in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        
        n_outliers = ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        sns.boxplot(
            data=df, 
            y=col, 
            ax=ax, 
            color="#74c0fc", 
            width=0.35,
            fliersize=2,
            flierprops={"marker": "o", "color": "#e03131", "alpha": 0.4}
        )
        
        # Draw IQR fences
        if lower_fence > df[col].min():
            ax.axhline(lower_fence, color="#d9480f", linestyle="--", linewidth=1.2, label=f"Lower Fence ({lower_fence:.1f})")
        if upper_fence < df[col].max():
            ax.axhline(upper_fence, color="#d9480f", linestyle="--", linewidth=1.2, label=f"Upper Fence ({upper_fence:.1f})")
            
        ax.set_title(f"{col}\n{n_outliers:,} Outliers ({pct_outliers:.2f}%)", fontsize=11, fontweight='bold')
        ax.set_ylabel(col, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        
    plt.suptitle("Statistical Outlier Inspection (Tukey's 1.5×IQR Rule)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "outlier_analysis.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

# ---------------------------------------------------------------------------
# 7. Comprehensive Statistical Summary Table
# ---------------------------------------------------------------------------
def generate_statistical_summary_table(df):
    print("\n--- 7. Statistical Summary Table ---")
    records = []
    
    for col in NUMERIC_FEATURES:
        overall_mean = df[col].mean()
        overall_med = df[col].median()
        overall_std = df[col].std()
        overall_min = df[col].min()
        overall_max = df[col].max()
        
        safe_mean = df[df[TARGET_COL] == 'Safe'][col].mean()
        unsafe_mean = df[df[TARGET_COL] == 'Unsafe'][col].mean()
        mean_diff = unsafe_mean - safe_mean
        
        safe_med = df[df[TARGET_COL] == 'Safe'][col].median()
        unsafe_med = df[df[TARGET_COL] == 'Unsafe'][col].median()
        
        records.append({
            'Feature': col,
            'Mean': overall_mean,
            'Median': overall_med,
            'Std Dev': overall_std,
            'Min': overall_min,
            'Max': overall_max,
            'Safe Mean': safe_mean,
            'Unsafe Mean': unsafe_mean,
            'Diff (Unsafe - Safe)': mean_diff,
            'Safe Median': safe_med,
            'Unsafe Median': unsafe_med
        })
        
    summary_df = pd.DataFrame(records)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return summary_df

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def run_all_eda():
    print("=" * 70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    df = load_data()
    print(f"Loaded dataset: {DATA_FILE.name} with {len(df):,} rows and {len(df.columns)} columns.")
    
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_safe_vs_unsafe_boxplots(df)
    plot_correlation_heatmap(df)
    plot_pairwise_relationships(df)
    plot_outlier_analysis(df)
    summary_df = generate_statistical_summary_table(df)
    
    print("\n" + "=" * 70)
    print("EDA EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return df, summary_df

if __name__ == "__main__":
    run_all_eda()
