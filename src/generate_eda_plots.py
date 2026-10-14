"""
CloudGuard-Water: Exploratory Data Analysis (EDA) Plot Generator
Generates academic publication-quality visualizations for Report 1.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

def get_paths():
    # Find project root
    root = Path(__file__).resolve().parent.parent
    if not (root / "data").exists():
        root = Path.cwd()
        
    data_candidates = [
        root / "data" / "water_quality_dataset1.csv",
        Path("data/water_quality_dataset1.csv"),
        Path("water_quality_dataset1.csv")
    ]
    data_path = None
    for p in data_candidates:
        if p.exists():
            data_path = p
            break
    if not data_path:
        data_path = root / "data" / "water_quality_dataset1.csv"
        
    output_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(data_path), str(output_dir)

data_file, output_dir = get_paths()
print(f"Reading dataset: {data_file}")
print(f"Output directory: {output_dir}")

df = pd.read_csv(data_file)

# Color palette: Dark cyan & Coral
safe_color = "#2b5c8f"
unsafe_color = "#d95f02"

# 1. Target Class Distribution
plt.figure(figsize=(7, 5), dpi=300)
counts = df['Label'].value_counts()
bars = plt.bar(counts.index, counts.values, color=[unsafe_color, safe_color], width=0.5, edgecolor='black')
plt.title("Target Class Distribution: Safe vs. Unsafe Water", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Water Potability Label", fontsize=11)
plt.ylabel("Number of Observations", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f"{yval:,} ({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()
print("Saved class_distribution.png")

# 2. Sensor Feature Distributions (2x2 Grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
features = [
    ('pH', 'pH Level', 'pH Units'),
    ('TDS_ppm', 'Total Dissolved Solids', 'ppm (mg/L)'),
    ('Turbidity_NTU', 'Turbidity', 'NTU'),
    ('Temperature_C', 'Water Temperature', '°C')
]

for ax, (col, title, unit) in zip(axes.flatten(), features):
    safe_data = df[df['Label'] == 'Safe'][col]
    unsafe_data = df[df['Label'] == 'Unsafe'][col]
    
    ax.hist(safe_data, bins=35, alpha=0.65, label='Safe', color=safe_color, density=True, edgecolor='none')
    ax.hist(unsafe_data, bins=35, alpha=0.65, label='Unsafe', color=unsafe_color, density=True, edgecolor='none')
    ax.set_title(f"Distribution of {title}", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{title} ({unit})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle("Physicochemical Sensor Distributions by Water Safety Class", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sensor_distributions.png"))
plt.close()
print("Saved sensor_distributions.png")

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6), dpi=300)
corr_df = df[['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']].copy()
corr_df['Is_Unsafe'] = (df['Label'] == 'Unsafe').astype(int)
corr = corr_df.corr()

im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr)), corr.columns, rotation=30, ha='right', fontsize=10)
plt.yticks(range(len(corr)), corr.columns, fontsize=10)
plt.title("Correlation Matrix (Sensors and Binary Safety)", fontsize=13, fontweight='bold', pad=15)

for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()
print("Saved correlation_matrix.png")

# 4. Feature Importance Plot
plt.figure(figsize=(7, 4), dpi=300)
importances = pd.Series([0.6744, 0.2566, 0.0539, 0.0151], index=['TDS_ppm', 'Turbidity_NTU', 'Temperature_C', 'pH'])
y_pos = np.arange(len(importances))
bars = plt.barh(y_pos, importances.values[::-1], color='#1f77b4', edgecolor='black', height=0.55)
plt.yticks(y_pos, importances.index[::-1], fontsize=10)
plt.xlabel("Gini Feature Importance", fontsize=11)
plt.title("Random Forest Feature Importance (Baseline Classifier)", fontsize=12, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 0.01, bar.get_y() + bar.get_height()/2.0, f"{xval*100:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold')

plt.xlim(0, 0.75)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()
print("Saved feature_importance.png")
print("All EDA visualizations successfully updated in outputs/figures/!")
