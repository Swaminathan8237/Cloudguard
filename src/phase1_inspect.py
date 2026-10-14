"""
CloudGuard-Water — Phase 1: Dataset Inspection & Verification
=============================================================
Purpose: Independently verify every property of water_quality_dataset1.csv
         before any ML work begins.

This script reads the ACTUAL CSV and reports:
  1. File metadata (name, size, format)
  2. Dataset shape (rows, columns)
  3. Schema (column names, dtypes)
  4. Data quality (missing, duplicates, nulls, invalid values)
  5. Target analysis (unique values, class counts, percentages)
  6. Numerical summary (min, max, mean, median, std)
  7. Correlation matrix
  8. Outlier detection (IQR method — report only, do NOT remove)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Locate the dataset
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "water_quality_dataset1.csv"

if not DATA_FILE.exists():
    # Fallback: try relative path from cwd
    DATA_FILE = Path("data") / "water_quality_dataset1.csv"
    if not DATA_FILE.exists():
        print("ERROR: water_quality_dataset1.csv not found.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 2. File metadata
# ---------------------------------------------------------------------------
file_size_bytes = os.path.getsize(DATA_FILE)
print("=" * 70)
print("PHASE 1: DATASET INSPECTION & VERIFICATION")
print("=" * 70)

print("\n--- 1. File Information ---")
print(f"  Filename       : {DATA_FILE.name}")
print(f"  Full path      : {DATA_FILE.resolve()}")
print(f"  File size      : {file_size_bytes:,} bytes ({file_size_bytes / 1024:.1f} KB / {file_size_bytes / (1024*1024):.2f} MB)")
print(f"  File format    : CSV (Comma-Separated Values)")

# ---------------------------------------------------------------------------
# 3. Load the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# ---------------------------------------------------------------------------
# 4. Dataset shape
# ---------------------------------------------------------------------------
n_rows, n_cols = df.shape
print("\n--- 2. Dataset Shape ---")
print(f"  Number of rows    : {n_rows:,}")
print(f"  Number of columns : {n_cols}")

# ---------------------------------------------------------------------------
# 5. Schema — exact column names and data types
# ---------------------------------------------------------------------------
print("\n--- 3. Schema (Column Names & Data Types) ---")
print(f"  {'Column Name':<20} {'Pandas dtype':<15} {'Python type (sample)':<25}")
print(f"  {'-'*20} {'-'*15} {'-'*25}")
for col in df.columns:
    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
    py_type = type(sample_val).__name__ if sample_val is not None else "N/A"
    print(f"  {col:<20} {str(df[col].dtype):<15} {py_type:<25}")

# ---------------------------------------------------------------------------
# 6. Data quality checks
# ---------------------------------------------------------------------------
print("\n--- 4. Data Quality ---")

# Missing values
print("\n  Missing values per column:")
missing = df.isnull().sum()
total_missing = missing.sum()
for col in df.columns:
    print(f"    {col:<20} : {missing[col]}")
print(f"    {'TOTAL':<20} : {total_missing}")

# Duplicate rows
n_dupes = df.duplicated().sum()
print(f"\n  Exact duplicate rows : {n_dupes}")

# Null / NaN check (redundant but explicit)
null_count = df.isna().sum().sum()
print(f"  Total NaN values     : {null_count}")

# Check for non-numeric values in numeric columns
numeric_cols = ['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']
print("\n  Non-numeric / invalid value check:")
for col in numeric_cols:
    # Try converting to numeric; errors become NaN
    converted = pd.to_numeric(df[col], errors='coerce')
    n_invalid = converted.isna().sum() - df[col].isna().sum()  # new NaNs = non-numeric
    print(f"    {col:<20} : {n_invalid} non-numeric values")

# Check for physically impossible values
print("\n  Physically plausible range check:")
range_checks = {
    'pH':            (0, 14,    "pH scale is 0–14"),
    'TDS_ppm':       (0, 50000, "TDS should be non-negative"),
    'Turbidity_NTU': (0, 10000, "Turbidity should be non-negative"),
    'Temperature_C': (-10, 100, "Water temperature in plausible range"),
}
for col, (lo, hi, reason) in range_checks.items():
    out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"    {col:<20} : {out_of_range} values outside [{lo}, {hi}]  ({reason})")

# ---------------------------------------------------------------------------
# 7. Target variable analysis
# ---------------------------------------------------------------------------
print("\n--- 5. Target Variable Analysis ---")
target_col = 'Label'
print(f"  Target column      : {target_col}")
print(f"  Data type          : {df[target_col].dtype}")
unique_vals = df[target_col].unique()
print(f"  Unique values      : {list(unique_vals)}")
print(f"  Number of classes  : {len(unique_vals)}")

print("\n  Class distribution:")
counts = df[target_col].value_counts()
percs = df[target_col].value_counts(normalize=True) * 100
for label in counts.index:
    print(f"    {label:<10} : {counts[label]:>7,} samples  ({percs[label]:>6.2f}%)")

imbalance_ratio = counts.max() / counts.min()
print(f"\n  Imbalance ratio    : {imbalance_ratio:.2f}:1 (majority / minority)")

# ---------------------------------------------------------------------------
# 8. Numerical summary statistics
# ---------------------------------------------------------------------------
print("\n--- 6. Numerical Feature Summary ---")
desc = df[numeric_cols].describe(percentiles=[0.25, 0.50, 0.75]).T
desc['median'] = df[numeric_cols].median()
summary_cols = ['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']
print(desc[summary_cols].to_string(float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.2f}"))

# Per-class summary
print("\n  Per-class feature means:")
class_means = df.groupby(target_col)[numeric_cols].mean()
print(class_means.to_string(float_format=lambda x: f"{x:.2f}"))

# ---------------------------------------------------------------------------
# 9. Correlation matrix
# ---------------------------------------------------------------------------
print("\n--- 7. Feature Correlation Matrix ---")
# Include binary-encoded target for correlation
corr_df = df[numeric_cols].copy()
corr_df['Is_Unsafe'] = (df[target_col] == 'Unsafe').astype(int)
corr = corr_df.corr()
print(corr.to_string(float_format=lambda x: f"{x:+.4f}"))

# ---------------------------------------------------------------------------
# 10. Outlier detection (IQR method — report only)
# ---------------------------------------------------------------------------
print("\n--- 8. Outlier Detection (IQR Method) ---")
print("  NOTE: Outliers are REPORTED only, NOT removed.\n")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_below = (df[col] < lower).sum()
    n_above = (df[col] > upper).sum()
    n_total = n_below + n_above
    pct = (n_total / len(df)) * 100
    print(f"  {col}:")
    print(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"    Lower fence = {lower:.2f}, Upper fence = {upper:.2f}")
    print(f"    Below lower  : {n_below:,}")
    print(f"    Above upper  : {n_above:,}")
    print(f"    Total outliers: {n_total:,} ({pct:.2f}%)")
    print()

# ---------------------------------------------------------------------------
# 11. Summary assessment
# ---------------------------------------------------------------------------
print("=" * 70)
print("PHASE 1 ASSESSMENT")
print("=" * 70)
print(f"""
  Dataset            : {DATA_FILE.name}
  Rows               : {n_rows:,}
  Columns            : {n_cols}
  Missing values     : {total_missing}
  Duplicate rows     : {n_dupes}
  Target classes     : {list(unique_vals)}
  Class balance      : {imbalance_ratio:.2f}:1
  Timestamp column   : NOT PRESENT (no temporal variable exists)
  
  SUITABILITY: The dataset is suitable for binary classification
  of water quality (Safe vs. Unsafe) using pH, TDS, Turbidity,
  and Temperature as input features.
  
  NOTE: No timestamp or temporal ordering variable exists.
  Time-series forecasting or future-state prediction is NOT
  supported by this dataset.
""")
print("=" * 70)
print("PHASE 1 COMPLETE — Awaiting instruction for Phase 2.")
print("=" * 70)
