"""
Script: src/add_engineered_columns.py
Purpose: Temporarily add the 5 engineered columns to data/water_quality_dataset1.csv
         as requested for the course document / lab alignment:
           1. pH_change
           2. TDS_change
           3. rolling_mean
           4. rolling_std
           5. unsafe_in_next_20_minutes

Note: A clean backup of the original 5-column CSV is preserved at:
      data/water_quality_dataset1_original.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path("data/water_quality_dataset1.csv")

def add_columns():
    print(f"Loading dataset from: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)
    print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Original columns: {list(df.columns)}")
    
    # 1. pH_change: First-order rate of change (pH_t - pH_{t-1})
    df['pH_change'] = df['pH'].diff().fillna(0.0).round(4)
    
    # 2. TDS_change: First-order rate of change (TDS_t - TDS_{t-1})
    df['TDS_change'] = df['TDS_ppm'].diff().fillna(0.0).round(2)
    
    # 3. rolling_mean: Rolling 10-observation window (20 min) mean of TDS
    df['rolling_mean'] = df['TDS_ppm'].rolling(window=10, min_periods=1).mean().round(2)
    
    # 4. rolling_std: Rolling 10-observation window (20 min) standard deviation of TDS
    df['rolling_std'] = df['TDS_ppm'].rolling(window=10, min_periods=1).std().fillna(0.0).round(2)
    
    # 5. unsafe_in_next_20_minutes: Binary indicator (1 = Unsafe, 0 = Safe)
    # Looking ahead 10 observations (10 * 2 min = 20 min).
    # If in the next 10 steps water is Unsafe, set to 1, else 0.
    is_unsafe = (df['Label'] == 'Unsafe').astype(int)
    # Lookahead rolling max (inverted shift)
    unsafe_future = is_unsafe.iloc[::-1].rolling(window=10, min_periods=1).max().iloc[::-1].astype(int)
    df['unsafe_in_next_20_minutes'] = unsafe_future
    
    # Save back to CSV
    df.to_csv(DATA_PATH, index=False)
    print(f"\nUpdated shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Updated columns: {list(df.columns)}")
    print("\nFirst 5 rows of updated dataset:")
    print(df.head(5).to_string())

if __name__ == "__main__":
    add_columns()
