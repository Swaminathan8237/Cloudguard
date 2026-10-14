"""
CloudGuard-Water: Baseline ML Models Evaluation
Models: Logistic Regression, Decision Tree, Random Forest
Dataset: Water Quality Dataset from Water Sources in Kenya (100,800 rows)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_data_path():
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "water_quality_dataset1.csv",
        Path(__file__).resolve().parent / "data" / "water_quality_dataset1.csv",
        Path("data/water_quality_dataset1.csv"),
        Path("water_quality_dataset1.csv"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(Path("data") / "water_quality_dataset1.csv")

def main():
    print("=" * 60)
    print("CLOUDGUARD-WATER: BASELINE MODEL BENCHMARKING")
    print("=" * 60)
    
    dataset_file = get_data_path()
    print(f"Loading dataset from: {dataset_file}")
    df = pd.read_csv(dataset_file)
    X = df[['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']]
    y = (df['Label'] == 'Unsafe').astype(int) # 1 = Unsafe, 0 = Safe
    
    # 1. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    }
    
    print("\n--- 5-Fold Stratified Cross-Validation Performance ---")
    results = {}
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            'Accuracy': (res['test_accuracy'].mean(), res['test_accuracy'].std()),
            'Precision': (res['test_precision'].mean(), res['test_precision'].std()),
            'Recall': (res['test_recall'].mean(), res['test_recall'].std()),
            'F1': (res['test_f1'].mean(), res['test_f1'].std()),
        }
        print(f"\nModel: {name}")
        print(f"  Accuracy : {res['test_accuracy'].mean():.4f} +/- {res['test_accuracy'].std():.4f}")
        print(f"  Precision: {res['test_precision'].mean():.4f} +/- {res['test_precision'].std():.4f}")
        print(f"  Recall   : {res['test_recall'].mean():.4f} +/- {res['test_recall'].std():.4f}")
        print(f"  F1-Score : {res['test_f1'].mean():.4f} +/- {res['test_f1'].std():.4f}")

    # 2. Train a final Random Forest to inspect Feature Importances
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Random Forest Feature Importances ---")
    for feat, imp in importances.items():
        print(f"  {feat:<18} : {imp:.4f} ({imp*100:.2f}%)")

if __name__ == "__main__":
    main()
