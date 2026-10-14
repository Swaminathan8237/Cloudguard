# Phase 3 — Data Preprocessing & Model Training Report

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

| Model | Accuracy | Precision (Safe) | Recall (Safe) | F1 (Safe) | Precision (Unsafe) | Recall (Unsafe) | F1 (Unsafe) | Macro F1 | Weighted F1 | TN | FP | FN | TP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9999 | 0.9997 | 1.0000 | 0.9998 | 1.0000 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 6,048 | 0 | 2 | 14,110 |
| Decision Tree | 0.9999 | 0.9998 | 0.9998 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 6,047 | 1 | 1 | 14,111 |
| Random Forest | 0.9999 | 0.9998 | 0.9998 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 6,047 | 1 | 1 | 14,111 |

### 3.2 Confusion Matrices & Error Breakdown

| Model | True Negatives (TN)<br>(Safe → Safe) | False Positives (FP)<br>(Safe → Unsafe) | False Negatives (FN)<br>(Unsafe → Safe) | True Positives (TP)<br>(Unsafe → Unsafe) |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 6,048 | 0 | 2 | 14,110 |
| **Decision Tree** | 6,047 | 1 | 1 | 14,111 |
| **Random Forest** | 6,047 | 1 | 1 | 14,111 |

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
  - **Random Forest**: Produced only 1 False Negative out of 14,112 unsafe samples (Recall: 99.993%).
  - **Decision Tree**: Produced only 1 False Negative out of 14,112 unsafe samples (Recall: 99.993%).
  - **Logistic Regression**: Produced only 2 False Negatives out of 14,112 unsafe samples (Recall: 99.986%).

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
  - **`TDS_ppm`**: 67.81%
  - **`Turbidity_NTU`**: 25.16%
  - **`Temperature_C`**: 5.41%
  - **`pH`**: 1.63%

---

## 6. Persisted Model Artifacts (joblib)

The trained models have been saved to [`models/`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/) for downstream deployment:
- [`models/logistic_regression.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/logistic_regression.joblib)
- [`models/decision_tree.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/decision_tree.joblib)
- [`models/random_forest.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/random_forest.joblib)

---

**Phase 3 Complete.** Ready for Phase 4 (Detailed Evaluation, Model Comparison & Selection).
