# CloudGuard-Water: Proactive Water Quality Deterioration Prediction Using IoT Data and Machine Learning

**Course**: Machine Learning for Cloud Services  
**Course Code / Assignment**: Project Report 1  
**Candidate Name**: Swaminathan (Unique ID / Roll No: `E0224033`)  
**Submission Date**: 05 September 2026  

---

## 1. Project Overview

**CloudGuard-Water** extends traditional reactive water quality classification into a **proactive, cloud-based early-warning deterioration prediction system**. 

While conventional IoT water monitoring only classifies current water safety ($x_t \rightarrow \text{Safe} / \text{Unsafe}$), **CloudGuard-Water** utilizes historical telemetry (lag features, rolling trends, and rate of change) to answer:
> *"Given sensor observations up to time $t$, will this water source become unsafe 20 minutes from now?"*

This provides critical lead time for water utilities to automatically close intake valves, initiate chemical treatment, or dispatch emergency field maintenance before contaminated water reaches distribution conduits.

---

## 2. Directory Structure

```text
LB_Proj/
├── data/
│   └── water_quality_dataset1.csv     # Raw dataset from Mendeley Data (100,800 rows x 5 cols, 2.83 MB)
│
├── docs/
│   ├── Report_1.md                    # Complete academic Report 1 draft (Markdown)
│   ├── Project Work - Data Dictionary.docx  # Completed course submission template (Word document)
│   ├── Project Work - Data Dictionary.pdf   # Exported course template (PDF)
│   ├── PROJECT USE CASE (1).pdf       # Official course syllabus and use-case catalog
│   └── Project Work - Data Dictionary_backup.docx # Clean backup of initial course template
│
├── src/
│   ├── audit_dataset.py               # Complete Phase 1 dataset inspection and verification script
│   ├── generate_eda_plots.py          # Publication-ready Exploratory Data Analysis (EDA) visualizer
│   ├── train_baseline.py              # 5-fold cross-validation for Logistic Regression, Decision Tree, RF
│   └── populate_docx.py               # Automated script to populate course Word templates
│
├── outputs/
│   └── figures/                       # Generated publication-quality figures (300 DPI)
│       ├── class_distribution.png     # Target class distribution (Safe: 30%, Unsafe: 70%)
│       ├── sensor_distributions.png   # 2x2 grid of physicochemical parameter distributions
│       ├── correlation_matrix.png     # Feature correlation heatmap with safety status
│       └── feature_importance.png     # Baseline Random Forest Gini feature importances
│
├── Report_1.md                        # Root copy of Report 1 for quick IDE access
└── README.md                          # Project documentation and reproduction guide
```

---

## 3. Dataset Information

- **Dataset Name**: Water Quality Dataset from Water Sources in Kenya
- **Source**: Mendeley Data (Elsevier)
- **DOI**: `10.17632/575ctg3nnr.2` (Version 2, April 2026)
- **Contributors**: Samson Otieno Ooko, Laura Cheptegei, Simon Karume (Kabarak University, Kenya)
- **Cadence**: 2-minute sampling interval across 10 distinct 14-day monitoring campaigns
- **Shape**: 100,800 rows $\times$ 5 columns
- **Storage Size**: 2.83 MB on disk (2,964,409 bytes); 9.07 MB in RAM
- **Missing Values**: 0 across all columns
- **Features**:
  - `pH` (Range: 6.09 – 9.96)
  - `TDS_ppm` (Range: 28.6 – 2806.1 ppm)
  - `Turbidity_NTU` (Range: 0.00 – 72.75 NTU)
  - `Temperature_C` (Range: 16.56 – 31.51 °C)
- **Target**: `Label` (`Safe`: 30,240 / 30.0%, `Unsafe`: 70,560 / 70.0%)

---

## 4. How to Run and Reproduce

Ensure you are using Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `python-docx` installed.

### 1. Run the Dataset Audit (Phase 1)
```powershell
py -3.10 src/audit_dataset.py
```
*Verifies file hash, column types, zero missing values, zero duplicates, class distribution, and the 10 sequential 14-day deployment blocks.*

### 2. Generate EDA Visualizations
```powershell
py -3.10 src/generate_eda_plots.py
```
*Outputs 4 high-resolution plots to `outputs/figures/`.*

### 3. Evaluate Baseline Classifiers
```powershell
py -3.10 src/train_baseline.py
```
*Evaluates Logistic Regression, Decision Tree, and Random Forest via 5-Fold Stratified Cross-Validation, confirming TDS (67.4%) and Turbidity (25.7%) as primary baseline predictors.*

### 4. Regenerate Course Word Template
```powershell
py -3.10 src/populate_docx.py
```
*Populates `docs/Project Work - Data Dictionary.docx` with verified course metadata, student details, and data dictionary.*
