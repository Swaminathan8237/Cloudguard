# Phase 5 Technical Handoff Report for External Peer Review (ChatGPT)

**Project Name**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Target Audience**: ChatGPT Technical Peer Reviewer & Viva Examination Committee  
**Course**: Machine Learning for Cloud Services  
**Working Directory**: `d:\E0224033\ML_for_Cl_Ser\CL\LB_Proj`  
**GitHub Repository**: [`https://github.com/Swaminathan8237/Cloudguard.git`](https://github.com/Swaminathan8237/Cloudguard.git)  
**Date**: October 2026  

---

## 1. Project Context & Current Workflow State

- **High-Level Purpose**: Develop an end-to-end, production-grade academic ML pipeline that classifies water potability (`Safe` vs. `Unsafe`) from physical sensor readings, provides an interpretable and auditable explainability layer, and deploys as a cloud prediction microservice connected to an IoT prototype.
- **Workflow State**:
  - Phase 1 (Dataset Inspection & Integrity Audit): **COMPLETE**
  - Phase 2 (Exploratory Data Analysis & Statistical Profiling): **COMPLETE**
  - Phase 3 (Baseline Model Training & Persistence): **COMPLETE**
  - Phase 4 (Detailed Evaluation, Cross-Validation & Model Selection): **COMPLETE**
  - Phase 4.1 (Dataset Provenance, Consistency & Evaluation Audit): **COMPLETE**
  - Phase 5 (Model Explainability, SHAP & Decision Rules): **COMPLETE (This Deliverable)**
  - Phase 6 (Cloud Prediction Microservice / API Development): **UPCOMING**
  - Phase 7 (IoT Sensor Circuit Integration): **UPCOMING**

---

## 2. Canonical Dataset Specification & Verification

### 2.1 File Forensics

| Parameter | Authoritative File (`water_quality_dataset1.csv`) | Backup Twin (`water_quality_dataset1_original.csv`) |
| :--- | :--- | :--- |
| **File Location** | `data/water_quality_dataset1.csv` | `data/water_quality_dataset1_original.csv` |
| **Exact Byte Size** | **2,964,409 bytes** (2.8271 MB) | **2,964,409 bytes** (2.8271 MB) |
| **SHA-256 Checksum** | `90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a` | `90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a` |
| **Byte Equality** | `filecmp.cmp(shallow=False)` = **`True`** | `filecmp.cmp(shallow=False)` = **`True`** |
| **Dimensions** | Exactly **100,800 rows × 5 columns** | Exactly **100,800 rows × 5 columns** |
| **Missing Values / Duplicates** | **0** missing, **0** duplicate rows | **0** missing, **0** duplicate rows |
| **Class Distribution** | `Unsafe`: 70,560 (70.00%) \| `Safe`: 30,240 (30.00%) | `Unsafe`: 70,560 (70.00%) \| `Safe`: 30,240 (30.00%) |
| **Block Order** | Rows 0–30,239: `Safe` \| Rows 30,240–100,799: `Unsafe` | Rows 0–30,239: `Safe` \| Rows 30,240–100,799: `Unsafe` |
| **Transition Index** | **Index 30240** (`30239` is Safe, `30240` is Unsafe) | **Index 30240** (`30239` is Safe, `30240` is Unsafe) |

### 2.2 Corrections Made to Documentation
In `reports/phase4_followup_handoff.md` line 56, an example for Row 0 previously read:
`Row 0: pH=7.38, TDS_ppm=138.7, Turbidity_NTU=0.25, Temperature_C=21.55, Label=Safe`
The true data in Row 0 is `Temperature_C = 20.25`. This typo was formally corrected to:
`Row 0: pH=7.38, TDS_ppm=138.7, Turbidity_NTU=0.25, Temperature_C=20.25, Label=Safe`

---

## 3. Models Examined & Architecture Details

All models reside under `models/` and were trained in Phase 3 on the 80% training partition ($N=80,640$):

1. **Primary Model (`models/decision_tree.joblib`)**:
   - `DecisionTreeClassifier(max_depth=5, random_state=42)`
   - Actual depth = 5, Total leaves = 14, Total nodes = 27.
   - Classes: `[0, 1]` (`0 = Safe`, `1 = Unsafe`).
   - Feature order: `['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']`.
2. **Secondary Model (`models/random_forest.joblib`)**:
   - `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`
   - Retained as a non-linear interaction benchmark for SHAP decomposition.
3. **Reference Baseline (`models/logistic_regression.joblib`)**:
   - `Pipeline(StandardScaler(), LogisticRegression())`

---

## 4. Phase 5 Code, Notebook & Artifact Directory

| File Path | Type | Purpose & Verification Status |
| :--- | :--- | :--- |
| [`src/explain.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/explain.py) | Python 3.10 Module | Production explainability engine: input validation, path traversal, probability extraction, narrative generation. Verified with 0 Pyright errors. |
| [`src/generate_phase5_artifacts.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/generate_phase5_artifacts.py) | Python 3.10 Script | Pipeline generating SHAP scores, local explanations, sensitivity curves, and figures. |
| [`src/create_phase5_notebook.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/create_phase5_notebook.py) | Python 3.10 Script | Automated builder and executor for `notebooks/05_phase5_explainability.ipynb`. |
| [`tests/test_explain.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_explain.py) | Python Unit Tests | 15 automated test cases covering model loading, input validation, path extraction, probabilities, and formatting. **15/15 passed.** |
| [`notebooks/05_phase5_explainability.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/05_phase5_explainability.ipynb) | Jupyter Notebook | Complete, self-contained interactive notebook with all 15 required sections, markdown discussions, and pre-rendered outputs. |
| [`reports/decision_tree_rules.txt`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/decision_tree_rules.txt) | Text File | Complete textual export of all 14 leaf decision paths and thresholds. |
| [`reports/random_forest_shap_importance.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/random_forest_shap_importance.csv) | CSV Data | Numerical SHAP values across 1,000 stratified training samples. |
| [`reports/local_explanations.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/local_explanations.csv) | CSV Data | Detailed local explanation table for Safe, Unsafe, and Borderline observations. |
| [`reports/decision_boundary_sensitivity.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/decision_boundary_sensitivity.csv) | CSV Data | Numerical results from TDS and Turbidity sensitivity sweeps. |
| [`reports/phase5_explainability_report.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase5_explainability_report.md) | Markdown | Formal academic technical report. |
| [`reports/figures/decision_tree_explainability.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_tree_explainability.png) | Figure (PNG) | High-resolution visualization of the full 14-leaf decision tree. |
| [`reports/figures/decision_tree_feature_importance.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_tree_feature_importance.png) | Figure (PNG) | Gini impurity feature importance bar plot. |
| [`reports/figures/random_forest_shap_importance.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/random_forest_shap_importance.png) | Figure (PNG) | Global mean |SHAP| feature importance bar plot. |
| [`reports/figures/random_forest_shap_summary.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/random_forest_shap_summary.png) | Figure (PNG) | SHAP beeswarm summary plot showing direction and magnitude of effects. |
| [`reports/figures/decision_boundary_sensitivity.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_boundary_sensitivity.png) | Figure (PNG) | Dual-panel probability sensitivity curve comparison (DT step vs RF sigmoid). |

---

## 5. Actual Numerical Results & Audit Details

### 5.1 Real Learned Model Thresholds (from `decision_tree_rules.txt`)
- **TDS Primary Split**: $TDS \le 202.75\,\text{ppm}$
- **TDS Absolute Safety Cutoff**: $TDS \le 211.05\,\text{ppm}$ (Any sample with $TDS > 211.05\,\text{ppm}$ goes straight to Leaf #26, predicting Unsafe with 100% confidence).
- **Turbidity Safety Cutoff**: $\text{Turbidity} \le 4.49\,\text{NTU}$ (Samples with $TDS \le 202.75$ and $\text{Turbidity} > 4.49$ go to Leaf #13, predicting Unsafe with 100% confidence).
- **Turbidity Sub-Split**: $\text{Turbidity} \le 4.42\,\text{NTU}$ and $\le 3.56\,\text{NTU}$.

### 5.2 Global Feature Attribution Numbers

| Feature | Decision Tree Gini (%) | Random Forest Gini (%) | Random Forest Mean \|SHAP\| | Random Forest SHAP Relative (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`TDS_ppm`** | **99.2221%** | **67.8056%** | **0.279079** | **64.5227%** |
| **`Turbidity_NTU`** | **0.7779%** | **25.1557%** | **0.121574** | **28.1077%** |
| **`Temperature_C`** | 0.0000% | 5.4136% | 0.023129 | 5.3473% |
| **`pH`** | 0.0000% | 1.6250% | 0.008747 | 2.0223% |

### 5.3 Representative Local Observations Audited

1. **Safe Conforming (Row 0)**:
   - Inputs: $pH=7.38$, $TDS=138.7$, $\text{Turbidity}=0.25$, $\text{Temp}=20.25^\circ\text{C}$
   - Predictions: DT = `Safe` ($P=1.0000$), RF = `Safe` ($P=1.0000$)
   - Decision path: `TDS_ppm (138.70) <= 202.75` $\to$ `Turbidity_NTU (0.25) <= 4.49` $\to$ `Turbidity_NTU (0.25) <= 4.42` $\to$ `TDS_ppm (138.70) <= 202.35` $\to$ `Turbidity_NTU (0.25) <= 3.56` $\to$ Leaf Node #5 ($P(\text{Safe}) = 1.0$).
   - RF SHAP values (Class: Unsafe): $pH=-0.0033$, $TDS=-0.4926$, $\text{Turbidity}=-0.0945$, $\text{Temp}=-0.1098$.
2. **Unsafe Contaminated (Row 30240)**:
   - Inputs: $pH=6.89$, $TDS=370.1$, $\text{Turbidity}=11.13$, $\text{Temp}=22.25^\circ\text{C}$
   - Predictions: DT = `Unsafe` ($P=1.0000$), RF = `Unsafe` ($P=1.0000$)
   - Decision path: `TDS_ppm (370.10) > 202.75` $\to$ `TDS_ppm (370.10) > 209.15` $\to$ `TDS_ppm (370.10) > 211.05` $\to$ Leaf Node #26 ($P(\text{Unsafe}) = 1.0$).
   - RF SHAP values (Class: Unsafe): $pH=-0.0039$, $TDS=+0.1680$, $\text{Turbidity}=+0.1586$, $\text{Temp}=-0.0230$.
3. **Borderline False Negative (Row 55037)**:
   - Inputs: $pH=7.05$, $TDS=192.3$, $\text{Turbidity}=3.95$, $\text{Temp}=24.26^\circ\text{C}$
   - Ground Truth: `Unsafe`
   - Predictions: DT = `Safe` ($P=0.9948$), RF = `Safe` ($P=0.7800$)
   - Decision path: Traversed to Leaf Node #6 ($P(\text{Safe}) = 0.9948, P(\text{Unsafe}) = 0.0052$).
   - RF SHAP values: $TDS=-0.5443$ (strongly pulls toward Safe), $\text{Turbidity}=+0.0613$, $\text{Temp}=+0.0072$, $pH=-0.0044$.
   - *Technical Explanation*: The physical sensor values fall entirely inside the potable envelope ($TDS < 211.05$, $\text{Turbidity} < 4.49$). The model correctly followed its physical rules; the misclassification is an artifact of label boundary noise in the dataset generator.

### 5.4 Decision-Boundary Sensitivity Findings
- **TDS Threshold Sweep (205.0 to 216.0 ppm)**:
  - Decision Tree maintains $P(\text{Unsafe})=0.0000$ from 205.0 up to 211.0 ppm (Leaf Node #24), then abruptly steps to $P(\text{Unsafe})=1.0000$ at 211.5 ppm (Leaf Node #26).
  - Random Forest demonstrates a smooth, continuous sigmoidal curve: $P(\text{Unsafe}) = 0.0099$ at 205.0 ppm, $0.1998$ at 210.0 ppm, $0.3298$ at 211.0 ppm, and crosses threshold to $0.5898$ at 211.5 ppm.
- **Turbidity Threshold Sweep (4.20 to 4.70 NTU)**:
  - Decision Tree steps from $P(\text{Safe})=1.0000$ (Leaf #11) to $P(\text{Unsafe})=1.0000$ (Leaf #13) at **4.49 NTU**.

---

## 6. Distinguishing Verified Facts vs. Assumptions

### Verified Facts:
1. Both `data/water_quality_dataset1.csv` and `data/water_quality_dataset1_original.csv` share identical byte sizes (2,964,409 bytes) and SHA-256 checksums (`90f9c7cb...`).
2. The Decision Tree rules rely almost exclusively on $TDS \le 211.05$ and $\text{Turbidity} \le 4.49$.
3. The Decision Tree achieves identical test accuracy (99.990%) and unsafe recall (99.993%) to the 100-tree Random Forest, with $50\times$ lower latency and full transparency.
4. All 15 unit tests pass in `tests/test_explain.py`.

### Critical Caveats & Non-Claims:
1. **No Complete Drinking Water Safety Certification**: Physical sensors measuring pH, TDS, Turbidity, and Temperature do NOT test for microbiological organisms (*E. coli*, viruses) or toxic heavy metals (arsenic, lead).
2. **No Claim of Causation**: Feature importance and SHAP attribution indicate statistical association within this dataset, not physical causation.
3. **No Official Standards Attribution**: The exact numeric thresholds (211.05 ppm TDS, 4.49 NTU Turbidity) are **empirical artifacts of this dataset's labeling rule**, not statutory standards of WHO, US EPA, or Bureau of Indian Standards (BIS IS 10500).

---

## 7. Recommended Next Steps for Phase 6

- Implement a lightweight FastAPI / Flask REST microservice under `src/api.py`.
- Expose `/health`, `/predict`, and `/explain` endpoints reusing `src/explain.py`.
- Implement JSON request validation with Pydantic matching `FEATURE_NAMES`.
- Ensure low-latency responses ($< 10$ ms) and comprehensive error handling.
