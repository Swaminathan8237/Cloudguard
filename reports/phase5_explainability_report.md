# Phase 5 Technical Report: Model Explainability, SHAP & Decision Rules

**Project Name**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services  
**Author**: Lead ML Production Engineer & Explainable-AI Specialist  
**Canonical Dataset**: [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv)  
**Executed Notebook**: [`notebooks/05_phase5_explainability.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/05_phase5_explainability.ipynb)  
**Date**: October 2026  

---

## Executive Summary

Phase 5 delivers a transparent, mathematically grounded, and production-ready **Explainability Layer** for the CloudGuard-Water system. The objective is to ensure that predictions are not opaque black-box outputs, but auditable physical decisions that can be interpreted by water treatment plant operators, environmental regulators, and cloud microservice consumers.

### Key Deliverables & Empirical Discoveries:
1. **Decision Tree Architecture & Rule Extraction**:
   - The primary production model (`models/decision_tree.joblib`, `max_depth=5`) was audited: actual depth = 5, 14 terminal leaf nodes, 27 total nodes.
   - Text rules were exported to [`reports/decision_tree_rules.txt`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/decision_tree_rules.txt). The primary physical splits governing water classification are **$TDS \le 211.05\,\text{ppm}$** and **$\text{Turbidity} \le 4.49\,\text{NTU}$**.
2. **Global Feature Attribution (Impurity vs. SHAP)**:
   - *Decision Tree Impurity*: Greedy single-tree optimization allocates **99.22%** of importance to `TDS_ppm` and **0.78%** to `Turbidity_NTU`, with `pH` (0.00%) and `Temperature_C` (0.00%) receiving zero splits.
   - *Random Forest SHAP*: Game-theoretic Shapley decomposition (`shap.TreeExplainer` across $N=1,000$ stratified training samples) reveals a more nuanced multi-tree reality: `TDS_ppm` accounts for **64.52%** (mean |SHAP| = 0.2791), `Turbidity_NTU` accounts for **28.11%** (0.1216), `Temperature_C` accounts for **5.35%** (0.0231), and `pH` accounts for **2.02%** (0.0087).
3. **Local Explanations**:
   - Evaluated representative samples from the canonical dataset and held-out test partition:
     - **Safe Conforming (Row 0)**: $pH=7.38, TDS=138.7, Turbidity=0.25, Temp=20.25^\circ\text{C}$ $\to$ Classified as **Safe (100.00%)** via 5 consecutive sub-threshold comparisons.
     - **Unsafe Contaminated (Row 30240)**: $pH=6.89, TDS=370.1, Turbidity=11.13, Temp=22.25^\circ\text{C}$ $\to$ Classified as **Unsafe (100.00%)** via root split $TDS > 202.75$ and terminal split $TDS > 211.05$.
      - **Borderline False Negative (Row 55037)**: $pH=7.05, TDS=192.3, Turbidity=3.95, Temp=24.26^\circ\text{C}$ $\to$ Ground truth `Unsafe`, but model predicts `Safe` ($P(\text{Safe})=99.48\%$). The explainability engine identifies this as a statistical classification (not a potability certification) and automatically triggers a retrospective `ALERT: KNOWN MODEL FALSE NEGATIVE` with an `ACTION REQUIRED` advisory requiring comprehensive certified laboratory testing because physical sensors alone cannot detect microbial pathogens, toxic heavy metals, or dataset boundary noise.
4. **Decision-Boundary Sensitivity**:
   - Perturbing TDS across 205.0 to 216.0 ppm demonstrates a deterministic step switch in the Decision Tree at **211.05 ppm**, whereas the Random Forest exhibits a smooth, sigmoidal probability transition between 208 and 214 ppm.
   - Perturbing Turbidity across 4.20 to 4.70 NTU reveals a step transition at **4.49 NTU**.
5. **Reusable Code Module & Defensive Safety Logic**:
   - Implemented [`src/explain.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\src\explain.py) with input validation, path traversal, empirical probability extraction, non-certifying advisory wording, and retrospective error detection.
   - Validated by 17 automated unit tests in [`tests/test_explain.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\tests\test_explain.py) (100% pass rate).

---

## 1. Project Context & Model Architecture Audit

### 1.1 Model Selection Rationale
In Phase 4, the **Decision Tree (`max_depth=5`)** was selected as the primary production model over Random Forest and Logistic Regression because:
1. **Auditable White-Box Logic**: Real-world municipal water monitoring demands verifiable physical rules over uninterpretable ensemble voting.
2. **Performance Parity**: Achieved identical test accuracy (**99.990%**, 20,158 / 20,160 correct) and identical Unsafe Recall (**99.993%**, 1 false negative) to the 100-tree Random Forest.
3. **Microsecond Latency**: Inference executes in $<0.02$ ms with an in-memory footprint of ~5 KB, making it ideal for cloud containerization and edge ESP32 deployment.

### 1.2 Model Artifact Audit

| Attribute | Primary Model: `decision_tree.joblib` | Secondary Model: `random_forest.joblib` |
| :--- | :--- | :--- |
| **Model Class** | `sklearn.tree.DecisionTreeClassifier` | `sklearn.ensemble.RandomForestClassifier` |
| **Target Classes** | `[0, 1]` (`0 = Safe`, `1 = Unsafe`) | `[0, 1]` (`0 = Safe`, `1 = Unsafe`) |
| **Input Features** | `['pH', 'TDS_ppm', 'Turbidity_NTU', 'Temperature_C']` | 4 features |
| **Max Depth Parameter** | 5 | 8 |
| **Actual Tree Depth** | 5 | 8 |
| **Terminal Leaves** | 14 leaves | Varies across 100 trees |
| **Total Node Count** | 27 nodes (13 internal decision splits, 14 leaves) | ~2,500 total nodes |

---

## 2. Decision Tree Rule Extraction

The complete rule set was extracted using `sklearn.tree.export_text` and saved in [`reports/decision_tree_rules.txt`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/decision_tree_rules.txt). The tree structure decomposes into clear physical boundaries:

```
|--- TDS_ppm <= 202.75
|   |--- Turbidity_NTU <= 4.49
|   |   |--- Turbidity_NTU <= 4.42
|   |   |   |--- TDS_ppm <= 202.35
|   |   |   |   |--- Turbidity_NTU <= 3.56 -> class: 0 (Safe)
|   |   |   |   |--- Turbidity_NTU >  3.56 -> class: 0 (Safe) [Leaf #6: 99.48% Safe]
|   |   |   |--- TDS_ppm >  202.35
|   |   |   |   |--- Turbidity_NTU <= 3.61 -> class: 0 (Safe)
|   |   |   |   |--- Turbidity_NTU >  3.61 -> class: 1 (Unsafe)
|   |   |--- Turbidity_NTU >  4.42
|   |   |   |--- TDS_ppm <= 183.95 -> class: 0 (Safe)
|   |   |   |--- TDS_ppm >  183.95 -> class: 1 (Unsafe)
|   |--- Turbidity_NTU >  4.49 -> class: 1 (Unsafe) [Leaf #13: 100% Unsafe]
|--- TDS_ppm >  202.75
|   |--- TDS_ppm <= 209.15
|   |   |--- Turbidity_NTU <= 3.91
|   |   |   |--- Turbidity_NTU <= 3.50 -> class: 0 (Safe)
|   |   |   |--- Turbidity_NTU >  3.50
|   |   |   |   |--- TDS_ppm <= 205.50 -> class: 1 (Unsafe)
|   |   |   |   |--- TDS_ppm >  205.50 -> class: 0 (Safe)
|   |   |--- Turbidity_NTU >  3.91 -> class: 1 (Unsafe)
|   |--- TDS_ppm >  209.15
|   |   |--- TDS_ppm <= 211.05
|   |   |   |--- Turbidity_NTU <= 3.35 -> class: 0 (Safe)
|   |   |   |--- Turbidity_NTU >  3.35 -> class: 1 (Unsafe)
|   |   |--- TDS_ppm >  211.05 -> class: 1 (Unsafe) [Leaf #26: 100% Unsafe]
```

### Visual Representation
The full decision tree is rendered in [`reports/figures/decision_tree_explainability.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_tree_explainability.png).

---

## 3. Global Feature Attribution: Gini Impurity vs. SHAP

### 3.1 Comparison of Methods

| Feature Name | DT Impurity Importance (%) | RF Impurity Importance (%) | RF SHAP Mean \|Value\| | RF SHAP Relative Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **`TDS_ppm`** | **99.22%** | **67.81%** | **0.2791** | **64.52%** |
| **`Turbidity_NTU`** | **0.78%** | **25.16%** | **0.1216** | **28.11%** |
| **`Temperature_C`** | 0.00% | 5.41% | 0.0231 | 5.35% |
| **`pH`** | 0.00% | 1.63% | 0.0087 | 2.02% |

### 3.2 Methodological Divergence Explained
- **Why Decision Tree Gives 99.2% to TDS**:
  The CART algorithm uses greedy recursive splitting. At the root, splitting on `TDS_ppm <= 202.75` purifies the majority of Unsafe observations. Because `TDS_ppm` and `Turbidity_NTU` explain almost all variance, `pH` and `Temperature_C` are never selected for splits.
- **Why Random Forest & SHAP Allocate Importance to pH and Temperature**:
  Random Forest randomly samples a subset of features at each node (`max_features='sqrt'`). In trees where `TDS_ppm` is excluded from the candidate split pool, the algorithm is forced to split on `Turbidity_NTU`, `Temperature_C`, or `pH`. SHAP averages marginal contributions across all feature permutations, proving that temperature and pH carry secondary predictive associations even though the single greedy tree does not require them.
- Figures saved:
  - [`reports/figures/decision_tree_feature_importance.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/decision_tree_feature_importance.png)
  - [`reports/figures/random_forest_shap_importance.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/random_forest_shap_importance.png)
  - [`reports/figures/random_forest_shap_summary.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/random_forest_shap_summary.png)
  - [`reports/random_forest_shap_importance.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/random_forest_shap_importance.csv)

---

## 4. Local Explanations for Real Observations

We audited three representative observations from the canonical dataset and held-out test partition:

| Observation Category | Row Index | Sensor Inputs (`pH`, `TDS`, `Turb`, `Temp`) | Ground Truth | DT Prediction (Probabilities) | RF Prediction (Probabilities) | RF Local SHAP (Class: Unsafe) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Safe Conforming** | **0** | `7.38, 138.7 ppm, 0.25 NTU, 20.25°C` | `Safe` | **`Safe`** ($P=1.000$) | **`Safe`** ($P=1.000$) | `TDS: -0.4926, Turb: -0.0945, Temp: -0.1098, pH: -0.0033` (all pulling toward Safe) |
| **Unsafe Contaminated** | **30240** | `6.89, 370.1 ppm, 11.13 NTU, 22.25°C` | `Unsafe` | **`Unsafe`** ($P=1.000$) | **`Unsafe`** ($P=1.000$) | `TDS: +0.1680, Turb: +0.1586, Temp: -0.0230, pH: -0.0039` (TDS & Turb push Unsafe) |
| **Borderline False Negative** | **55037** | `7.05, 192.3 ppm, 3.95 NTU, 24.26°C` | `Unsafe` | **`Safe`** ($P=0.9948$) | **`Safe`** ($P=0.7800$) | `TDS: -0.5443, Turb: +0.0613, Temp: +0.0072, pH: -0.0044` (TDS dominates Safe prediction) |

### Audit of the Test False Negative (Row 55037)
Row 55037 was the single Unsafe observation misclassified as Safe in the 20,160-sample test set. 
- *Physical Measurements*: pH 7.05 is neutral, TDS 192.3 ppm is well below the 211.05 ppm cutoff, and Turbidity 3.95 NTU is below 4.49 NTU.
- *Decision Path*: It followed `TDS_ppm <= 202.75` $\to$ `Turbidity_NTU <= 4.49` $\to$ `Turbidity_NTU <= 4.42` $\to$ `TDS_ppm <= 202.35` $\to$ `Turbidity_NTU > 3.56` into Leaf Node #6.
- *Leaf Distribution*: During training, 99.48% of samples reaching Leaf #6 were `Safe`. The model assigned an estimated 99.48% probability of Safe based on empirical training proportions.
- *Why the Recommendation Logic Was Corrected*:
  Previously, following the model prediction alone caused the system to output an inappropriate `NORMAL` recommendation. This has been corrected:
  1. **Model Classification Framing**: Predictions are explicitly identified as statistical model classifications, never as drinking-water safety certifications.
  2. **Non-Certifying Phrasing**: A `Safe` prediction never produces unsupported wording such as "water is safe to drink" or "parameters comply with official guidelines."
  3. **Empirical Probabilities**: Probabilities are described as estimated empirical training proportions and not guarantees.
  4. **Retrospective Error Detection**: Known retrospective discrepancies (actual `Unsafe` vs. predicted `Safe`) are audited and reported as `ALERT: KNOWN MODEL FALSE NEGATIVE`.
  5. **Mandatory Laboratory Testing Notice**: The operational recommendation overrules the Safe prediction with `ACTION REQUIRED: Do NOT treat this water as potable. Comprehensive laboratory microbiological, toxicological, and chemical testing is strictly required.`
- Full local explanation table saved in [`reports/local_explanations.csv`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\local_explanations.csv).

---

## 5. Decision-Boundary Sensitivity Analysis

We evaluated the robustness and probability dynamics of the models under small physical perturbations across the two primary decision boundaries:

### 5.1 TDS Sensitivity Sweep (205.0 to 216.0 ppm)
Fixed baseline: $pH=7.20$, $\text{Turbidity}=2.00\,\text{NTU}$, $\text{Temperature}=22.00^\circ\text{C}$.
- At **211.0 ppm**: Decision Tree outputs $P(\text{Unsafe}) = 0.0000$ (Leaf Node #24). Random Forest outputs $P(\text{Unsafe}) = 0.3298$.
- At **211.5 ppm**: Decision Tree abruptly steps to $P(\text{Unsafe}) = 1.0000$ (Leaf Node #26). Random Forest crosses the 0.50 threshold to $P(\text{Unsafe}) = 0.5898$.
- *Conclusion*: The Decision Tree exhibits an exact step response at 211.05 ppm. The Random Forest smooths the boundary into a continuous sigmoidal transition between 208 and 214 ppm.

### 5.2 Turbidity Sensitivity Sweep (4.20 to 4.70 NTU)
Fixed baseline: $pH=7.20$, $TDS=150.0\,\text{ppm}$, $\text{Temperature}=22.00^\circ\text{C}$.
- Below **4.42 NTU**: Traverses into Leaf Node #6 ($P(\text{Safe}) = 0.9948$).
- Between **4.42 and 4.49 NTU**: Traverses into Leaf Node #11 ($P(\text{Safe}) = 1.0000$).
- At **4.50 NTU and above**: Traverses into Leaf Node #13 ($P(\text{Unsafe}) = 1.0000$).
- Data saved in [`reports/decision_boundary_sensitivity.csv`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\decision_boundary_sensitivity.csv); curves plotted in [`reports/figures/decision_boundary_sensitivity.png`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\figures\decision_boundary_sensitivity.png).

---

## 6. Prediction Probabilities and Calibration Assessment

- **Decision Tree Brier Score on Test Set ($N=20,160$)**: **`0.000109`**
- **Random Forest Brier Score on Test Set ($N=20,160$)**: **`0.000171`**
- **Calibration Assessment**:
  Because the test accuracy is ~99.99% and Brier scores are on the order of $10^{-4}$, the empirical leaf proportions are already well-calibrated confidence estimators. Applying post-hoc calibration (e.g. Platt scaling or Isotonic regression) is unnecessary and would alter the white-box probability mapping of the Decision Tree.

---

## 7. Reusable Software Module & Automated Testing

### 7.1 Module Architecture (`src/explain.py`)
The newly built module exposes 7 key public functions:
1. `load_model(model_path)`: Safe loader with path resolution.
2. `validate_input(features)`: Validates dictionaries, DataFrames, and lists against schema, range, and type constraints.
3. `predict_with_probabilities(model, X)`: Returns class indices, calibrated probabilities, and string labels.
4. `extract_decision_path(model, sample)`: Deconstructs Decision Tree node traversal into human-auditable steps.
5. `get_feature_importances(model)`: Returns sorted global feature importance percentages.
6. `generate_human_explanation(explanation_dict, actual_label=None)`: Produces structured multi-line plain English operational summaries with strict non-certifying language, empirical probability labeling, retrospective false-negative error analysis, and explicit advisories stating that certified laboratory testing (WHO, BIS IS 10500) remains necessary.
7. `explain_sample(model, features, actual_label=None)`: End-to-end wrapper combining validation, prediction, path extraction, retrospective ground-truth comparison, and report formatting.

### 7.2 Automated Test Suite (`tests/test_explain.py`)
We ran the unit test suite across 17 test cases:
- Model loading verification and failure handling.
- Input validation across dicts, DataFrames, and lists.
- Missing feature, null, NaN, and dimension error rejection.
- Exact agreement with `model.predict` and `model.predict_proba`.
- Correct probability mapping ($0 = \text{Safe}, 1 = \text{Unsafe}$).
- Correct decision path extraction and node threshold comparisons.
- Formatting of human-readable text ensuring non-certifying phrasing and no claims of potability or official compliance.
- Explicit verification that estimated probabilities are not described as guarantees.
- Retrospective false-negative detection and error explanation for Row 55037.
- Feature importance calculations.
- Artifact preservation check.
- **Result**: **17 passed in 0.249s (0 failures, 0 errors).**

---

## 8. Critical Scientific Limitations

1. **Physical Scope Limitation**: Physical sensors measuring pH, TDS, Turbidity, and Temperature **cannot detect microbiological pathogens (*E. coli*, Salmonella, noroviruses) or toxic heavy metals (arsenic, mercury, lead)**. Predictions classify conformity to physical sensor thresholds, but do NOT certify water as safe for human consumption.
2. **Correlation vs. Physical Causation**: High SHAP or Gini importance values denote strong predictive utility within this dataset, not physical or biochemical causation.
3. **Statutory Standards vs. Empirical Thresholds**: The learned threshold of 211.05 ppm TDS is an **empirical characteristic of this specific dataset**, not a statutory standard of WHO, BIS IS 10500, or US EPA.

---

## 9. Summary & Next Steps

Phase 5 has been fully verified and documented. All required artifacts, test suites, figures, and notebooks are operational.

### Recommended Next Phase:
- **Phase 6: Cloud Prediction Microservice & API Development**:
  Package the trained `decision_tree.joblib` and `src/explain.py` into a lightweight, containerized FastAPI/Flask microservice exposing endpoints for single-reading predictions, batch predictions, and live explainability payloads.
