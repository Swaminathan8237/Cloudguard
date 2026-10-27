# CloudGuard-Water: Consolidated Project Notebook Audit, Correction & Verification Report

**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning  
**Course**: Machine Learning for Cloud Services (MLCS)  
**Author**: Lead Systems, ML Deployment & IoT Verification Specialist  
**Canonical Dataset**: `data/water_quality_dataset1.csv` (100,800 rows, 4 physical features, 1 target label)  
**Consolidated Notebook**: [`notebooks/CloudGuard_Water_Complete_Project.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/CloudGuard_Water_Complete_Project.ipynb)  
**Byte-Preserving Backup**: [`notebooks/CloudGuard_Water_Complete_Project_backup.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/CloudGuard_Water_Complete_Project_backup.ipynb)  
**Exported Document (PDF)**: [`reports/CloudGuard_Water_Complete_Project.pdf`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/CloudGuard_Water_Complete_Project.pdf) (33 pages, 1.24 MB)  
**Exported Document (HTML)**: [`reports/CloudGuard_Water_Complete_Project.html`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/CloudGuard_Water_Complete_Project.html) (753 KB)  
**Initial Audit Checklist**: [`reports/initial_audit_checklist.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/initial_audit_checklist.md)  
**Artifact Index**: [`reports/complete_project_artifact_index.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/complete_project_artifact_index.md)  
**Date**: October 2026  

---

## 1. Executive Summary & Verification Context

This report details the technical audit, repair, clean-kernel top-to-bottom execution, cryptographic integrity verification, and publication-grade export of the consolidated project notebook: [`notebooks/CloudGuard_Water_Complete_Project.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/CloudGuard_Water_Complete_Project.ipynb).

Rather than regenerating a disconnected notebook, the existing 16-section monograph was systematically audited against the repository's authoritative sources: the frozen model artifacts (`models/decision_tree.joblib`, `models/random_forest.joblib`, `models/logistic_regression.joblib`), the canonical dataset (`data/water_quality_dataset1.csv`), the active firmware (`firmware/cloudguard_esp32/cloudguard_esp32.ino` and `config.h`), and the verified regression test suite (`tests/`).

A byte-preserving backup was captured at [`notebooks/CloudGuard_Water_Complete_Project_backup.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/CloudGuard_Water_Complete_Project_backup.ipynb) prior to modification. All 13 original phase notebooks remain unaltered in `notebooks/`.

---

## 2. Identified Inconsistencies & Authoritative Resolutions

Twelve specific contradictions were audited, resolved, and verified against empirical code and artifact evidence:

### Issue 1: Decision Tree Depth, Structure, and Leaf Count
- **Contradiction**: Earlier draft text mentioned `max_depth=6`, 57 nodes, and 29 leaves, while the actual model artifact in execution output reported `max_depth=5`.
- **Authoritative Evidence**: Direct Python inspection of `models/decision_tree.joblib`:
  - `model_dt.max_depth` = **5**
  - `model_dt.get_depth()` = **5**
  - `model_dt.tree_.node_count` = **27**
  - `model_dt.get_n_leaves()` = **14**
- **Resolution**: Updated all prose, section titles, architecture callouts, comparison tables, and conclusions across Sections 1, 4, 5, 7, and 15 to state configured `max_depth=5`, actual depth 5, 27 nodes, and 14 leaves. Added programmatic assertions confirming these exact values during execution.

### Issue 2: EDA Descriptive Statistics Discrepancy
- **Contradiction**: The narrative text reported placeholder means: TDS 645.2 ppm, Turbidity 10.45 NTU, pH 7.34, Temperature 24.08°C. The executed descriptive statistics table computed different values.
- **Authoritative Evidence**: Recomputed via `df.describe()` on `data/water_quality_dataset1.csv` ($N=100,800$):
  - **pH**: Mean **$7.59$** (std: $0.80$, min: $6.09$, 25%: $7.01$, median: $7.38$, 75%: $8.10$, max: $9.96$)
  - **TDS**: Mean **$692.65\,\text{ppm}$** (std: $647.33$, min: $28.60$, 25%: $163.00$, median: $472.85$, 75%: $1023.32$, max: $2806.10$)
  - **Turbidity**: Mean **$11.31\,\text{NTU}$** (std: $9.87$, min: $0.00$, 25%: $1.92$, median: $9.97$, 75%: $18.54$, max: $72.75$)
  - **Temperature**: Mean **$24.88^\circ\text{C}$** (std: $2.69$, min: $16.56$, 25%: $23.09$, median: $25.04$, 75%: $26.93$, max: $31.51$)
- **Resolution**: Replaced all placeholder statistics in the Section 3 narrative and added programmatic assertions asserting that means match these recomputed values. Removed speculative causal attributions.

### Issue 3: False Negative Case #55037 & Leaf Node Discrepancy
- **Contradiction**: Text claimed Case #55037 terminates at Leaf Node #14, while the execution output reported Leaf Node #6.
- **Authoritative Evidence**: Execution of `extract_decision_path()` on `df.loc[55037]` ($pH=7.05, TDS=192.3, Turbidity=3.95, Temp=24.26$):
  - Is index 55037 in test set? **True** (`X_test.loc[55037]`).
  - True ground truth label: `Unsafe` ($1$).
  - Predicted label: `Safe` ($0$).
  - Estimated model class probabilities: $P(\text{Safe}) = 0.9948, P(\text{Unsafe}) = 0.0052$.
  - Exact traversal path: Node #0 ($TDS \le 202.75$) $\to$ Node #1 ($Turb \le 4.49$) $\to$ Node #2 ($Turb \le 4.42$) $\to$ Node #3 ($TDS \le 202.35$) $\to$ Node #4 ($Turb > 3.565$) $\to$ **Leaf Node #6**.
  - Conforming potable sample ($pH=7.25, TDS=185.0, Turb=0.45, Temp=22.0$) branches left at Node #4 ($Turb \le 3.565$) and terminates at **Leaf Node #5** ($P(\text{Safe}) = 1.0000$).
  - High TDS contaminated sample ($TDS=1850.0$) branches right at Node #0 ($TDS > 202.75$) and terminates at **Leaf Node #26** ($P(\text{Unsafe}) = 1.0000$).
- **Resolution**: Updated Section 7 and Section 14 to report Leaf Node #5 for conforming potable water, Leaf Node #6 for Case #55037, and Leaf Node #26 for high-TDS runoff. Used the precise terminology *"estimated model class probabilities from the trained Decision Tree"*.

### Issue 4: Drinking Water Standards & Statutory Requirements
- **Contradiction**: Conflated WHO and BIS standards; listed TDS permissible limit as $1000\,\text{mg/L}$ instead of $2000\,\text{mg/L}$; listed a non-existent 15–25°C "potable limit".
- **Authoritative Evidence**: Official **Bureau of Indian Standards BIS IS 10500:2012** (Table 1) and **WHO Guidelines for Drinking-water Quality (4th Edition)**:
  - **pH**: BIS Acceptable $6.5 - 8.5$; Permissible: No relaxation. WHO: Operational guideline $6.5 - 8.5$.
  - **TDS**: BIS Acceptable $500\,\text{mg/L}$; Permissible: $2000\,\text{mg/L}$ in absence of alternate source. WHO: Palatable $< 600\,\text{mg/L}$; unacceptable $> 1200\,\text{mg/L}$.
  - **Turbidity**: BIS Acceptable $1\,\text{NTU}$; Permissible: $5\,\text{NTU}$ in absence of alternate source. WHO: Desirable median $< 1\,\text{NTU}$ for disinfection; appearance acceptable $< 5\,\text{NTU}$.
  - **Temperature**: No statutory limit. WHO states cool water is aesthetically preferred ($< 25^\circ\text{C}$), but temperature is an environmental parameter without a health-based standard.
- **Resolution**: Restructured Section 2 Data Dictionary into a comparative table separating BIS IS 10500:2012 acceptable/permissible limits from WHO guidance. Removed the claimed 15–25°C statutory limit. Labeled min/max values as "observed dataset ranges". Explicitly noted that meeting physical thresholds does not establish freedom from microbiological pathogens or toxic heavy metals.

### Issue 5: Inconsistent ESP32 Pin Assignments
- **Contradiction**: Section 10 text cited GPIO 36 for pH, 39 for TDS, 34 for Turbidity, while firmware configuration used a different mapping.
- **Authoritative Evidence**: Source files [`firmware/cloudguard_esp32/config.h`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/config.h) and `config.h.example`:
  ```c
  #define PIN_PH_ADC          34       // ADC1_CH6 (Input-only, pH probe via Option D voltage divider)
  #define PIN_TDS_ADC         35       // ADC1_CH7 (Input-only, TDS probe direct 3.3V connection)
  #define PIN_TURB_ADC        32       // ADC1_CH4 (Turbidity sensor via Option D voltage divider)
  #define PIN_DS18B20         4        // GPIO 4 for DS18B20 1-Wire bus (requires 4.7k pull-up to 3.3V)
  ```
- **Resolution**: Updated Section 1 and Section 10 diagrams and tables to:
  - pH: **GPIO 34** (ADC1_CH6)
  - TDS: **GPIO 35** (ADC1_CH7)
  - Turbidity: **GPIO 32** (ADC1_CH4)
  - DS18B20: **GPIO 4** (1-Wire digital bus)
  Added an explicit notice that GPIO capabilities and ADC noise depend on specific board variants (e.g., GPIO 34/35 are input-only), and that physical wiring remains pending hardware verification.

### Issue 6: Unsupported ADC Calibration Claims
- **Contradiction**: A 3rd-order polynomial transfer curve was described as "calibrated" without supporting empirical fit data.
- **Authoritative Evidence**: Firmware `sensor_conversions.h` and `cloudguard_esp32.ino` rely natively on `analogReadMilliVolts()`, which applies the ESP32 chip's factory-burned eFuse calibration constants.
- **Resolution**: Clarified in Section 12 that the polynomial curve is an illustrative reference model for non-linearity characterization, not a substitute for factory eFuse calibration. Stated that multi-point physical bench calibration with precision voltage references remains **PENDING PHYSICAL COMMISSIONING**. Verified Option D voltage divider mathematics ($R_1 = 15\,\text{k}\Omega, R_2 = 10\,\text{k}\Omega$, ratio $0.4000$, multiplier $M=2.5000\times$, $5.0\,\text{V} \to 2.0\,\text{V}$, providing $450\,\text{mV}$ headroom below the $2.45\,\text{V}$ non-linear threshold).

### Issue 7: Identifier Detection Incorrectly Flagged Sensor Feature
- **Contradiction**: Section 6 code used `'id' in c.lower()`, which falsely reported `['Turbidity_NTU']` as an identifier because the word `turbidity` contains `id`.
- **Authoritative Evidence**: Python regular expression evaluation against the 5 canonical columns.
- **Resolution**: Replaced substring check with boundary-aware regex:
  ```python
  id_cols = [c for c in df.columns if re.search(r"\b(id|station|device|sensor_id|sample_id)\b", c, re.I)]
  ```
  Added an assertion verifying that `len(id_cols) == 0`.

### Issue 8: Model Comparison & Phase Summary Discrepancy
- **Contradiction**: Phase summary claimed Logistic Regression achieved 70.0% and Random Forest 100.0%, while the held-out test table showed all models at $\approx 99.99\%$.
- **Authoritative Evidence**: Saved artifact [`reports/model_comparison.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/model_comparison.csv):
  - Majority-Class Dummy Baseline: $70.00\%$ accuracy, Safe Recall $0.0000$, Macro F1 $0.4118$.
  - Logistic Regression (StandardScaler Pipeline): Accuracy $99.99\%$ ($0.999901$), Macro F1 $0.999882$, $TN=6048, FP=0, FN=2, TP=14110$.
  - Decision Tree (`max_depth=5`): Accuracy $99.99\%$ ($0.999901$), Macro F1 $0.999882$, $TN=6047, FP=1, FN=1, TP=14111$.
  - Random Forest ($100$ trees, `max_depth=8`): Accuracy $99.99\%$ ($0.999901$), Macro F1 $0.999882$, $TN=6047, FP=1, FN=1, TP=14111$.
- **Resolution**: Reconciled Section 5 and Section 15 by explicitly including the Majority-Class Dummy Baseline ($70.00\%$) and stating that all three trained models achieve $99.99\%$ test accuracy. Corrected Random Forest to unrounded $99.99\%$.

### Issue 9: Rigorous Cryptographic Hash Comparisons
- **Contradiction**: The previous hash audit only printed truncated 16-character prefixes without asserting equality against expected values.
- **Authoritative Evidence**: Canonical reference hash `90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a`.
- **Resolution**: Implemented two-stage hash verification:
  1. Section 2 asserts canonical dataset hash matches `CANONICAL_EXPECTED_HASH`, verifies backup byte-for-byte identity, and records a baseline dictionary of full 64-character hashes.
  2. Section 16 recomputes hashes for all protected files, prints both baseline and current full 64-character hashes, and asserts strict equality: `assert all_hashes_matched`.

### Issue 10: Machine-Specific Local File Paths
- **Contradiction**: Prose contained machine-specific absolute links like `file:///d:/E0224033/...`.
- **Resolution**: Removed all absolute local URIs. Used portable relative paths and Markdown code formatting (e.g., `api/main.py`, `dashboard/`, `models/decision_tree.joblib`).

### Issue 11: Scope of Dashboard Verification
- **Contradiction**: Text previously implied complete browser dashboard testing while executing FastAPI endpoints in Python.
- **Resolution**: Clearly delineated the testing tiers in Section 14:
  - In-process API integration tests (`TestClient`).
  - Static file asset availability tests.
  - Live browser automated tests (executed with genuine screenshots in `reports/figures/phase8_dashboard_*.png`).
  - Physical live hardware streaming explicitly marked as **PENDING PHYSICAL COMMISSIONING**.

### Issue 12: Project Provenance & Historical Claims
- **Contradiction**: Potential for unverified historical claims regarding geographic sampling locations.
- **Resolution**: Restricted provenance strictly to verified facts: Mendeley Data, DOI `10.17632/x6zncm47mt.1`, titled "Drinking Water Potability", containing 100,800 static, independent observations.

---

## 3. Protected Artifact Integrity Verification

Pre- and post-consolidation cryptographic SHA-256 hashes confirm zero modifications to protected datasets and authoritative trained models:

```
========================================================================================================================
PROTECTED ARTIFACT FILE                     BASELINE SHA-256 (64-CHAR HEX)                                      POST-EXECUTION SHA-256                                              STATUS
========================================================================================================================
data/water_quality_dataset1.csv             90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a   90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a   MATCH (UNCHANGED)
data/water_quality_dataset1_original.csv    90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a   90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a   MATCH (UNCHANGED)
models/decision_tree.joblib                 0951a58754ac752f1ec544d1202a9bae0f6f0e2b9b9129f5f71a2e60c5ddc3d7   0951a58754ac752f1ec544d1202a9bae0f6f0e2b9b9129f5f71a2e60c5ddc3d7   MATCH (UNCHANGED)
models/logistic_regression.joblib           06cf1403fc868faecf3772349296e791e2159c2f1af0aeb0f873004a7b62fe04   06cf1403fc868faecf3772349296e791e2159c2f1af0aeb0f873004a7b62fe04   MATCH (UNCHANGED)
models/random_forest.joblib                 b82bdbb7cf8f6d8e6bda1fef3ec9844237e7d35fd7adc6a87eeafd7714fe0713   b82bdbb7cf8f6d8e6bda1fef3ec9844237e7d35fd7adc6a87eeafd7714fe0713   MATCH (UNCHANGED)
========================================================================================================================
```

---

## 4. Execution Evidence & Quality Assurance

### 4.1 Clean-Kernel Execution Result
The corrected consolidated notebook was executed top-to-bottom using `src/create_consolidated_notebook.py` with `nbclient.NotebookClient`:
- **Execution Environment**: Python 3.10.0 (Windows)
- **Execution Time**: $\approx 14.2\,\text{seconds}$
- **Outcome**: **Exit Code 0**
- **Cell Errors**: **0 errors across all 16 sections**

### 4.2 In-Process Regression Test Suite
Section 15 of the notebook discovers and runs the project's automated test suite:
```
Ran 68 tests in 0.590s
OK (All 68 tests passed)
```

### 4.3 Static Code Analysis
Static type checking was executed with Pyright:
```
npx pyright
0 errors, 0 warnings, 0 informations
```

---

## 5. PDF and HTML Export Verification

### 5.1 Export Commands
1. **HTML Export**:
   ```powershell
   & python -m nbconvert --to html notebooks/CloudGuard_Water_Complete_Project.ipynb --output-dir reports/ --output CloudGuard_Water_Complete_Project.html
   ```
   *Outcome*: Generated `reports/CloudGuard_Water_Complete_Project.html` (753,268 bytes).
2. **PDF Export**:
   ```powershell
   Start-Process -FilePath "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" `
     -ArgumentList '--headless=new', '--disable-gpu', '--no-pdf-header-footer', `
     '--print-to-pdf="reports/CloudGuard_Water_Complete_Project.pdf"', `
     'file:///reports/CloudGuard_Water_Complete_Project.html' -Wait
   ```
   *Outcome*: Generated `reports/CloudGuard_Water_Complete_Project.pdf` (1,238,083 bytes).

### 5.2 PDF Visual Inspection Across All 33 Pages
Visual inspection via image rendering tools verified:
- **Total Pages**: Exactly **33 pages**.
- **Page 1–4**: Project Title, Scope, 3-tier System Architecture Diagram, Interactive Table of Contents.
- **Page 5–6**: Section 2 Data Dictionary separating BIS IS 10500:2012 from WHO guidance, observed dataset ranges, boundary rules.
- **Page 7–10**: Full SHA-256 hash verification output, Section 3 EDA with recomputed means (pH 7.59, TDS 692.65, Turb 11.31, Temp 24.88), distribution histograms, Pearson correlation heatmap.
- **Page 11–15**: Section 4 train/test split, model artifact verification (`max_depth=5`, 27 nodes, 14 leaves), Section 5 evaluation matrix including Dummy Baseline, Decision Tree confusion matrix.
- **Page 16–20**: Section 6 provenance audit with word-boundary regex (empty identifier list `[]`), Section 7 Decision Tree ladder (Conforming sample at Leaf #5, Case #55037 at Leaf #6, High TDS at Leaf #26), Section 8 FastAPI `/health`, `/predict`, `/explain` responses.
- **Page 21–25**: Section 9 1D training-range warnings and HTTP 422 domain rejection, Section 10 authoritative firmware pins (GPIO 34, 35, 32, 4), Section 11 power rails and DS18B20 fault handling, Section 12 ADC polynomial reference model, Section 13 Option D headroom table.
- **Page 26–28**: Option D mathematical verification (450mV margin), Section 14 web dashboard features and 6 preset scenarios evaluation table.
- **Page 29–31**: Section 15 unified Phase 1–8 results matrix, in-process unit test suite execution (68 tests passed in 0.590s), Section 16 pathogen/heavy metal limitations and non-certification notice.
- **Page 32–33**: Future work roadmap, post-execution SHA-256 hash comparison output displaying full 64-character hashes and `MATCH (UNCHANGED)` across all files.

---

## 6. Summary of Outstanding Limitations & Pending Hardware Work

1. **Physical Sensor Wet Testing**: Bench calibration of analog probes using standardized buffer solutions (pH 4.01, 7.00, 9.21) and standard conductivity solutions ($1413\,\mu\text{S/cm}$) remains **strictly pending physical hardware sessions**.
2. **Live Hardware Telemetry**: Live ESP32-to-API wireless ingestion will occur during scheduled physical laboratory sessions. The software pipeline is verified and standby-ready.
3. **Safety Non-Certification**: 4 physical parameters cannot detect microbiological pathogens (*E. coli*, noroviruses) or toxic heavy metals (lead, arsenic). Certified laboratory assays under statutory standards (WHO / BIS IS 10500:2012) remain strictly required to certify drinking water safety.
4. **Physical Actuation**: Automatic control of valves and pumps remains strictly disabled.

*The consolidated notebook correction, verification, and PDF/HTML exports are complete and fully verified.*
