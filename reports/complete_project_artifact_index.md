# CloudGuard-Water: Complete Project Artifact Index

**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning  
**Course**: Machine Learning for Cloud Services (MLCS)  
**Date**: October 2026  

---

## 1. Consolidated Project Notebook & Publication Exports

| Artifact | File Path | File Size | Description |
| :--- | :--- | :---: | :--- |
| **Consolidated Notebook** | [`notebooks/CloudGuard_Water_Complete_Project.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/CloudGuard_Water_Complete_Project.ipynb) | 360 KB | Comprehensive, de-duplicated, 16-section executable project notebook covering Phases 1–8. |
| **Exported PDF Monograph**| [`reports/CloudGuard_Water_Complete_Project.pdf`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/CloudGuard_Water_Complete_Project.pdf) | 1.24 MB | 33-page publication-grade PDF with tables, plots, code cells, and verified outputs. |
| **Exported HTML Report**  | [`reports/CloudGuard_Water_Complete_Project.html`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/CloudGuard_Water_Complete_Project.html) | 753 KB | Standalone, interactive HTML export with full syntax highlighting. |
| **Consolidation Report**  | [`reports/complete_project_notebook_report.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/complete_project_notebook_report.md) | 16 KB | Technical report documenting coverage matrix, deduplication strategy, and export verification. |

---

## 2. Protected Authoritative Original Phase Notebooks

All original phase notebooks are preserved unchanged:

| Phase | Original Notebook File | Size | Phase Focus |
| :---: | :--- | :---: | :--- |
| **Phase 1** | [`notebooks/01_dataset_inspection.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/01_dataset_inspection.ipynb) | 176 KB | Dataset schema, missing values, duplicates, and initial inspection. |
| **Phase 2** | [`notebooks/02_eda.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/02_eda.ipynb) | 1.91 MB | Exploratory data analysis, distributions, correlation analysis, class imbalance. |
| **Phase 3** | [`notebooks/03_baseline_models.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/03_baseline_models.ipynb) | 385 KB | Baseline training: Logistic Regression, Decision Tree, Random Forest. |
| **Phase 4** | [`notebooks/04_phase4_validation.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/04_phase4_validation.ipynb) | 109 KB | 5-fold cross-validation, confusion matrices, and model selection. |
| **Phase 4.1**| [`notebooks/04a_phase4_audit.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/04a_phase4_audit.ipynb) | 39 KB | Dataset provenance, Mendeley Data audit, and atemporal validation. |
| **Phase 5** | [`notebooks/05_phase5_explainability.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/05_phase5_explainability.ipynb) | 757 KB | Decision tree path extraction, SHAP analysis, Case #55037 error audit. |
| **Phase 6** | [`notebooks/06_phase6_cloud_api.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/06_phase6_cloud_api.ipynb) | 40 KB | FastAPI microservice implementation (`/health`, `/predict`, `/explain`). |
| **Phase 6.1**| [`notebooks/06a_phase6_verification.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/06a_phase6_verification.ipynb) | 50 KB | Probability terminology, class mapping, 1D training-range warnings. |
| **Phase 7** | [`notebooks/07_phase7_iot_integration.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07_phase7_iot_integration.ipynb) | 47 KB | IoT hardware architecture, sensor inventory, pinout definitions. |
| **Phase 7.1**| [`notebooks/07a_phase7_verification.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07a_phase7_verification.ipynb) | 364 KB | Power rail separation, common ground star bus, DS18B20 1-Wire pullup. |
| **Phase 7.2**| [`notebooks/07b_phase7_adc_validation.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07b_phase7_adc_validation.ipynb) | 382 KB | ESP32 ADC non-linearity, `analogReadMilliVolts()`, polynomial curve. |
| **Phase 7.3**| [`notebooks/07c_phase7_final_readiness.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07c_phase7_final_readiness.ipynb) | 45 KB | Option D ($15\text{k}:10\text{k}$, $M=2.50\times$, 450mV headroom), firmware build. |
| **Phase 8** | [`notebooks/08_phase8_dashboard_integration.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/08_phase8_dashboard_integration.ipynb) | 28 KB | Web dashboard, canvas charts, sensor cards, demo scenarios, banners. |

---

## 3. Authoritative Datasets & Trained Models

| Artifact | File Path | Cryptographic SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Canonical Dataset** | [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv) | `90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a` | **Verified Unchanged** |
| **Dataset Backup** | [`data/water_quality_dataset1_original.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1_original.csv) | `90f9c7cb7404fb69b655ee27293c91b82ee3a346712e06035f8259393835a04a` | **Verified Unchanged** |
| **Decision Tree Model**| [`models/decision_tree.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/decision_tree.joblib) | `0951a58754ac752f1ec544d1202a9bae0f6f0e2b9b9129f5f71a2e60c5ddc3d7` | **Verified Unchanged** |
| **Logistic Regression**| [`models/logistic_regression.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/logistic_regression.joblib) | `06cf1403fc868faecf3772349296e791e2159c2f1af0aeb0f873004a7b62fe04` | **Verified Unchanged** |
| **Random Forest Model**| [`models/random_forest.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/random_forest.joblib) | `b82bdbb7cf8f6d8e6bda1fef3ec9844237e7d35fd7adc6a87eeafd7714fe0713` | **Verified Unchanged** |

---

## 4. Software Source Code & Web Dashboard

| Component | File Path | Description |
| :--- | :--- | :--- |
| **REST Microservice** | [`api/main.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/api/main.py) | FastAPI service exposing `/health`, `/predict`, `/explain`, and mounting `/dashboard`. |
| **Explainability Engine** | [`src/explain.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/explain.py) | Decision Tree path extraction, learned rules, and plain-English narrative generator. |
| **Dashboard Layout** | [`dashboard/index.html`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/index.html) | Semantic, accessible HTML5 dashboard structure. |
| **Dashboard CSS** | [`dashboard/css/styles.css`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/css/styles.css) | Vanilla CSS custom properties design system with dark slate theme and glassmorphism. |
| **Dashboard Controller** | [`dashboard/js/app.js`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/js/app.js) | Main frontend orchestrator managing API polling, mode switching, and states. |
| **Sensor Cards UI** | [`dashboard/js/components/sensorCards.js`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/js/components/sensorCards.js) | Real-time sensor cards with animated gauges and WHO/BIS status badges. |
| **Canvas Plotter** | [`dashboard/js/components/charts.js`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/js/components/charts.js) | High-performance HTML5 Canvas 2D rolling time-series telemetry renderer. |
| **Tree Ladder UI** | [`dashboard/js/components/explanation.js`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/js/components/explanation.js) | Renders Decision Tree traversal ladder, feature pills, and narratives. |
| **Demo Stream Manager** | [`dashboard/js/components/demoStream.js`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/dashboard/js/components/demoStream.js) | 6 preset demo scenarios with Gaussian noise jitter generator. |
| **Consolidation Script** | [`src/create_consolidated_notebook.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/create_consolidated_notebook.py) | Automation script that builds and executes the complete project notebook. |

---

## 5. Firmware & Embedded Hardware Artifacts

| Component | File Path | Description |
| :--- | :--- | :--- |
| **ESP32 Firmware Sketch** | [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/cloudguard_esp32.ino) | Arduino-ESP32 Wi-Fi telemetry client, oversampling, and dispatch. |
| **Conversions Header** | [`firmware/cloudguard_esp32/sensor_conversions.h`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/sensor_conversions.h) | Option D ADC calibration equations and physical conversion mathematics. |
| **Circuit Diagram** | [`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png) | High-resolution schematic diagram of sensor circuit and ESP32 pinouts. |

---

## 6. Automated Test Suites (68 Tests Total)

| Test Module | File Path | Test Count | Description |
| :--- | :--- | :---: | :--- |
| **API Endpoints** | [`tests/test_api.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_api.py) | 20 | Microservice routes, domain boundaries, finite checks, HTTP 422/503. |
| **Explainability** | [`tests/test_explain.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_explain.py) | 17 | Decision path extraction, rule generation, Case #55037 error audit. |
| **IoT Integration** | [`tests/test_iot_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_iot_integration.py) | 18 | Sensor conversions, polynomial ADC calibration, noise filtering. |
| **Dashboard** | [`tests/test_dashboard_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_dashboard_integration.py) | 13 | Static delivery, mandatory banners, preset scenarios, error handling. |
| **Total Test Suite** | `python -m unittest discover tests` | **68** | **100% Passing** across all modules in $< 0.45\,\text{s}$. |

---

## 7. Key Technical Figures

| Figure | Path | Description |
| :--- | :--- | :--- |
| **IoT Wiring Diagram** | [`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png) | Electrical wiring diagram showing Option D voltage dividers and rails. |
| **Dashboard Initial** | [`reports/figures/phase8_dashboard_initial.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase8_dashboard_initial.png) | Live browser screenshot of running dashboard in potable demo stream mode. |
| **Dashboard Contaminated** | [`reports/figures/phase8_dashboard_final.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase8_dashboard_final.png) | Live browser screenshot showing contaminated high-TDS runoff classification. |
| **Decision Tree Rules** | [`reports/decision_tree_rules.txt`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/decision_tree_rules.txt) | Exported text representation of the 27-node Decision Tree (depth 5, 14 leaves). |
