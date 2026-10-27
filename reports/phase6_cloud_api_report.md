# Phase 6 Technical Report: Cloud Prediction Microservice & REST API

**Project Name**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services  
**Author**: Lead ML Deployment Engineer & Python Backend Specialist  
**Canonical Dataset**: [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv)  
**Primary Model**: [`models/decision_tree.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/decision_tree.joblib)  
**API Implementation**: [`api/main.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/api/main.py)  
**Executed Notebook**: [`notebooks/06_phase6_cloud_api.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/06_phase6_cloud_api.ipynb)  
**Date**: October 2026  

---

## Executive Summary

Phase 6 transitions the CloudGuard-Water system from an offline research evaluation pipeline into a modular, asynchronous **Cloud REST Microservice** built with **FastAPI**, **Pydantic v2**, and **Uvicorn**. The microservice exposes high-performance, non-blocking inference endpoints designed for downstream integration with physical IoT microcontrollers (ESP32), cloud data ingestion layers, and administrative monitoring dashboards.

### Core Deliverables & Technical Accomplishments:
1. **Asynchronous REST Microservice Architecture**:
   - Implemented [`api/main.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/api/main.py) featuring clean architectural separation between HTTP request routing, Pydantic data modeling, and domain machine learning logic (`src/explain.py`).
   - Configured lifecycle management via FastAPI `lifespan` context manager, loading the primary model once into application memory on startup with microsecond in-memory inference latency.
2. **Standardized REST Endpoints**:
   - `GET /health`: Readiness probe verifying microservice availability and in-memory model loading. Returns `HTTP 200 OK` when healthy and a controlled `HTTP 503 Service Unavailable` if model artifacts are missing or corrupt.
   - `POST /predict`: Real-time water potability classification (`Safe` vs. `Unsafe`) with estimated model class probabilities from the trained Decision Tree (`P(Safe)`, `P(Unsafe)`), per-feature training-range warnings, and defensive non-certification safety notices.
   - `POST /explain`: Decision Tree explainability endpoint returning exact traversed node indices, feature comparison rules, learned thresholds, terminal leaf distribution, and a structured plain-English operational narrative.
3. **Defensive Input Validation & Range Verification**:
   - **Physical Domain Boundaries**: Enforces strict aqueous boundaries via Pydantic (`0.0 <= pH <= 14.0`, `TDS_ppm >= 0.0`, `Turbidity_NTU >= 0.0`, `-10.0 <= Temperature_C <= 100.0`).
   - **Corrupt Input Rejection**: Rejects `NaN`, `+Inf`, `-Inf`, nulls, non-numeric strings, missing fields, and extra unmapped fields (`extra="forbid"`) with standard `HTTP 422 Unprocessable Entity`.
   - **Per-Feature Training-Range Warnings**: Inputs within physical bounds but outside the canonical dataset training distribution (`pH [6.09, 9.96]`, `TDS [28.6, 2806.1] ppm`, `Turbidity [0.0, 72.75] NTU`, `Temp [16.56, 31.51] °C`) are accepted (HTTP 200) but generate explicit per-feature warnings. These are 1D marginal bounds and not a complete multivariate anomaly or out-of-distribution detector. Being inside observed ranges does not certify safety.
4. **Defensive Operational Guardrails**:
   - Live inference never claims water is *"safe to drink"*, *"guaranteed potable"*, or *"compliant with official guidelines"*.
   - Explanations explicitly reinforce that physical sensors cannot detect microbiological pathogens (*E. coli*, noroviruses) or toxic heavy metals (lead, arsenic), mandating certified laboratory testing under WHO / BIS IS 10500 standards.
   - Live predictions forbid user-supplied ground-truth labels from altering predictions or overruling model classifications.
5. **Quality Assurance & Verification**:
   - Automated test suite in [`tests/test_api.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\tests\test_api.py) containing 20 unit and integration tests using FastAPI's `TestClient`.
   - Combined regression suite of **37 automated tests** (20 API tests + 17 Phase 5 explainability tests) passing cleanly with 0 failures and 0 errors.
   - Static analysis verified via Pyright across `api/`, `src/`, and `tests/` with **0 errors, 0 warnings, 0 informations**.
   - Containerization prepared with a slim, multi-stage [`Dockerfile`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\Dockerfile) and [`.dockerignore`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\.dockerignore).

---

## 1. Microservice Architecture & Design

### 1.1 Architectural Pattern
The CloudGuard-Water REST API follows a layered microservice design:

```
[ IoT Client (ESP32) / Dashboard / Web UI ]
                     │  HTTP POST (JSON)
                     ▼
        ┌─────────────────────────┐
        │       FastAPI ASGI      │
        │   api/main.py (Router)  │
        └────────────┬────────────┘
                     │  Pydantic Validation (Domain & Finite Checks)
                     ▼
        ┌─────────────────────────┐
        │   Data Quality Layer    │
        │ Range & Anomaly Checks  │
        └────────────┬────────────┘
                     │  Ordered Features: [pH, TDS_ppm, Turbidity_NTU, Temperature_C]
                     ▼
        ┌─────────────────────────┐
        │     ML Engine Layer     │
        │   src/explain.py        │
        │ Decision Tree (Depth 5) │
        └────────────┬────────────┘
                     │  Probabilities + Exact Decision Path
                     ▼
        ┌─────────────────────────┐
        │   Defensive Response    │
        │  Classification + Steps │
        │  + Safety Disclaimers   │
        └─────────────────────────┘
```

### 1.2 Separation of Concerns
- **`api/main.py`**: Handles network routing, Pydantic schema validation, CORS security, HTTP status codes, and exception mapping.
- **`src/explain.py`**: Reusable core ML module containing model loading, feature validation, decision path extraction, and human-readable narrative generation.
- **`models/decision_tree.joblib`**: Read-only primary model artifact serialized with joblib, loaded once into memory on startup.

---

## 2. API Endpoint Specification

### 2.1 Readiness Probe: `GET /health`
- **Purpose**: Verifies that the microservice is operational and confirms that the primary ML model artifact is loaded in memory.
- **Success Response (`HTTP 200 OK`)**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "DecisionTreeClassifier",
    "api_version": "1.0.0",
    "classes": ["Safe", "Unsafe"],
    "features": ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"],
    "detail": null
  }
  ```
- **Failure Response (`HTTP 503 Service Unavailable`)**:
  Returned when the model artifact cannot be found or loaded from disk.
  ```json
  {
    "status": "unhealthy",
    "model_loaded": false,
    "model_name": null,
    "api_version": "1.0.0",
    "classes": ["Safe", "Unsafe"],
    "features": ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"],
    "detail": "Default model 'models/decision_tree.joblib' not found in candidate paths."
  }
  ```

### 2.2 Live Inference: `POST /predict`
- **Request Headers**: `Content-Type: application/json`
- **Input Schema**:
  ```json
  {
    "pH": 7.38,
    "TDS_ppm": 138.7,
    "Turbidity_NTU": 0.25,
    "Temperature_C": 20.25
  }
  ```
- **Success Response (`HTTP 200 OK`)**:
  ```json
  {
    "model_classification": "Safe",
    "estimated_probabilities": {
      "Safe": 1.0,
      "Unsafe": 0.0
    },
    "model": "DecisionTreeClassifier(max_depth=5)",
    "warnings": [],
    "safety_notice": "This output is a statistical machine learning classification based on 4 physical sensor inputs (pH, TDS, turbidity, temperature), not a certified drinking-water potability declaration. Physical sensors cannot detect microbiological pathogens (e.g., E. coli, Salmonella, noroviruses), toxic heavy metals (e.g., arsenic, lead, mercury), or hazardous organic chemicals. Standardized laboratory testing under statutory drinking-water standards (such as WHO guidelines or BIS IS 10500) remains strictly necessary to establish actual potability. Do not automatically actuate physical equipment (valves, pumps) based solely on this prediction."
  }
  ```

### 2.3 Transparent Explainability: `POST /explain`
- **Request Headers**: `Content-Type: application/json`
- **Input Schema**: Identical 4-feature schema as `/predict`.
- **Success Response (`HTTP 200 OK`)**:
  ```json
  {
    "model_classification": "Safe",
    "estimated_probabilities": {
      "Safe": 1.0,
      "Unsafe": 0.0
    },
    "model": "DecisionTreeClassifier(max_depth=5)",
    "leaf_node_id": 5,
    "decision_path_steps": [
      {
        "node_id": 0,
        "feature": "TDS_ppm",
        "value": 138.7,
        "threshold": 202.75,
        "direction": "left",
        "rule": "TDS_ppm (138.70) <= 202.75"
      },
      {
        "node_id": 1,
        "feature": "Turbidity_NTU",
        "value": 0.25,
        "threshold": 4.49,
        "direction": "left",
        "rule": "Turbidity_NTU (0.25) <= 4.49"
      },
      {
        "node_id": 2,
        "feature": "Turbidity_NTU",
        "value": 0.25,
        "threshold": 4.42,
        "direction": "left",
        "rule": "Turbidity_NTU (0.25) <= 4.42"
      },
      {
        "node_id": 3,
        "feature": "TDS_ppm",
        "value": 138.7,
        "threshold": 202.35,
        "direction": "left",
        "rule": "TDS_ppm (138.70) <= 202.35"
      },
      {
        "node_id": 4,
        "feature": "Turbidity_NTU",
        "value": 0.25,
        "threshold": 3.56,
        "direction": "left",
        "rule": "Turbidity_NTU (0.25) <= 3.56"
      }
    ],
    "primary_influencing_features": ["TDS_ppm", "Turbidity_NTU"],
    "human_explanation": "Model Classification: SAFE (Statistical Prediction — Not a Potability Certification)\n\nEstimated Model Probabilities (Empirical Training Proportions — Not Guarantees):\n  - Estimated P(Safe)  : 100.00%\n  - Estimated P(Unsafe): 0.00%\n\nDecision Path Traversal:\n  Traversed 5 decision node(s) to Leaf Node #5:\n    1. TDS_ppm (138.70) <= 202.75 (Split at node #0)\n    2. Turbidity_NTU (0.25) <= 4.49 (Split at node #1)\n    3. Turbidity_NTU (0.25) <= 4.42 (Split at node #2)\n    4. TDS_ppm (138.70) <= 202.35 (Split at node #3)\n    5. Turbidity_NTU (0.25) <= 3.56 (Split at node #4)\n\nPrimary Influencing Features: TDS_ppm, Turbidity_NTU\n\nOperational Recommendation & Safety Notice:\n  ADVISORY: Measured physical sensor values fall within the model's learned classification envelope. However, this statistical classification does NOT certify potability or consumption safety, nor does it establish compliance with statutory drinking-water standards (such as WHO, US EPA, or BIS IS 10500). Physical sensors cannot detect microbial pathogens (e.g., E. coli, Salmonella), toxic heavy metals (arsenic, lead, mercury), or organic chemical pollutants. Comprehensive certified laboratory testing remains strictly necessary to establish actual drinking-water potability.",
    "warnings": [],
    "safety_notice": "..."
  }
  ```

---

## 3. Data Validation & Boundary Analysis

### 3.1 Two-Tier Validation Hierarchy
We distinguish between **invalid requests** (unphysical, corrupt, or malformed data) and **unusual requests** (physically valid measurements that fall outside the training dataset):

| Tier | Validation Type | Applied Criteria | HTTP Status | Response Action |
| :--- | :--- | :--- | :---: | :--- |
| **Tier 1** | **Physical Domain & Schema Constraints** | $0 \le pH \le 14$<br>$TDS \ge 0$<br>$\text{Turbidity} \ge 0$<br>$-10 \le \text{Temp} \le 100^\circ\text{C}$<br>Finite float only (`no NaN/Inf`)<br>`extra="forbid"` | **`422 Unprocessable Entity`** | Request is rejected immediately. No model inference is executed. |
| **Tier 2** | **Per-Feature Training-Range Warnings** | $pH \notin [6.09, 9.96]$<br>$TDS \notin [28.6, 2806.1]$<br>$\text{Turbidity} \notin [0.0, 72.75]$<br>$\text{Temp} \notin [16.56, 31.51]^\circ\text{C}$ | **`200 OK`** | Request is processed by the model, but descriptive warning strings are added to the `warnings` array. Note: These are 1D marginal bounds and not a multivariate anomaly detector; being inside ranges does not certify safety. |

### 3.2 Live Inference vs. Retrospective Auditing
In Phase 5, we audited Row 55037 where ground truth was known to be `Unsafe` despite physical readings falling inside the potable envelope ($pH=7.05, TDS=192.3, \text{Turbidity}=3.95, \text{Temp}=24.26$). In live operational inference:
1. Ground truth is strictly unavailable.
2. The model outputs its true empirical probability ($P(\text{Safe})=99.48\%, P(\text{Unsafe})=0.52\%$).
3. The API forbids user-supplied ground-truth labels from overriding model predictions.
4. The mandatory defensive safety notice ensures operators understand that a `Safe` prediction is never an absolute potability guarantee.

---

## 4. Automated Verification & Test Results

### 4.1 Test Suite Breakdown (`tests/test_api.py`)
The API test suite consists of 20 comprehensive automated tests:
1. `test_root_endpoint`: Verifies service metadata and Swagger links (`GET /`).
2. `test_health_healthy_status`: Verifies readiness probe returns 200 and healthy flag.
3. `test_health_unhealthy_when_model_missing`: Verifies readiness probe returns 503 when model is missing.
4. `test_predict_valid_safe_observation`: Verifies potable classification for Row 0 ($P(\text{Safe})=1.0$).
5. `test_predict_valid_unsafe_observation`: Verifies contaminated classification for Row 30240 ($P(\text{Unsafe})=1.0$).
6. `test_predict_borderline_sample_probabilities`: Verifies estimated model class probabilities from the trained Decision Tree for Row 55037 ($P(\text{Safe})=0.9948$).
7. `test_explain_valid_observation`: Verifies full path steps, leaf node ID, and non-certifying narrative.
8. `test_reject_missing_features`: Rejects payloads missing mandatory features (422).
9. `test_reject_nonnumeric_values`: Rejects string values in numeric fields (422).
10. `test_reject_null_values`: Rejects `null`/`None` values (422).
11. `test_reject_nan_and_infinity_strings`: Rejects `NaN`, `+Inf`, and `-Inf` (422).
12. `test_reject_invalid_physical_domains`: Rejects negative TDS, negative turbidity, and pH outside $[0, 14]$ (422).
13. `test_reject_extra_unexpected_fields`: Rejects unrecognized payload keys (`extra="forbid"`) (422).
14. `test_out_of_training_range_generates_warnings`: Populates 4 warnings for out-of-distribution values (200).
15. `test_in_range_values_produce_no_warnings`: Asserts in-range inputs yield zero warnings (200).
16. `test_individual_per_feature_training_range_warnings`: Verifies isolated single-feature deviations trigger exactly 1 per-feature warning (200).
17. `test_probability_terminology_and_disclaimer`: Verifies estimated model probability terminology and disclaims calibration/safety guarantees.
18. `test_in_range_values_do_not_certify_safety`: Verifies safety notices appear even on in-range inputs.
19. `test_predict_and_explain_return_503_when_model_unavailable`: Controlled 503 response on missing model.
20. `test_safety_wording_never_claims_safe_to_drink`: Asserts prohibited certifying phrases never appear.

### 4.2 Full Regression Execution Output
Executing all tests across `tests/`:

```text
Ran 37 tests in 0.773s

OK (20 API tests passed, 17 Phase 5 explainability tests passed, 0 failures, 0 errors)
```

- **Static Type Checking**: Pyright completed with **0 errors, 0 warnings, 0 informations**.

---

## 5. Local Execution Instructions

### 5.1 Starting the Microservice
From the repository root (`d:\E0224033\ML_for_Cl_Ser\CL\LB_Proj`), start the ASGI development server:

```powershell
& "C:\Users\Swaminathan\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 5.2 Interactive Documentation Endpoints
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Interactive Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI 3.1 JSON Schema**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### 5.3 Verification via `curl`

```powershell
# 1. Check Service Readiness
curl -X GET http://127.0.0.1:8000/health

# 2. Submit Live Water Quality Prediction
curl -X POST http://127.0.0.1:8000/predict `
     -H "Content-Type: application/json" `
     -d '{"pH": 7.35, "TDS_ppm": 185.0, "Turbidity_NTU": 2.10, "Temperature_C": 22.50}'

# 3. Request Decision Path Explanation
curl -X POST http://127.0.0.1:8000/explain `
     -H "Content-Type: application/json" `
     -d '{"pH": 7.35, "TDS_ppm": 185.0, "Turbidity_NTU": 2.10, "Temperature_C": 22.50}'
```

---

## 6. Operational & Security Considerations

1. **Local Scope vs. Cloud Production**:
   - The current server runs on local loopback (`127.0.0.1`).
   - For public cloud deployment, configure a production reverse proxy (Nginx or Cloudflare), enable HTTPS/TLS encryption, and enforce token-based API authentication (e.g. JWT or API keys).
2. **Defensive Actuation Prohibition**:
   - The microservice does **not** actuate physical valves, pumps, or chemical dosing equipment. Machine learning predictions are advisory and must not automate physical interventions without human confirmation and laboratory testing.
3. **Containerization**:
   - Provided [`Dockerfile`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/Dockerfile) packages the API into an isolated `python:3.10-slim` container exposing port 8000 with a native `/health` probe.

---

## 7. Phase 6 Completion & Phase 7 Roadmap

Phase 6 has achieved all stated objectives. The API is tested, documented, container-ready, and operational.

### Recommended Next Steps (Phase 7: IoT Circuit & Sensor Ingestion):
1. **ESP32 Microcontroller Setup**: Wire analog pH sensor, analog TDS probe, turbidity sensor, and DS18B20 digital temperature probe to the ESP32 ADC pins.
2. **Firmware Development**: Implement an Arduino/ESP-IDF firmware client that samples physical sensor voltages, converts them to physical units, serializes a JSON payload, and posts to `/predict` via Wi-Fi.
3. **Local Monitoring Dashboard**: Develop a real-time web or OLED dashboard to visualize sensor streams, classifications, probabilities, and safety alerts.
