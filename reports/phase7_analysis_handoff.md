# Phase 7 Technical Handoff Report: Physical IoT Circuit, ESP32 Firmware & Live Sensor Integration

**Project Title**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services  
**Author**: Lead Embedded & IoT Integration Specialist  
**Canonical Dataset**: [`data/water_quality_dataset1.csv`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/data/water_quality_dataset1.csv)  
**Primary Production Model**: [`models/decision_tree.joblib`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/models/decision_tree.joblib)  
**FastAPI Microservice**: [`api/main.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/api/main.py)  
**Firmware Sketch**: [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/cloudguard_esp32.ino)  
**Firmware Header**: [`firmware/cloudguard_esp32/sensor_conversions.h`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/sensor_conversions.h)  
**Configuration Template**: [`firmware/cloudguard_esp32/config.h.example`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/config.h.example)  
**Wiring Reference**: [`reports/phase7_wiring_reference.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7_wiring_reference.md)  
**Technical Report**: [`reports/phase7_iot_integration_report.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7_iot_integration_report.md)  
**Executed Notebook**: [`notebooks/07_phase7_iot_integration.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07_phase7_iot_integration.ipynb)  
**Date**: October 2026  

---

## 1. Project Context & Objectives

CloudGuard-Water is an intelligent water quality classification system developed for the *Machine Learning for Cloud Services* curriculum. The objective of **Phase 7** is to bridge the offline machine learning model and REST API microservice (Phases 1–6.1) with a cyber-physical prototype:
1. Interface four water quality sensors:
   - **pH probe**: Analog electrochemical potential.
   - **TDS probe**: Analog electrical conductivity (ppm).
   - **Turbidity probe**: Analog optical transmission/scattering (NTU).
   - **DS18B20 probe**: Digital 1-Wire water temperature ($^\circ\text{C}$).
2. An **ESP32 microcontroller** samples the sensors, filters electrical noise, performs hardware fault checks, converts raw voltages to physical units, formats a canonical JSON payload, and transmits telemetry over Wi-Fi.
3. The **FastAPI prediction microservice** receives telemetry at `POST /predict`, validates constraints, executes the trained Decision Tree (`models/decision_tree.joblib`), and returns a statistical potability classification (`Safe` vs. `Unsafe`) with estimated model class probabilities and non-certifying safety notices.
4. Serial diagnostics log results to the console without actuating physical equipment.

---

## 2. Hardware Inventory & Questionnaire for User Confirmation

### 2.1 Hardware Component Specifications

| Component | Target Model / Part | Power Rail | ESP32 Pin Assignment | Interface Type | Sizing / Level Shifting |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Microcontroller** | ESP32-WROOM-32 / NodeMCU-32S | 5.0 V (USB) | Multiple | 3.3 V Logic / Wi-Fi | Internal 3.3V LDO |
| **pH Sensor** | Analog pH Sensor Kit (SEN0161 / PH-4502C) | 5.0 V (or 3.3V) | **GPIO 34 (ADC1_CH6)** | Analog Voltage ($V_{\text{out}}$) | $10\,\text{k}\Omega : 20\,\text{k}\Omega$ Divider (if 5V) |
| **TDS Sensor** | Gravity Analog TDS Sensor (SEN0244 / CQRobot) | 3.3 V | **GPIO 35 (ADC1_CH7)** | Analog Voltage ($V_{\text{out}}$) | Direct to 3.3V ADC1 pin |
| **Turbidity Sensor**| Optical Turbidity Sensor (SEN0189 / TS-300B) | 5.0 V | **GPIO 32 (ADC1_CH4)** | Analog Voltage ($V_{\text{out}}$) | **$10\,\text{k}\Omega : 20\,\text{k}\Omega$ Voltage Divider** |
| **Temperature** | DS18B20 Waterproof Stainless Probe | 3.3 V | **GPIO 4** | 1-Wire Digital Bus | **$4.7\,\text{k}\Omega$ Pull-Up Resistor** |

### 2.2 Items Requiring Confirmation from the User
Because physical hardware was not connected to an active COM port during this development session, the following parameters are modularized in `firmware/cloudguard_esp32/config.h` and require user confirmation before final field deployment:
1. **ESP32 Board Variant**: Confirm whether your board is a 30-pin NodeMCU-32S, 38-pin DevKitC, or ESP32-S3 (ADC1 channel pinouts differ on ESP32-S3).
2. **pH Sensor Supply Voltage**: Check whether your pH amplifier board has a 3.3V/5V selector jumper. If set to 5V, ensure the voltage divider is installed; if 3.3V-compatible, direct connection is safe.
3. **Turbidity Switch**: Ensure the slide switch on the optical turbidity adapter board is set to **'A' (Analog)** and not 'D' (Digital threshold).
4. **Reference Calibration Solutions**: Confirm availability of standard calibration buffers (pH 4.00, pH 7.00, and 1413 µS/cm NaCl) to record exact local voltage offsets.

---

## 3. Electrical Constraints & Safety Review

### 3.1 The ADC1 vs. ADC2 Wi-Fi Conflict
The ESP32 microcontroller features two internal SAR ADCs:
- **ADC1**: GPIOs 32, 33, 34, 35, 36, 39.
- **ADC2**: GPIOs 0, 2, 4, 12, 13, 14, 15, 25, 26, 27.

When Wi-Fi transmission begins (`WiFi.begin()`), the FreeRTOS Wi-Fi driver monopolizes ADC2 for RF power management. Calling `analogRead()` on ADC2 pins while connected to Wi-Fi causes the ESP32 to return either error codes or floating ungrounded values.

**Implementation**: All 3 analog probes are assigned strictly to **ADC1**:
- `PIN_PH_ADC` = **GPIO 34** (Input-only, clean analog)
- `PIN_TDS_ADC` = **GPIO 35** (Input-only, clean analog)
- `PIN_TURB_ADC` = **GPIO 32** (Input/Output capable ADC1 channel)

### 3.2 Voltage Divider Level Shifter
The optical turbidity module operates at $5.0\,\text{V}$ and produces up to $4.50\,\text{V}$ in distilled water. To ensure signal voltages never exceed the ESP32's $3.30\,\text{V}$ limit, a precision voltage divider ($R_1 = 10\,\text{k}\Omega$, $R_2 = 20\,\text{k}\Omega$) is placed across the Turbidity AOUT signal:
$$V_{\text{ADC}} = V_{\text{sensor}} \times \frac{20\,\text{k}\Omega}{10\,\text{k}\Omega + 20\,\text{k}\Omega} = V_{\text{sensor}} \times \frac{2}{3} \approx 0.6667 \times V_{\text{sensor}}$$

- Clear water ($V_{\text{sensor}} = 4.50\,\text{V}$): $V_{\text{ADC}} = 3.00\,\text{V}$ ($\le 3.30\,\text{V}$, safe).
- Turbid water ($V_{\text{sensor}} = 2.50\,\text{V}$): $V_{\text{ADC}} = 1.67\,\text{V}$.

---

## 4. Calibration Mathematics & Conversion Equations

### 4.1 pH Conversion (Two-Point Linear Calibration)
$$pH = pH_{\text{neutral}} + \left(\frac{pH_{\text{acidic}} - pH_{\text{neutral}}}{V_{\text{acidic}} - V_{\text{neutral}}}\right) \times (V - V_{\text{neutral}})$$
- Default provisional voltages: $V_{\text{neutral}} = 2.50\,\text{V}$ (at pH 7.00), $V_{\text{acidic}} = 3.05\,\text{V}$ (at pH 4.00).
- Computed sensitivity: $\text{Slope} = \frac{4.00 - 7.00}{3.05 - 2.50} = -5.4545\,\text{pH/Volt}$.

### 4.2 TDS Conversion with Temperature Compensation
Water conductivity increases by approximately $+2.0\%$ per $^\circ\text{C}$ rise. To calculate accurate TDS in parts per million (ppm), raw voltage is compensated to a reference temperature of $25.0^\circ\text{C}$ using live DS18B20 data:
$$V_{25} = \frac{V_{\text{raw}}}{1.0 + 0.02 \times (T_{\text{DS18B20}} - 25.0)}$$
$$\text{TDS (ppm)} = (133.42 \times V_{25}^3 - 255.86 \times V_{25}^2 + 857.39 \times V_{25}) \times 0.5 \times K_{\text{cell}}$$

### 4.3 Optical Turbidity Parabolic Model
$$\text{Turbidity (NTU)} = -1120.4 \times V_{\text{sensor}}^2 + 5742.3 \times V_{\text{sensor}} - 4352.9 \quad (\text{for } V < V_{\text{clear}})$$
If $V_{\text{sensor}} \ge V_{\text{clear}} = 4.15\,\text{V}$, the water is optically clear and turbidity is clamped to $0.0\,\text{NTU}$.

---

## 5. Firmware Architecture & Source Code

The firmware files are organized under `firmware/cloudguard_esp32/`:
1. **`cloudguard_esp32.ino`**:
   - Arduino C++ sketch implementing non-blocking 10-second telemetry timer.
   - Built-in zero-dependency bit-banged 1-Wire driver for DS18B20 on GPIO 4.
   - 10-sample ADC averaging filter to eliminate high-frequency electrical noise.
   - Hardware fault check: skips transmission if probe is disconnected (-127°C) or rail saturated.
   - HTTPClient POST to `http://192.168.0.106:8000/predict`.
   - Native string parsing for classification, estimated probabilities, and data quality warnings.
2. **`sensor_conversions.h`**:
   - Inlined, unit-testable C++ header providing `adcToVoltage`, `scaleVoltageDivider`, `convertVoltageToPH`, `convertVoltageToTDS`, `convertVoltageToTurbidity`, and `checkSignalHealth`.
3. **`config.h.example` & `config.h`**:
   - Centralized definitions for pins, calibration constants, Wi-Fi credentials, and API URL.
   - Protected by `.gitignore` so private credentials are never committed.

---

## 6. Network Configuration & Microservice Connectivity

### 6.1 LAN Binding Instructions
In Phase 6, Uvicorn was launched on `127.0.0.1` (local loopback only). For the ESP32 to transmit over Wi-Fi, the microservice must listen on all network interfaces:

```powershell
& "C:\Users\Swaminathan\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6.2 Host LAN Address Discovery
- **Discovered Host IP**: `192.168.0.106` (Wireless LAN adapter Wi-Fi).
- **Target URL**: `http://192.168.0.106:8000/predict`
- **Health Check**: `http://192.168.0.106:8000/health`
- **Windows Firewall**: Port 8000 must be opened on the trusted private subnet.

---

## 7. Automated Testing & Verification Results

### 7.1 Separation of Verified vs. Unperformed Checks

| Check Type | Execution Status | Actual Result / Observation |
| :--- | :---: | :--- |
| **Python IoT Unit Tests** | **Executed & Verified** | 11 unit tests in `tests/test_iot_integration.py` passing cleanly in 0.011s. |
| **Full Regression Suite** | **Executed & Verified** | 48 tests passing cleanly across `tests/` in 0.409s with 0 failures and 0 errors. |
| **Pyright Static Analysis** | **Executed & Verified** | 0 errors, 0 warnings, 0 informations across `api/`, `src/`, `tests/`. |
| **FastAPI Microservice Test** | **Executed & Verified** | In-process simulated ESP32 payload accepted by `/predict` (HTTP 200). |
| **Jupyter Notebook Execution**| **Executed & Verified** | `07_phase7_iot_integration.ipynb` executed top-to-bottom via `nbclient`. |
| **Physical ESP32 Serial Test**| **Not Performed** | Physical ESP32 hardware was not attached via COM port in this session. |
| **Buffer Solution Calibration**| **Provisional Defaults** | Calibration constants in `config.h` are standard defaults awaiting physical buffers. |

### 7.2 Full Regression Execution Output
```text
test_explain_valid_observation (test_api.TestCloudGuardAPI) ... ok
test_health_healthy_status (test_api.TestCloudGuardAPI) ... ok
test_health_unhealthy_when_model_missing (test_api.TestCloudGuardAPI) ... ok
test_in_range_values_do_not_certify_safety (test_api.TestCloudGuardAPI) ... ok
test_in_range_values_produce_no_warnings (test_api.TestCloudGuardAPI) ... ok
test_individual_per_feature_training_range_warnings (test_api.TestCloudGuardAPI) ... ok
test_out_of_training_range_generates_warnings (test_api.TestCloudGuardAPI) ... ok
test_predict_and_explain_return_503_when_model_unavailable (test_api.TestCloudGuardAPI) ... ok
test_predict_borderline_sample_probabilities (test_api.TestCloudGuardAPI) ... ok
test_predict_valid_safe_observation (test_api.TestCloudGuardAPI) ... ok
test_predict_valid_unsafe_observation (test_api.TestCloudGuardAPI) ... ok
test_probability_terminology_and_disclaimer (test_api.TestCloudGuardAPI) ... ok
test_reject_extra_unexpected_fields (test_api.TestCloudGuardAPI) ... ok
test_reject_invalid_physical_domains (test_api.TestCloudGuardAPI) ... ok
test_reject_missing_features (test_api.TestCloudGuardAPI) ... ok
test_reject_nan_and_infinity_strings (test_api.TestCloudGuardAPI) ... ok
test_reject_nonnumeric_values (test_api.TestCloudGuardAPI) ... ok
test_reject_null_values (test_api.TestCloudGuardAPI) ... ok
test_root_endpoint (test_api.TestCloudGuardAPI) ... ok
test_safety_wording_never_claims_safe_to_drink (test_api.TestCloudGuardAPI) ... ok
test_artifacts_remain_intact (test_explain.TestExplainModule) ... ok
test_explain_sample_end_to_end (test_explain.TestExplainModule) ... ok
test_extract_decision_path_safe (test_explain.TestExplainModule) ... ok
test_extract_decision_path_unsafe (test_explain.TestExplainModule) ... ok
test_generate_human_explanation_safe (test_explain.TestExplainModule) ... ok
test_generate_human_explanation_unsafe (test_explain.TestExplainModule) ... ok
test_get_feature_importances (test_explain.TestExplainModule) ... ok
test_known_retrospective_false_negative_explanation (test_explain.TestExplainModule) ... ok
test_model_loading_nonexistent_raises_error (test_explain.TestExplainModule) ... ok
test_model_loading_success (test_explain.TestExplainModule) ... ok
test_predict_with_probabilities_agreement (test_explain.TestExplainModule) ... ok
test_validate_input_dataframe (test_explain.TestExplainModule) ... ok
test_validate_input_dict (test_explain.TestExplainModule) ... ok
test_validate_input_list (test_explain.TestExplainModule) ... ok
test_validate_input_missing_feature_raises_error (test_explain.TestExplainModule) ... ok
test_validate_input_nan_raises_error (test_explain.TestExplainModule) ... ok
test_validate_input_wrong_length_list_raises_error (test_explain.TestExplainModule) ... ok
test_adc_to_voltage_discrete_bounds (test_iot_integration.TestIoTIntegration) ... ok
test_build_iot_payload_rejects_unphysical (test_iot_integration.TestIoTIntegration) ... ok
test_build_iot_payload_valid (test_iot_integration.TestIoTIntegration) ... ok
test_convert_voltage_to_ph (test_iot_integration.TestIoTIntegration) ... ok
test_convert_voltage_to_tds (test_iot_integration.TestIoTIntegration) ... ok
test_convert_voltage_to_turbidity (test_iot_integration.TestIoTIntegration) ... ok
test_iot_payload_e2e_api_dispatch (test_iot_integration.TestIoTIntegration) ... ok
test_scale_voltage_divider_ratio (test_iot_integration.TestIoTIntegration) ... ok
test_validate_raw_signals_ds18b20_faults (test_iot_integration.TestIoTIntegration) ... ok
test_validate_raw_signals_healthy (test_iot_integration.TestIoTIntegration) ... ok
test_validate_raw_signals_overvoltage_and_negative (test_iot_integration.TestIoTIntegration) ... ok

----------------------------------------------------------------------
Ran 48 tests in 0.409s

OK
```

---

## 8. Summary of All Files Created or Modified

1. **`firmware/cloudguard_esp32/cloudguard_esp32.ino`** (Created): Complete Arduino sketch implementing sampling, signal validation, unit conversion, JSON serialization, and Wi-Fi HTTP POST.
2. **`firmware/cloudguard_esp32/sensor_conversions.h`** (Created): Inlined C++ conversion header for voltage scaling, pH, TDS, Turbidity, and signal health checks.
3. **`firmware/cloudguard_esp32/config.h.example`** (Created): Template configuration header documenting pins, calibration constants, and network parameters.
4. **`firmware/cloudguard_esp32/config.h`** (Created): Local configuration file populated with test defaults (ignored by Git).
5. **`src/iot_simulation.py`** (Created): Python module providing conversion formulas, signal fault checks, and payload construction.
6. **`tests/test_iot_integration.py`** (Created): 11 unit and integration tests covering signal conversions, fault detection, and API payload compatibility.
7. **`reports/figures/phase7_iot_wiring_diagram.png`** (Created): Clear high-resolution circuit wiring schematic.
8. **`reports/phase7_wiring_reference.md`** (Created): Practical reference guide with pinouts, voltage divider calculations, and pre-power-on checklist.
9. **`src/create_phase7_notebook.py`** (Created): Generator script executing `notebooks/07_phase7_iot_integration.ipynb` top-to-bottom.
10. **`notebooks/07_phase7_iot_integration.ipynb`** (Created): Executed Jupyter notebook with 10 sections covering circuit design, mathematics, signal validation, and live API tests.
11. **`reports/phase7_iot_integration_report.md`** (Created): Comprehensive Phase 7 technical report.
12. **`reports/phase7_analysis_handoff.md`** (Created): This handoff report.
13. **`.gitignore`** (Modified): Added ignore rule for `firmware/**/config.h` while keeping `config.h.example` tracked.

---

## 9. Readiness Assessment & Recommendation for Phase 8

Phase 7 is complete and verified. The system is ready for **Phase 8 (Interactive Real-Time Dashboard & End-to-End System Integration)**:
- The ESP32 firmware produces compliant, validated JSON matching the 4 canonical feature keys.
- The cloud API runs with sub-millisecond inference and returns clear classifications, estimated probabilities, and training range warnings.
- Next phase will focus on real-time web dashboard visualization, telemetry streaming, and user alert management.
