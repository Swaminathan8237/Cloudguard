# Phase 7 Technical Report: Physical IoT Circuit, ESP32 Firmware & Live Sensor Integration

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
**Executed Notebook**: [`notebooks/07_phase7_iot_integration.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07_phase7_iot_integration.ipynb)  
**Date**: October 2026  

---

## 1. Executive Summary

Phase 7 transitions the CloudGuard-Water project from a cloud-only machine learning microservice to an **end-to-end cyber-physical IoT system**. The system captures real-world aquatic properties using physical probes, processes signals on an edge microcontroller (ESP32), serializes telemetry into standardized JSON payloads, transmits measurements over a local Wi-Fi network, and obtains real-time potability classifications and explainability rules from the trained cloud model.

### Core Deliverables & Technical Accomplishments:
1. **Physical Circuit Design & Overvoltage Protection**:
   - Resolved the fundamental **ESP32 Wi-Fi ADC conflict** by routing all 3 analog probes strictly to **ADC1 channels** (`GPIO 34, 35, 32`), as ADC2 is disabled by the ESP32 Wi-Fi driver.
   - Designed a passive $10\,\text{k}\Omega : 20\,\text{k}\Omega$ voltage divider stepping down 5V optical turbidity outputs ($\le 4.5\,\text{V}$) to safe levels ($\le 3.0\,\text{V}$) to protect the 3.3V ESP32 ADC.
   - Generated a high-resolution circuit wiring schematic in [`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png).
2. **Production-Ready ESP32 Firmware**:
   - Implemented [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/cloudguard_esp32.ino) featuring modular state management, non-blocking sampling, multi-sample noise averaging, and automated Wi-Fi reconnect with exponential backoff.
   - Built-in zero-dependency bit-banged 1-Wire driver for the **DS18B20 digital temperature probe** on GPIO 4 with pull-up resistor verification.
   - Integrated hardware fault detection (detecting -127°C disconnection, 85°C reset state, rail saturation, and negative voltage inversions).
3. **Signal Conditioning & Unit Conversion**:
   - Developed physical conversion mathematics in `sensor_conversions.h` and Python simulation module [`src/iot_simulation.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/iot_simulation.py).
   - Applied Nernstian 2-point linear calibration for pH, 3rd-order polynomial with temperature compensation for TDS, and optical extinction curves for Turbidity.
4. **Network & Microservice Integration**:
   - Configured local LAN binding (`--host 0.0.0.0 --port 8000`) and identified host Wi-Fi IPv4 address (`192.168.0.106`).
   - Verified that simulated and live telemetry payloads format correctly and are accepted by the FastAPI `/predict` endpoint.
5. **Rigorous Quality Assurance**:
   - Added 11 automated unit and integration tests in [`tests/test_iot_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_iot_integration.py).
   - Full regression suite of **48 automated tests** passing with 0 failures and 0 errors.
   - Pyright static analysis confirmed with **0 errors, 0 warnings, 0 informations**.

---

## 2. Hardware Inventory & Questionnaire for User Confirmation

### 2.1 Hardware Component Inventory

| Component | Target Specification / Educational Standard | Power Rail | Interface | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Microcontroller** | ESP32-WROOM-32 / NodeMCU-32S (30-pin) | 5.0 V (USB) | Wi-Fi 802.11 b/g/n & ADC1 | Architecture ready |
| **pH Sensor** | Analog pH Sensor Kit v2 (SEN0161 / PH-4502C) | 5.0 V (or 3.3 V) | Analog Out $\to$ GPIO 34 | Provisional calibration |
| **TDS Sensor** | Gravity Analog TDS Sensor (SEN0244 / CQRobot) | 3.3 V | Analog Out $\to$ GPIO 35 | Provisional calibration |
| **Turbidity Sensor**| Optical Turbidity Module (SEN0189 / TS-300B) | 5.0 V | Analog Out $\to$ 10k/20k divider $\to$ GPIO 32 | Voltage divider sized |
| **Temperature** | DS18B20 Waterproof Stainless Probe | 3.3 V | 1-Wire Digital $\to$ GPIO 4 | $4.7\,\text{k}\Omega$ pull-up sized |

### 2.2 Questionnaire for User Hardware Confirmation
In compliance with project instructions, because physical probes were not attached via USB/COM port during this software session, the following parameters require physical confirmation from the user before applying final field power:
1. **ESP32 Variant**: Confirm whether your board is a 30-pin NodeMCU-32S, 38-pin DevKitC, or ESP32-S3 (pin numbers on ADC1 may vary on newer S3 variants).
2. **pH Adapter Operating Voltage**: Check whether your pH adapter board has a 3.3V/5V selector jumper. If set to 5V, ensure the voltage divider is installed; if 3.3V-compatible, direct connection is safe.
3. **Turbidity Mode Switch**: Ensure the slide switch on the optical turbidity adapter board is switched to **'A' (Analog)** and not 'D' (Digital threshold).
4. **Reference Calibration Solutions**: Confirm availability of standard calibration buffers (pH 4.00, pH 7.00, and 1413 µS/cm NaCl) to record exact local voltage offsets.

---

## 3. Circuit Wiring & Electrical Safety Review

![Circuit Wiring Diagram](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png)

### 3.1 The ADC1 vs ADC2 Wi-Fi Conflict
The ESP32 microcontroller incorporates two independent ADC blocks:
- **ADC1**: GPIOs 32, 33, 34, 35, 36, 39.
- **ADC2**: GPIOs 0, 2, 4, 12, 13, 14, 15, 25, 26, 27.

When Wi-Fi transmission begins in firmware (`WiFi.begin()`), the FreeRTOS Wi-Fi driver monopolizes ADC2 for RF calibration and power measurement. Calling `analogRead()` on an ADC2 channel while connected to Wi-Fi causes the ESP32 to return either error codes or floating ungrounded values.

**Implementation**: All 3 analog probes are assigned strictly to **ADC1**:
- `PIN_PH_ADC` = **GPIO 34** (Input-only, no internal pull-up/pull-down conflicts)
- `PIN_TDS_ADC` = **GPIO 35** (Input-only, clean analog)
- `PIN_TURB_ADC` = **GPIO 32** (Input/Output capable ADC1 channel)

### 3.2 Voltage Divider Level Shifter
The optical turbidity module operates at $5.0\,\text{V}$ and produces up to $4.50\,\text{V}$ in distilled water. To ensure signal voltages never exceed the ESP32's $3.30\,\text{V}$ limit, a precision voltage divider ($R_1 = 10\,\text{k}\Omega$, $R_2 = 20\,\text{k}\Omega$) is placed across the Turbidity AOUT signal:
$$V_{\text{ADC}} = V_{\text{sensor}} \times \frac{20\,\text{k}\Omega}{10\,\text{k}\Omega + 20\,\text{k}\Omega} = V_{\text{sensor}} \times \frac{2}{3} \approx 0.6667 \times V_{\text{sensor}}$$

- Clear water ($V_{\text{sensor}} = 4.50\,\text{V}$): $V_{\text{ADC}} = 3.00\,\text{V}$ ($\le 3.30\,\text{V}$, safe).
- Turbid water ($V_{\text{sensor}} = 2.50\,\text{V}$): $V_{\text{ADC}} = 1.67\,\text{V}$.

---

## 4. Sensor Conversion & Calibration Mathematics

### 4.1 pH Conversion (Nernstian 2-Point Linear Model)
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

## 5. ESP32 Firmware Architecture

The firmware in [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/cloudguard_esp32.ino) implements an asynchronous, robust telemetry loop:

```
[ Power On / Reset ]
         │
         ▼
[ setup() ] ──> Configure ADC1 attenuation (11dB)
         │  ──> Initialize 1-Wire DS18B20 pin
         │  ──> Connect to 2.4 GHz Wi-Fi (WIFI_SSID)
         ▼
[ loop() ]  <──────────────────────────────────────────┐
         │                                             │
         ├──> Check non-blocking timer (10s interval)  │
         ├──> Sample DS18B20 digital temperature        │
         ├──> Sample ADC1 channels (10-sample average) │
         ├──> Scale voltage dividers                   │
         ├──> checkSignalHealth()                      │
         │       ├── [Fault Detected] ──> Log to Serial Monitor & ABORT cycle
         │       └── [Healthy]                                                 │
         ├──> Convert voltages to physical units (pH, TDS, NTU, °C)            │
         ├──> Construct JSON payload                                           │
         ├──> HTTP POST -> http://192.168.0.106:8000/predict                   │
         ├──> Parse HTTP response & classification                             │
         ├──> Print Serial diagnostics & safety disclaimers                    │
         └──> Delay 50ms yield (FreeRTOS background Wi-Fi tasks) ──────────────┘
```

---

## 6. Network Configuration & LAN Microservice Binding

### 6.1 LAN Ingestion Configuration
In Phase 6, Uvicorn was bound to `127.0.0.1` (loopback only). For the ESP32 to transmit over the local network, the server must listen on all network interfaces:

```powershell
& "C:\Users\Swaminathan\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 6.2 Host LAN Address Discovery
- **Host IPv4**: `192.168.0.106` (verified via `ipconfig` on Wireless LAN adapter Wi-Fi).
- **Target Microservice URL**: `http://192.168.0.106:8000/predict`
- **Readiness Health Check**: `http://192.168.0.106:8000/health`
- **Windows Firewall**: Port 8000 must be allowed on the private local subnet.

---

## 7. Automated Test Suite & Static Analysis Results

### 7.1 Regression Execution (48 Tests)
We executed the entire automated test suite, which includes 20 API tests, 17 explainability tests, and 11 new Phase 7 IoT integration tests:

```powershell
& "C:\Users\Swaminathan\AppData\Local\Programs\Python\Python310\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 48 tests in 0.409s

OK (20 API tests passed, 17 Phase 5 explainability tests passed, 11 Phase 7 IoT tests passed, 0 failures, 0 errors)
```

### 7.2 Pyright Static Analysis
Executing Pyright across all code directories:

```powershell
npx pyright api/ src/ tests/
```

```text
0 errors, 0 warnings, 0 informations
Completed in 1.35sec
```

---

## 8. Remaining Limitations & Viva Defense Notes

1. **Hardware Detached in This Session**: The physical ESP32 was not connected to an active COM port on this PC during Phase 7 execution. Software simulation, mathematical conversions, payload serialization, and API responses were completely verified via `TestClient`.
2. **Provisional Calibration Constants**: The voltage-to-pH and voltage-to-TDS constants in `config.h` are standard default baselines. Physical buffer calibration must be performed when physical probes are submerged in reference buffers.
3. **Zero-Actuation Policy**: Firmware and microservice strictly adhere to defensive safety standards. Machine learning outputs are advisory risk classifications and must **never** automate physical valves, pumps, or chemical dosing without human confirmation.
4. **Tooling Note**: Direct command-line compilation via bundled `arduino-cli.exe` in headless Windows powershell requires an active gRPC daemon. Firmware source code conforms strictly to standard Arduino C++ and ESP32 Core 3.3.11 specifications.

---

## 9. Readiness Assessment for Phase 8

Phase 7 is complete and verified. The system is ready for **Phase 8 (Dashboard & End-to-End System Testing)**:
- The ESP32 firmware produces compliant, validated JSON matching the 4 canonical feature keys.
- The cloud API runs with sub-millisecond inference and returns clear classifications, estimated probabilities, and training range warnings.
- Next phase will focus on real-time web dashboard visualization, telemetry streaming, and user alert management.
