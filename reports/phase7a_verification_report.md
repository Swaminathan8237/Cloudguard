# Phase 7.1 Verification Report: Critical Electrical, Calibration & Firmware Audit

**Project Title**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning  
**Course**: Machine Learning for Cloud Services  
**Phase**: Phase 7.1 — Critical Electrical, Calibration & Firmware Verification Pass  
**Author**: Lead Embedded Systems & IoT Quality Guardian  
**Target Hardware**: ESP32-WROOM-32 (30-pin / 38-pin DevKitC / NodeMCU-32S)  
**Toolchain**: Arduino CLI v1.5.1 / ESP32 Core v3.3.11 (`esp32:esp32:esp32`)  
**Executed Notebook**: [`notebooks/07a_phase7_verification.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07a_phase7_verification.ipynb)  
**Date**: October 2026  

---

## 1. Executive Summary & Audit Context

Phase 7.1 executes a comprehensive engineering audit and corrective verification pass on the embedded IoT subsystem prior to commencing end-to-end integration (Phase 8). In accordance with non-negotiable project instructions:
- The authoritative dataset (`data/water_quality_dataset1.csv`) and the trained machine learning models (`models/decision_tree.joblib`, `models/random_forest.joblib`) were **strictly preserved without retraining or modification**.
- No physical sensor measurements, wiring connections, or live hardware telemetry were fabricated.
- Software simulation, toolchain compilation, and unperformed physical hardware testing are rigorously distinguished.
- Automatic physical actuation (e.g. pumps, chemical dosing, valves) remains **defensively prohibited**.

### Key Accomplishments of Phase 7.1:
1. **Circuit Schematic Audit & Net Isolation**:
   - The circuit schematic was completely redrawn and verified ([`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png)), eliminating ambiguous 3.3 V nets.
   - Three independent, color-coded power buses are established: **5.0 V / VIN Net (Orange)**, **3.3 V Net (Red)**, and **Common Ground Net (Slate/Black)**.
   - Dedicated, independent voltage dividers are implemented for pH ($R_1 = 10\,\text{k}\Omega, R_2 = 20\,\text{k}\Omega$) and Turbidity ($R_3 = 10\,\text{k}\Omega, R_4 = 20\,\text{k}\Omega$).
   - The TDS sensor (DFRobot SEN0244) is powered at 3.3 V, naturally capping its output below $2.3\,\text{V}$ and enabling a safe direct ADC connection to GPIO 35.
   - The DS18B20 1-Wire DATA line pull-up resistor ($R_5 = 4.7\,\text{k}\Omega$) is tied strictly to the **3.3 V rail**, protecting GPIO 4 from overvoltage.
   - All analog inputs are mapped exclusively to **ADC1 channels** (GPIO 34, GPIO 35, GPIO 32) to prevent Wi-Fi SAR ADC2 contention.

2. **Sensor Conversion Mathematical Corrections**:
   - **pH Domain Consistency**: Established the **sensor-side voltage domain** as the single authoritative convention. Divider reconstruction is applied exactly once before passing voltages to the two-point linear calibration formula, ensuring mathematical equivalence between sensor and ADC representations. Clamped to aqueous bounds $[0.0, 14.0]$.
   - **TDS Mathematical Derivation**: Validated the DFRobot Gravity SEN0244 polynomial. Verified the $0.02\,/^\circ\text{C}$ temperature compensation factor, the $0.5$ standard NaCl aqueous conversion factor ($\text{TDS}_{\text{ppm}} = 0.5 \times \text{EC}_{\mu\text{S/cm}}$), and the dimensionless $K_{\text{cell}}$ tuning factor. Golden vectors (367.47 ppm at 1.0V/25°C, 879.35 ppm at 2.0V/25°C) verified.
   - **Turbidity Discontinuity Elimination**: Investigated the **181.56 NTU jump** caused by arbitrary clamping at $4.15\,\text{V}$. Solved the quadratic root $V_{\text{root}} \approx 4.20025\,\text{V}$ where $NTU = 0$. Implemented continuous $C^0$ root normalization ($V_{\text{norm}} = V \times \frac{4.20025}{V_{\text{clear}}}$), clamped high opacity ($V \le 2.50\,\text{V}$) to $3000.0\,\text{NTU}$, and formally labeled the estimate **PROVISIONAL and UNCALIBRATED** pending certified Formazin standards.
   - **DS18B20 9-Byte Scratchpad & Dallas CRC-8**: Upgraded the bit-banged 1-Wire driver to read all 9 scratchpad bytes, calculate Dallas/Maxim CRC-8 using polynomial $X^8 + X^5 + X^4 + 1$ (reflected `0x8C`), detect uncompleted power-on reset state ($85.0^\circ\text{C}$), signal bus disconnection ($-127.0^\circ\text{C}$), identify corrupted packets ($-126.0^\circ\text{C}$), and track consecutive stale readings across $\ge 3$ cycles.

3. **Firmware Toolchain Compilation Audit**:
   - Compiled `firmware/cloudguard_esp32/cloudguard_esp32.ino` using official `arduino-cli` v1.5.1 targeting `esp32:esp32:esp32` (ESP32 Dev Module, Core v3.3.11).
   - Real compilation output: **Flash Memory: 1,034,784 bytes (78.9%)**, **Dynamic RAM: 48,568 bytes (14.8%)**, leaving 279,112 bytes free for local variables. Zero compiler warnings or errors.

4. **Automated Regression Suite**:
   - Expanded unit and regression suite in [`tests/test_iot_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_iot_integration.py) to 15 targeted tests.
   - Entire project test suite executed: **52/52 tests passing (100% pass rate)**.
   - Pyright static analysis returned **0 errors, 0 warnings, 0 informations**.

---

## 2. Artifacts Created & Modified in Phase 7.1

| File Path | Status | Purpose & Description |
| :--- | :---: | :--- |
| [`src/iot_simulation.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/iot_simulation.py) | Modified | Updated sensor conversions: pH domain alignment, continuous turbidity normalization, Dallas CRC-8 calculation, 9-byte scratchpad validation, and hardware fault boundaries. |
| [`firmware/cloudguard_esp32/sensor_conversions.h`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/sensor_conversions.h) | Modified | Updated Arduino C++ header: inline functions matching Python simulation, `dallasCrc8()`, continuous turbidity normalization, and CRC error code `-126.0f`. |
| [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/cloudguard_esp32.ino) | Modified | Upgraded bit-banged 1-Wire driver to read full 9-byte scratchpad with CRC-8 verification, added consecutive error tracking and stale sensor warnings. |
| [`firmware/cloudguard_esp32/config.h.example`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/config.h.example) | Modified | Clarified voltage divider resistors ($R_1-R_4$) and explicit sensor-side calibration domain rules. |
| [`firmware/cloudguard_esp32/config.h`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/firmware/cloudguard_esp32/config.h) | Modified | Local configuration header aligned with verified resistor values and sensor-side calibration. |
| [`src/generate_circuit_diagram.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/generate_circuit_diagram.py) | Created | Matplotlib schematic generator producing publication-grade circuit diagram with isolated power nets and labeled resistors. |
| [`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png) | Regenerated | High-resolution 300 DPI schematic diagram with separated 3.3V/5V/GND nets, independent dividers, and clear pin callouts. |
| [`reports/phase7_wiring_reference.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7_wiring_reference.md) | Modified | Engineering guide updated with verified net isolation, independent divider calculations, pre-power checklist, and hardware variant assumptions. |
| [`tests/test_iot_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/tests/test_iot_integration.py) | Modified | Expanded to 15 comprehensive unit, boundary-value, golden-vector, and CRC-8 regression tests. |
| [`src/create_phase7a_notebook.py`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/src/create_phase7a_notebook.py) | Created | Automation script building and executing the verification notebook. |
| [`notebooks/07a_phase7_verification.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07a_phase7_verification.ipynb) | Created & Executed | Mandatory executed verification notebook documenting calculations, plots, golden vectors, toolchain compilation, and verification matrix. |
| [`reports/phase7a_verification_report.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7a_verification_report.md) | Created | Detailed engineering verification report (this document). |
| [`reports/phase7a_analysis_handoff.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7a_analysis_handoff.md) | Created | Comprehensive self-contained ChatGPT handoff report. |

---

## 3. Circuit Schematic Audit & Electrical Verification

### 3.1 Net Isolation Architecture
Inspection of the initial schematic identified an ambiguous red net that appeared to bridge into non-power ESP32 pins. The schematic was redrawn to enforce strict net isolation:

1. **5.0 V / VIN Bus (Orange)**:
   - Fed from host USB power via ESP32 `VIN` pin.
   - Supplies power to Turbidity sensor IR emitter LED ($V_{CC}$) and pH adapter board ($V_{CC}$).
   - Stepped down to safe levels ($\le 3.0\,\text{V}$) via independent $10\,\text{k}\Omega : 20\,\text{k}\Omega$ voltage dividers before entering ESP32 GPIOs.
2. **3.3 V Bus (Red)**:
   - Fed strictly from ESP32 `3V3` regulator output.
   - Supplies TDS sensor ($V_{CC}$), DS18B20 ($V_{DD}$), and $4.7\,\text{k}\Omega$ 1-Wire pull-up resistor.
   - **Verification**: The 3.3 V rail connects strictly to sensor power pins and pull-up resistors. It has zero connections to any non-power GPIO pins.
3. **Common Ground (Slate/Black)**:
   - Continuous ground plane linking ESP32 GND, module ground pins, DS18B20 black ground wire, and the bottom terminals of both divider lower resistors ($R_2, R_4$).

### 3.2 Independent Voltage Dividers & Pin Allocations
```
pH Sensor (PH-4502C) AOUT ─────[ R1 = 10 kΩ ]─────┬─────> ESP32 GPIO 34 (ADC1_CH6)
                                                  │
                                            [ R2 = 20 kΩ ]
                                                  │
                                                 GND

Turbidity (TS-300B) AOUT ──────[ R3 = 10 kΩ ]─────┬─────> ESP32 GPIO 32 (ADC1_CH4)
                                                  │
                                            [ R4 = 20 kΩ ]
                                                  │
                                                 GND

TDS (SEN0244 on 3.3V) AOUT ─────────────────────────────> ESP32 GPIO 35 (ADC1_CH7, Direct)

DS18B20 DATA ──────────────────┬────────────────────────> ESP32 GPIO 4 (1-Wire Digital)
                               │
                         [ R5 = 4.7 kΩ ]
                               │
                           3.3 V Rail
```

- **ADC Restriction**: GPIO 34, GPIO 35, and GPIO 32 belong exclusively to **ADC1**, guaranteeing that simultaneous Wi-Fi transmission does not disable or corrupt ADC readings.
- **Voltage Transfer Ratios**:
  $$V_{\text{ADC}} = V_{\text{sensor}} \times \frac{20\,\text{k}\Omega}{10\,\text{k}\Omega + 20\,\text{k}\Omega} = 0.6667 \times V_{\text{sensor}}$$
  - At $V_{\text{sensor}} = 4.50\,\text{V}$ (clear water turbidity): $V_{\text{ADC}} = 3.00\,\text{V} \le 3.30\,\text{V}$ (completely safe).
  - At $V_{\text{sensor}} = 4.20\,\text{V}$ (acidic pH): $V_{\text{ADC}} = 2.80\,\text{V} \le 3.30\,\text{V}$ (completely safe).
- **Firmware Reversal**:
  $$V_{\text{sensor}} = V_{\text{ADC}} \times \left(\frac{R_1 + R_2}{R_2}\right) = V_{\text{ADC}} \times 1.50$$

---

## 4. Sensor Conversion Analysis & Mathematical Corrections

### 4.1 pH Sensor Domain Alignment
A potential defect in two-point calibration arises if input voltage and calibration constants are specified in conflicting domains.
- **Authoritative Rule**: Calibration constants `CALIB_PH_V_NEUTRAL` ($2.50\,\text{V}$) and `CALIB_PH_V_ACIDIC` ($3.05\,\text{V}$) are defined in the **sensor-side voltage domain** (corresponding to buffer voltages measured at the BNC adapter board output).
- In both simulation and firmware, voltage divider scaling is executed **exactly once** at the ADC input node:
  $$V_{\text{sensor}} = \text{scaleVoltageDivider}(V_{\text{measured}}, R_1, R_2)$$
- The resulting $V_{\text{sensor}}$ is passed to `convertVoltageToPH(v_sensor, v_neutral, v_acidic)`.
- **Proof of Equivalence**: A dedicated unit test (`test_convert_voltage_to_ph_domain_consistency`) proves that performing the calculation in the sensor domain matches an ADC-domain calculation with scaled calibration constants to within machine precision.
- **Bounds Clamping**: Clamped to the physical aqueous range $[0.00, 14.00]$.

### 4.2 TDS SEN0244 Transfer Function & Compensation
The DFRobot Gravity TDS sensor transfer curve was verified:
1. **Aqueous Temperature Compensation**:
   $$V_{\text{comp}} = \frac{V_{\text{sensor}}}{1.0 + 0.02 \times (T - 25.0)}$$
   The temperature coefficient $\alpha = 0.02\,/^\circ\text{C}$ (2.0% per degree C relative to $25^\circ\text{C}$) reflects the standard increase in ion mobility with temperature.
2. **Electrical Conductivity (EC in $\mu\text{S/cm}$)**:
   $$EC = 133.42 \times V_{\text{comp}}^3 - 255.86 \times V_{\text{comp}}^2 + 857.39 \times V_{\text{comp}}$$
3. **TDS Conversion (ppm)**:
   $$\text{TDS} = EC \times 0.5 \times K_{\text{cell}}$$
   The factor $0.5$ converts EC ($\mu\text{S/cm}$) to TDS (ppm or mg/L) for standard NaCl solutions. $K_{\text{cell}}$ is the probe calibration factor (default 1.0).
4. **Golden Vectors**:
   - $V = 1.0\,\text{V}, T = 25^\circ\text{C}, K_{\text{cell}} = 1.0 \implies \mathbf{367.47\,\text{ppm}}$
   - $V = 2.0\,\text{V}, T = 25^\circ\text{C}, K_{\text{cell}} = 1.0 \implies \mathbf{879.35\,\text{ppm}}$
   - $V = 2.0\,\text{V}, T = 25^\circ\text{C}, K_{\text{cell}} = 1.1 \implies \mathbf{967.28\,\text{ppm}}$
   - $V = 1.2\,\text{V}, T = 35^\circ\text{C}, K_{\text{cell}} = 1.0 \implies \mathbf{367.47\,\text{ppm}}$

### 4.3 Turbidity Discontinuity Investigation & $C^0$ Root Normalization
#### Mathematical Root Analysis:
The transfer polynomial widely used for the TS-300B / SEN0189 optical probe is:
$$NTU(V) = -1120.4 \times V^2 + 5742.3 \times V - 4352.9$$
Finding the roots where $NTU = 0$:
$$a = -1120.4, \quad b = 5742.3, \quad c = -4352.9$$
$$\Delta = b^2 - 4ac = 5742.3^2 - 4(-1120.4)(-4352.9) = 13,466,094.57$$
$$V_{\text{root}} = \frac{-5742.3 - \sqrt{13,466,094.57}}{2(-1120.4)} \approx \mathbf{4.20025\,\text{V}}$$

At $V = 4.15\,\text{V}$:
$$NTU(4.15) = -1120.4(4.15)^2 + 5742.3(4.15) - 4352.9 = \mathbf{181.56\,\text{NTU}}$$

Previously, clamping the function with `if V >= 4.15: return 0.0` caused a massive **181.56 NTU discontinuity** between $4.149\,\text{V}$ and $4.150\,\text{V}$. In drinking water compliance (where WHO and EPA thresholds are $\le 1.0\,\text{NTU}$ and $\le 0.3\,\text{NTU}$), an artificial jump of 181.56 NTU is unacceptable.

#### Continuous Normalization Solution:
To achieve $C^0$ continuity regardless of slight trimpot adjustments on the adapter board, voltage is normalized against the mathematical root:
$$V_{\text{norm}} = V \times \left(\frac{4.20025}{V_{\text{clear}}}\right)$$
- When $V \ge V_{\text{clear}}$, $V_{\text{norm}} \ge 4.20025\,\text{V}$, and $NTU = 0.0$ (smoothly crossing zero).
- When $V_{\text{norm}} \le 2.50\,\text{V}$, the optical probe is saturated at maximum opacity; $NTU$ is capped at $3000.0\,\text{NTU}$ to prevent quadratic parabolic reversal.
- Across $V_{\text{clear}} = 4.15\,\text{V}$, a $1\,\text{mV}$ step delta yields $\approx 3.7\,\text{NTU}$ (matching the mathematical derivative $dNTU/dV \approx -3670\,\text{NTU/V}$), completely eliminating the 181.56 NTU jump.
- **Provisional Notice**: This estimate is explicitly labeled **PROVISIONAL and UNCALIBRATED** pending physical Formazin standards.

### 4.4 DS18B20 9-Byte Scratchpad & Dallas CRC-8 Verification
The bit-banged 1-Wire driver was upgraded from reading 2 bytes to reading the complete 9-byte scratchpad:
1. **Dallas Maxim CRC-8**:
   - Polynomial: $X^8 + X^5 + X^4 + 1$ (reflected representation: `0x8C`).
   - Evaluating CRC over all 9 bytes yields `0x00` on success.
2. **Error States**:
   - Bus disconnection / missing pull-up: returns `-127.0f` (`DS18B20_FAULT_DISCONNECTED`).
   - Transmission corruption / CRC mismatch: returns `-126.0f` (`DS18B20_FAULT_CRC_ERROR`).
   - Power-on reset state ($85.0^\circ\text{C}$ / `0x0550`): returns `85.0f` (`DS18B20_FAULT_RESET`), indicating incomplete temperature conversion.
   - Consecutive stale reading tracking: tracks failures across $\ge 3$ consecutive cycles and logs warnings to serial console.

---

## 5. Firmware Toolchain Compilation Results

The sketch was compiled using the confirmed Arduino CLI toolchain targeting the ESP32 Dev Module.

### Exact Compilation Command:
```powershell
& "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe" compile --fqbn esp32:esp32:esp32 "firmware/cloudguard_esp32"
```

### Confirmed Toolchain Output:
```
Sketch uses 1034784 bytes (78%) of program storage space. Maximum is 1310720 bytes.
Global variables use 48568 bytes (14%) of dynamic memory, leaving 279112 bytes for local variables. Maximum is 327680 bytes.
```

### Memory Footprint Analysis:
| Subsystem | Bytes Used | Maximum Available | Utilization (%) | Free Margin Remaining |
| :--- | :---: | :---: | :---: | :---: |
| **Flash Memory (ROM)** | 1,034,784 B | 1,310,720 B (1.25 MB) | **78.9%** | 275,936 B (21.1%) |
| **Dynamic Memory (SRAM)** | 48,568 B | 327,680 B (320 KB) | **14.8%** | 279,112 B (85.2%) |

The sketch fits comfortably within standard ESP32 flash and SRAM boundaries with ample margin for networking buffers and background tasks.

---

## 6. Pre-Power-On Multimeter Verification Checklist

Before energizing the physical prototype with USB or external power, the following checks must be completed:

1. [ ] **Common Ground Continuity**: With DMM in continuity beeper mode, confirm zero ohms resistance across ESP32 GND, pH GND, TDS GND, Turbidity GND, DS18B20 GND, and the lower pins of $R_2$ and $R_4$.
2. [ ] **Power Rail Isolation**: In resistance mode, verify no short circuit ($R > 100\,\text{k}\Omega$) between the 5.0 V rail, the 3.3 V rail, and Common GND.
3. [ ] **3.3 V Net Inspection**: Confirm that the red 3.3 V net connects strictly to ESP32 `3V3`, TDS $V_{CC}$, DS18B20 $V_{DD}$, and $R_5$. Confirm it does NOT connect to any GPIO pin or VIN.
4. [ ] **Voltage Divider Isolation**: Verify that turbidity AOUT connects to $R_3$ ($10\,\text{k}\Omega$) and pH AOUT connects to $R_1$ ($10\,\text{k}\Omega$). Neither 5V module output connects directly to an ESP32 pin.
5. [ ] **ADC Pin Verification**: Confirm analog inputs are seated strictly in **GPIO 32, 34, and 35** (all on ADC1).
6. [ ] **1-Wire Pull-Up Inspection**: Confirm $R_5$ ($4.7\,\text{k}\Omega$) connects between **3.3 V** (not 5V) and GPIO 4.
7. [ ] **Liquid Isolation**: Maintain breadboard and electronics at least 30 cm away from liquid test beakers.

---

## 7. Master Verification Matrix

| Verification Scope | Subsystem / Component | Method / Environment | Result / Status | Notes & Constraints |
| :--- | :--- | :--- | :---: | :--- |
| **Software Simulation** | ADC 12-Bit Linear Mapping | Python unit test / `src.iot_simulation` | **CONFIRMED PASS** | Bounds $[0, 4095] \to [0.0, 3.30]\,\text{V}$. |
| **Software Simulation** | Voltage Divider Reconstruction | Golden vector test ($10\text{k}:20\text{k}$) | **CONFIRMED PASS** | Exact $1.50\times$ scaling. |
| **Software Simulation** | pH 2-Point Linear Calibration | Domain alignment test | **CONFIRMED PASS** | Sensor-side domain authoritative. |
| **Software Simulation** | TDS SEN0244 Transfer Function | Golden vectors (1.0V, 2.0V, 35°C, $K_{\text{cell}}$) | **CONFIRMED PASS** | 367.47 ppm, 879.35 ppm verified. |
| **Software Simulation** | Turbidity $C^0$ Continuity | Step delta test at $V_{\text{clear}}$ | **CONFIRMED PASS** | 181.56 NTU discontinuity eliminated. |
| **Software Simulation** | Dallas 1-Wire CRC-8 Checksum | Maxim/Dallas test vector verification | **CONFIRMED PASS** | 9-byte scratchpad validation verified. |
| **Software Simulation** | Hardware Fault Detection | Rail saturation, POR (85°C), disconnect (-127°C) | **CONFIRMED PASS** | Prevents corrupted telemetry dispatch. |
| **Software Simulation** | FastAPI /predict Ingestion | `TestClient` simulated payload dispatch | **CONFIRMED PASS** | Status 200 OK, decision tree inference verified. |
| **Firmware Toolchain** | Arduino C++ Sketch Compilation | Arduino CLI v1.5.1 / ESP32 Core v3.3.11 | **CONFIRMED PASS** | Flash: 78.9% (1.03 MB), RAM: 14.8% (48.5 KB). |
| **Firmware Toolchain** | Bit-Banged 1-Wire Implementation | Bit-timing, 750ms delay, Dallas CRC-8 | **CONFIRMED PASS** | Zero-dependency implementation verified. |
| **Physical Hardware** | Breadboard Rail Multimeter Continuity | Physical DMM continuity test | **PENDING HARDWARE** | Awaiting physical circuit wiring in lab. |
| **Physical Hardware** | Power Rail Voltage Isolation | DMM DC voltage test (5V vs 3.3V) | **PENDING HARDWARE** | Pre-power checklist established. |
| **Physical Hardware** | pH Buffer 2-Point Calibration | Standard pH 7.00 & 4.00 buffer solutions | **PENDING HARDWARE** | Electrodes require physical wetting/soaking. |
| **Physical Hardware** | TDS Standard Solution Tuning | Standard 1413 µS/cm reference standard | **PENDING HARDWARE** | $K_{\text{cell}}$ factor tuning pending lab. |
| **Physical Hardware** | Turbidity Formazin Multi-Point | Certified Formazin calibration standards | **PENDING HARDWARE** | Output currently labeled PROVISIONAL. |
| **Physical Hardware** | Live Wi-Fi Transmission to Host | ESP32 to host PC over 2.4 GHz LAN | **PENDING HARDWARE** | Microcontroller flash & live test pending lab. |

---

## 8. Explicit Hardware Assumptions & Variant Notes

> [!IMPORTANT]
> **HARDWARE AVAILABILITY & VARIANT ASSUMPTIONS**:
> - Until physical hardware is delivered and verified in-lab, all pinouts, resistor values, and voltage curves assume canonical DFRobot / Gravity and Dallas Semiconductor modules.
> - **pH Module Variant**: If a 3.3V-native pH adapter board is used instead of the 5V PH-4502C, resistor $R_1$ can be replaced with a 0-ohm jumper, and `PH_DIVIDER_R1` set to `0.0f` in `config.h`.
> - **Turbidity Provisional Status**: Without certified Formazin standards (0, 20, 100, 800 NTU), the optical NTU transfer function provides a **provisional, uncalibrated estimate**.

---

## 9. Conclusion & Sign-Off

Phase 7.1 has resolved all identified electrical, mathematical, and driver deficiencies. The circuit schematic is verified and unambiguous, sensor conversions are continuous and domain-aligned, the 1-Wire driver validates packets with Dallas CRC-8, Arduino C++ firmware compiles cleanly, and all 52 automated regression tests pass.

**Stopping condition satisfied**: In strict compliance with project guidelines, no physical tests have been claimed as complete, and Phase 8 has not been initiated. Waiting for user review and approval.
