# Phase 7.2 Verification Report: ESP32 ADC Accuracy & Sensor-Reading Validation

**Project Title**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning  
**Course**: Machine Learning for Cloud Services  
**Phase**: Phase 7.2 — ESP32 ADC Accuracy and Sensor-Reading Validation Pass  
**Author**: Lead Embedded Systems & IoT Quality Guardian  
**Target Hardware**: ESP32-WROOM-32 (30-pin / 38-pin DevKitC / NodeMCU-32S)  
**Toolchain**: Arduino CLI v1.5.1 / ESP32 Core v3.3.11 (`esp32:esp32:esp32`)  
**Executed Notebook**: [`notebooks/07b_phase7_adc_validation.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07b_phase7_adc_validation.ipynb)  
**Date**: October 2026  

---

## 1. Executive Summary & Audit Context

Phase 7.2 executes a focused corrective pass on the embedded IoT subsystem prior to commencing end-to-end integration (Phase 8). In accordance with non-negotiable project instructions:
- The authoritative dataset (`data/water_quality_dataset1.csv`) and the trained machine learning models (`models/decision_tree.joblib`, `models/random_forest.joblib`) were **strictly preserved without retraining or modification**.
- No physical sensor measurements, wiring connections, or live hardware telemetry were fabricated.
- Software simulation, toolchain compilation, and unperformed physical hardware testing are rigorously distinguished.
- Automatic physical actuation (e.g., pumps, chemical dosing, valves) remains **defensively prohibited**.

### Key Accomplishments of Phase 7.2:
1. **Calibrated ADC Voltage Acquisition**:
   - Replaced naive linear discrete count mappings (`counts / 4095.0 * 3.3V`) with the factory-calibrated `analogReadMilliVolts(pin)` method.
   - Evaluated classic ESP32 SAR ADC characteristics under `ADC_11db` attenuation: linear measurement range ($150\,\text{mV} - 2450\,\text{mV}$), non-linear compression knee ($2450\,\text{mV} - 3100\,\text{mV}$), and upper saturation.
   - Documented the Python linear formula as an idealized uncalibrated simulation model.
   - Added unit conversion helper `milliVoltsToVolts()` in C++ and `millivolts_to_volts()` in Python.
2. **Maximum Sensor Output & Voltage Divider Headroom Analysis**:
   - Conducted an in-depth hardware variant analysis of pH modules (PH-4502C with LM358 op-amp vs rail-to-rail clones vs DFRobot SEN0161) and turbidity modules (TS-300B / SEN0189).
   - Evaluated electrical headroom against the ESP32 absolute maximum rating ($3.60\,\text{V}$) across three standard resistor divider pairs:
     - **Option A (Baseline: $10\,\text{k}\Omega : 20\,\text{k}\Omega$, $M = 1.50\times$)**: At $4.50\,\text{V}$, $V_{\text{pin}} = 3.00\,\text{V} \le 3.30\,\text{V}$. At $5.00\,\text{V}$ rail saturation, $V_{\text{pin}} = 3.33\,\text{V} \le 3.60\,\text{V}$ (safe from electrical damage).
     - **Option B (Conservative: $10\,\text{k}\Omega : 15\,\text{k}\Omega$, $M = 1.667\times$)**: At $5.00\,\text{V}$, $V_{\text{pin}} = 3.00\,\text{V}$ ($+600\,\text{mV}$ headroom below 3.6V).
     - **Option C (Ultra-Conservative: $10\,\text{k}\Omega : 10\,\text{k}\Omega$, $M = 2.00\times$)**: At $5.00\,\text{V}$, $V_{\text{pin}} = 2.50\,\text{V}$ (matches the $2450\,\text{mV}$ linear ceiling of 11 dB attenuation).
   - Reconfirmed that the TDS probe (powered at 3.3V, output $\le 2.30\,\text{V}$) connects directly without divider attenuation ($M = 1.0\times$).
3. **Defensive Sensor Conversion & Fault Handling**:
   - Preserved sensor-side calibration convention for pH; divider scale is reconstructed exactly once.
   - Added strict non-finite guards: `isnan` or `isinf` voltages, temperatures, or calibration constants immediately return `NAN` in C++ or raise `ValueError` in Python.
   - Handled DS18B20 85.0°C power-up value: treated as a startup fault (`DS18B20_FAULT_RESET`) **only** when temperature conversion has not successfully completed. If verified completed across consecutive cycles, accepted as genuine boiling water.
   - Prevented failed, disconnected (-127°C), CRC-corrupted (-126°C), or stale sensor values from silently entering the prediction payload.
4. **Firmware Toolchain Compilation Audit**:
   - Successfully compiled the sketch using official `arduino-cli` v1.5.1 targeting `esp32:esp32:esp32` (ESP32 Dev Module, Core v3.3.11).
   - Confirmed resource utilization: Flash: 1,036,304 B (79.1%), RAM: 48,544 B (14.8%). Zero compiler warnings.
5. **Automated Regression Suite**:
   - Expanded unit and regression suite to 54 tests (20 API, 17 Explainability, 17 IoT Integration). 100% pass rate.
   - Pyright static analysis returned 0 errors.

---

## 2. Artifacts Created & Modified in Phase 7.2

| File Path | Status | Purpose & Description |
| :--- | :---: | :--- |
| [`src/iot_simulation.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\src\iot_simulation.py) | Modified | Added `millivolts_to_volts()`, documented ideal linear simulation model, added non-finite guards, and updated conversion completed logic. |
| [`firmware/cloudguard_esp32/sensor_conversions.h`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\firmware\cloudguard_esp32\sensor_conversions.h) | Modified | Added `milliVoltsToVolts()`, documented ideal linear ADC model, added non-finite guards returning `NAN`, and updated `checkSignalHealth`. |
| [`firmware/cloudguard_esp32/cloudguard_esp32.ino`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\firmware\cloudguard_esp32\cloudguard_esp32.ino) | Modified | Replaced raw ADC count reading with `analogReadMilliVolts()`, added DS18B20 conversion verification, and prevented stale/failed telemetry dispatch. |
| [`firmware/cloudguard_esp32/config.h.example`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\firmware\cloudguard_esp32\config.h.example) | Modified | Documented voltage divider headroom design options (A, B, C), provisional calibration status, and calibrated ADC parameters. |
| [`firmware/cloudguard_esp32/config.h`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\firmware\cloudguard_esp32\config.h) | Modified | Local configuration header aligned with verified resistor values and calibrated ADC sampling. |
| [`reports/phase7_wiring_reference.md`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\phase7_wiring_reference.md) | Modified | Updated with calibrated ADC architecture, headroom comparison table, and fault rejection details. |
| [`tests/test_iot_integration.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\tests\test_iot_integration.py) | Modified | Expanded to 17 comprehensive unit, boundary-value, golden-vector, headroom, and stale telemetry rejection tests. |
| [`src/create_phase7b_notebook.py`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\src\create_phase7b_notebook.py) | Created | Automation script building and executing the Phase 7.2 verification notebook. |
| [`notebooks/07b_phase7_adc_validation.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\notebooks\07b_phase7_adc_validation.ipynb) | Created & Executed | Executed verification notebook documenting ADC accuracy, headroom curves, golden vectors, toolchain compilation, and verification matrix. |
| [`reports/phase7b_adc_validation_report.md`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\phase7b_adc_validation_report.md) | Created | Detailed engineering verification report (this document). |
| [`reports/phase7b_analysis_handoff.md`](file:///d:/E0224033/ML_for_Cl_Ser\CL\LB_Proj\reports\phase7b_analysis_handoff.md) | Created | Comprehensive self-contained ChatGPT handoff report. |

---

## 3. ESP32 ADC Accuracy & Calibrated Voltage Acquisition

### 3.1 Nonlinearity of ESP32 SAR ADC
The ESP32 features two 12-bit SAR ADCs. A linear mapping ($V = \text{counts} \times \frac{3.30}{4095}$) suffers from physical hardware non-linearities:
- **Low-voltage deadband**: Inputs below $\approx 100 - 150\,\text{mV}$ read as 0 counts.
- **High-voltage compression knee**: Under 11 dB attenuation, the curve compresses non-linearly above $\approx 2450\,\text{mV}$, saturating before true $3.30\,\text{V}$.
- **Chip-to-chip variation**: Internal reference voltages vary across chips by up to $\pm 10\%$.

### 3.2 Production Firmware Calibration via `analogReadMilliVolts()`
The firmware resolves these physical limitations by calling `analogReadMilliVolts()`:
```cpp
float readAverageADCVoltage(uint8_t pin, int sampleCount, int delayMs) {
    uint32_t sumMilliVolts = 0;
    for (int i = 0; i < sampleCount; i++) {
        sumMilliVolts += analogReadMilliVolts(pin);
        if (delayMs > 0) delay(delayMs);
    }
    float avgMilliVolts = (float)sumMilliVolts / (float)sampleCount;
    return milliVoltsToVolts((uint32_t)avgMilliVolts);
}
```
- **eFuse Factory Tables**: `analogReadMilliVolts()` queries calibration curves burned into eFuses during Espressif factory testing.
- **Operating Attenuation**: `ADC_11db` provides calibrated readings from $150\,\text{mV}$ to $2450\,\text{mV}$ (linear) and up to $\approx 3100\,\text{mV}$ (calibrated).
- **Unit Conversion**: Converted to Volts via `milliVoltsToVolts(uint32_t mv)` before applying divider reconstruction.

---

## 4. Hardware Sensor Variants & Voltage Divider Headroom

### 4.1 Sensor Output Ranges
- **pH Sensor (PH-4502C)**: Powered at 5.0 V. Normal aqueous range (pH 14 to 0) outputs $1.20\,\text{V} - 3.78\,\text{V}$. Open-circuit or saturated op-amp reaches $3.50\,\text{V}$ (LM358) or $5.00\,\text{V}$ (rail-to-rail clones).
- **Turbidity Sensor (TS-300B / SEN0189)**: Powered at 5.0 V strictly. Clear water output is $4.15\,\text{V} - 4.50\,\text{V}$. Open-circuit or high-intensity ambient light reaches $4.80\,\text{V} - 5.00\,\text{V}$.
- **TDS Sensor (SEN0244)**: Powered strictly from 3.3 V rail. Output is physically capped at $\le 2.30\,\text{V}$. Directly connected to GPIO 35 ($M = 1.0\times$).

### 4.2 Resistor Divider Headroom Comparison Table

$$\text{Ratio} = \frac{R_{\text{lower}}}{R_{\text{upper}} + R_{\text{lower}}}, \quad M = \frac{R_{\text{upper}} + R_{\text{lower}}}{R_{\text{lower}}}$$

| Configuration | $R_{\text{upper}}$ ($R_1, R_3$) | $R_{\text{lower}}$ ($R_2, R_4$) | Divider Ratio | Multiplier ($M$) | Pin Voltage at 4.5V | Pin Voltage at 5.0V | Headroom to 3.6V Rating | ESP32 ADC Linear Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option A (Baseline)** | **$10\,\text{k}\Omega$** | **$20\,\text{k}\Omega$** | **0.6667** | **$1.500\times$** | **$3.000\,\text{V}$** | **$3.333\,\text{V}$** | **$+0.267\,\text{V}$** | Safe from damage ($3.33\text{V} < 3.6\text{V}$). Clear water reads at $3.00\text{V}$ (calibrated region). |
| **Option B (Conservative)** | **$10\,\text{k}\Omega$** | **$15\,\text{k}\Omega$** | **0.6000** | **$1.667\times$** | **$2.700\,\text{V}$** | **$3.000\,\text{V}$** | **$+0.600\,\text{V}$** | Recommended for rail-to-rail boards. Even at 5.0V saturation, pin is strictly $\le 3.00\text{V}$. |
| **Option C (Ultra-Conservative)** | **$10\,\text{k}\Omega$** | **$10\,\text{k}\Omega$** | **0.5000** | **$2.000\times$** | **$2.250\,\text{V}$** | **$2.500\,\text{V}$** | **$+1.100\,\text{V}$** | Maximum linear headroom. 5.0V maps to $2.50\text{V}$ (matches 11dB $2450\text{mV}$ linear ceiling). |

- **Parametric Firmware**: Both `sensor_conversions.h` and `src/iot_simulation.py` use parametric formulas `multiplier = (r1 + r2) / r2`. Swapping resistors in `config.h` automatically recomputes the exact multiplier without code changes.

---

## 5. Defensive Sensor Conversions & Fault Handling

### 5.1 pH 2-Point Linear Calibration (Sensor-Side Domain)
- All calibration constants (`CALIB_PH_V_NEUTRAL = 2.50 V`, `CALIB_PH_V_ACIDIC = 3.05 V`) reside in the **sensor-side domain**.
- Divider reconstruction ($V_{\text{sensor}} = V_{\text{ADC}} \times 1.50$) is applied **exactly once** before conversion.
- Clamped strictly to $[0.00, 14.00]$. Non-finite inputs return `NAN` in C++ or raise `ValueError` in Python.

### 5.2 TDS Conversion (Direct Domain)
- Reconstructed voltage equals ADC pin voltage ($M = 1.0\times$).
- Temperature compensation normalized to $25.0^\circ\text{C}$ ($\alpha = 0.02\,/^\circ\text{C}$).
- Multiplied by $0.5$ NaCl factor and provisional $K_{\text{cell}} = 1.00$.

### 5.3 Turbidity Continuous Root Normalization
- Solves quadratic root at $V_{\text{root}} \approx 4.20025\,\text{V}$.
- Continuous normalization: $V_{\text{norm}} = V_{\text{sensor}} \times \left(\frac{4.20025}{V_{\text{clear}}}\right)$.
- Capped at $3000.0\,\text{NTU}$ at high opacity ($V_{\text{norm}} \le 2.50\,\text{V}$).
- Explicitly labeled **PROVISIONAL and UNCALIBRATED** pending certified Formazin standards.

### 5.4 DS18B20 85.0°C Power-Up State Handling
- Scratchpad power-on default is `0x0550` ($+85.0^\circ\text{C}$).
- When reading 85.0°C, the firmware triggers a second complete 750 ms conversion cycle. If the second conversion still reads 85.0°C and passes Dallas CRC-8, it is accepted as genuine boiling water.
- If conversion was interrupted or read before completion (`conversion_completed == False`), it is rejected as a startup fault (`DS18B20_FAULT_RESET`).

### 5.5 Prevention of Stale / Failed Telemetry Dispatch
- In `cloudguard_esp32.ino`: If `!isHealthy` or if any converted physical unit is NaN, telemetry transmission is immediately aborted (`return`).
- In `build_iot_payload()`: Passing fault codes (`-127.0`, `-126.0`), NaNs, Infinities, or out-of-scale values raises `ValueError`, guaranteeing failed or stale data cannot silently enter the prediction microservice.

---

## 6. Real Firmware Toolchain Compilation Results

The sketch was compiled using the confirmed Arduino CLI toolchain targeting the ESP32 Dev Module.

### Exact Compilation Command:
```powershell
& "C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe" compile --fqbn esp32:esp32:esp32 "firmware/cloudguard_esp32"
```

### Confirmed Toolchain Output:
```text
Sketch uses 1036304 bytes (79%) of program storage space. Maximum is 1310720 bytes.
Global variables use 48544 bytes (14%) of dynamic memory, leaving 279136 bytes for local variables. Maximum is 327680 bytes.
```
- **Exit Code**: `0 (Success)`
- **Compiler Warnings / Errors**: 0
- **Flash Utilization**: 79.1% ($1,036,304\,\text{bytes}$ of 1.31 MB app partition)
- **RAM Utilization**: 14.8% ($48,544\,\text{bytes}$ of 320 KB SRAM)
- **Remaining Free Stack/Heap**: $279,136\,\text{bytes}$ available for runtime execution

---

## 7. Master Verification Matrix

| Scope | Subsystem / Component | Method / Environment | Result / Status | Notes & Constraints |
| :--- | :--- | :--- | :---: | :--- |
| **Software Simulation** | ADC Ideal Linear Mapping | Python unit test / `src.iot_simulation` | **CONFIRMED PASS** | Ideal simulation model documented; bounds $[0, 4095]$. |
| **Software Simulation** | Millivolts to Volts Conversion | Golden vector unit test | **CONFIRMED PASS** | $2450\,\text{mV} \to 2.450\,\text{V}$; non-finite rejected. |
| **Software Simulation** | Divider Headroom (Options A, B, C) | Parametric formula test ($1.50\times, 1.667\times, 2.00\times$) | **CONFIRMED PASS** | Safe headroom verified for $5.0\,\text{V}$ rail saturation. |
| **Software Simulation** | pH 2-Point Linear Calibration | Domain consistency unit test | **CONFIRMED PASS** | Sensor-side domain verified; clamps to $[0.0, 14.0]$. |
| **Software Simulation** | TDS SEN0244 Transfer Function | Golden vectors (1.0V, 2.0V, 35°C, $K_{\text{cell}}$) | **CONFIRMED PASS** | $367.47\,\text{ppm}, 879.35\,\text{ppm}$ verified. |
| **Software Simulation** | Turbidity $C^0$ Continuity | Step delta test at $V_{\text{clear}}$ | **CONFIRMED PASS** | 181.56 NTU discontinuity completely eliminated. |
| **Software Simulation** | Dallas 1-Wire CRC-8 Checksum | Maxim/Dallas test vector verification | **CONFIRMED PASS** | 9-byte scratchpad validation verified. |
| **Software Simulation** | DS18B20 85.0°C Startup Logic | Conversion completion verification | **CONFIRMED PASS** | Distinguishes POR startup state from genuine 85°C. |
| **Software Simulation** | Defensive Telemetry Protection | `build_iot_payload` fault rejection | **CONFIRMED PASS** | Stale data, NaNs, and fault codes strictly rejected. |
| **Software Simulation** | FastAPI /predict Ingestion | `TestClient` simulated payload dispatch | **CONFIRMED PASS** | Status 200 OK, decision tree inference verified. |
| **Firmware Toolchain** | Arduino C++ Sketch Compilation | Arduino CLI v1.5.1 / ESP32 Core v3.3.11 | **CONFIRMED PASS** | Flash: 79.1% (1.03 MB), RAM: 14.8% (48.5 KB). |
| **Firmware Toolchain** | `analogReadMilliVolts()` Sampling | Calibrated eFuse sampling implementation | **CONFIRMED PASS** | Factory calibration curves queried at runtime. |
| **Physical Hardware** | Multimeter Rail Isolation & Ground | DMM continuity and resistance checks | **PENDING HARDWARE** | Pre-power checklist established. Awaiting lab setup. |
| **Physical Hardware** | Standard pH 7.00 & 4.00 Wet Calibration | Glass electrode in buffer solutions | **PENDING HARDWARE** | Probe wetting, soaking, and offset tuning in lab. |
| **Physical Hardware** | TDS 1413 µS/cm Reference Tuning | Standard conductivity solution | **PENDING HARDWARE** | $K_{\text{cell}}$ factor fine-tuning pending lab. |
| **Physical Hardware** | Formazin Turbidity Multi-Point | Certified Formazin calibration standards | **PENDING HARDWARE** | Optical NTU transfer curve labeled provisional. |
| **Physical Hardware** | Live Wi-Fi Transmission to Host | ESP32 to host PC over 2.4 GHz LAN | **PENDING HARDWARE** | Microcontroller flash & live test pending lab. |

---

## 8. Explicit Hardware Assumptions & Variant Notes

> [!IMPORTANT]
> **HARDWARE AVAILABILITY & VARIANT ASSUMPTIONS**:
> - Until physical hardware is delivered and verified in-lab, all pinouts, resistor values, and voltage curves assume canonical DFRobot / Gravity and Dallas Semiconductor modules.
> - **pH Module Variant**: If a 3.3V-native pH adapter board is used instead of the 5V PH-4502C, resistor $R_1$ can be replaced with a 0-ohm jumper, and `PH_DIVIDER_R1` set to `0.0f` in `config.h`.
> - **Turbidity Provisional Status**: Without certified Formazin standards (0, 20, 100, 800 NTU), the optical NTU transfer function provides a **provisional, uncalibrated estimate**.

---

**Phase 7.2 is complete. In accordance with non-negotiable project rules, work on Phase 8 has not been initiated. The system is paused awaiting review.**
