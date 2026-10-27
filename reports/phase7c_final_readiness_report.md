# Phase 7.3 Verification Report: Final ADC Headroom Decision and Hardware Commissioning Readiness

**Project Title**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification Using Machine Learning  
**Course**: Machine Learning for Cloud Services  
**Phase**: Phase 7.3 — Final ADC Headroom Decision and Hardware Commissioning Readiness Pass  
**Author**: Lead Embedded Systems & IoT Quality Guardian  
**Target Hardware**: ESP32-WROOM-32 (30-pin / 38-pin DevKitC / NodeMCU-32S)  
**Toolchain**: Arduino CLI v1.5.1 / ESP32 Core v3.3.11 (`esp32:esp32:esp32`)  
**Executed Notebook**: [`notebooks/07c_phase7_final_readiness.ipynb`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/notebooks/07c_phase7_final_readiness.ipynb)  
**Wiring Reference**: [`reports/phase7_wiring_reference.md`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/phase7_wiring_reference.md)  
**Circuit Diagram**: [`reports/figures/phase7_iot_wiring_diagram.png`](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png)  
**Date**: October 2026  

---

## 1. Executive Summary & Audit Context

Phase 7.3 executes the final focused engineering correction and hardware readiness verification pass prior to beginning Phase 8 (End-to-End System Integration and Dashboard Verification). In strict adherence to non-negotiable project instructions:

1. **Dataset and Trained Models Preserved**: The canonical dataset (`data/water_quality_dataset1.csv`) and all trained machine learning models (`models/decision_tree.joblib`, `models/random_forest.joblib`, `models/logistic_regression.joblib`) were **strictly preserved without alteration or retraining**.
2. **No Hardware Results Fabricated**: No physical hardware measurements, multimeter readings, probe calibrations, or live Wi-Fi transmission logs were invented.
3. **Strict Scope Separation**: Software unit tests, firmware toolchain compilation, and unperformed physical hardware verification are rigorously segregated.
4. **No Actuation**: Automatic physical equipment actuation (valves, pumps, chemical dosing) remains strictly prohibited.
5. **No Dashboard Started**: Dashboard development has not been started; Phase 8 remains strictly paused pending user review.

---

## 2. Final Voltage-Divider Headroom Decision: Option D

### 2.1 The Engineering Dilemma of Options A, B, and C
Analog sensor modules powered from the 5.0 V rail (e.g., TS-300B optical turbidity and PH-4502C analog pH conditioning boards) can swing up to $5.00\,\text{V}$ under maximum illumination, extreme chemical bounds, or fault saturation.
- **Option A ($10\,\text{k}\Omega : 20\,\text{k}\Omega$, $M = 1.500\times$, Ratio $0.6667$)**:
  - $5.00\,\text{V}$ maps to $3.333\,\text{V}$.
  - While below the $3.60\,\text{V}$ absolute maximum rating, it significantly exceeds the ESP32 classic SAR ADC effective accuracy linear limit ($2.45\,\text{V}$) and operates in the saturation compression knee, leaving only $267\,\text{mV}$ of margin to electrical damage.
- **Option B ($10\,\text{k}\Omega : 15\,\text{k}\Omega$, $M = 1.667\times$, Ratio $0.6000$)**:
  - $5.00\,\text{V}$ maps to $3.000\,\text{V}$.
  - Still operates in the non-linear compression knee above $2.45\,\text{V}$.
- **Option C ($10\,\text{k}\Omega : 10\,\text{k}\Omega$, $M = 2.000\times$, Ratio $0.5000$)**:
  - $5.00\,\text{V}$ maps to $2.500\,\text{V}$.
  - Substantially better, but $2.50\,\text{V}$ still slightly exceeds the stated $2.45\,\text{V}$ linear ceiling by $50\,\text{mV}$.

### 2.2 Specification & Adoption of Option D
To ensure that all normal operational signals and worst-case $5.0\,\text{V}$ fault voltages remain strictly within the linear accuracy zone with verified margin, **Option D is officially adopted**:
- **Upper Resistor ($R_{\text{upper}} = R_1 = R_3$)**: $15\,\text{k}\Omega$ ($1\%$ metal-film, standard E24 series).
- **Lower Resistor ($R_{\text{lower}} = R_2 = R_4$)**: $10\,\text{k}\Omega$ ($1\%$ metal-film, standard E24 series).
- **Divider Transfer Ratio**:
  $$\text{Ratio} = \frac{R_2}{R_1 + R_2} = \frac{10\,\text{k}\Omega}{15\,\text{k}\Omega + 10\,\text{k}\Omega} = 0.4000$$
- **Reconstruction Multiplier**:
  $$M = \frac{R_1 + R_2}{R_2} = \frac{15\,\text{k}\Omega + 10\,\text{k}\Omega}{10\,\text{k}\Omega} = 2.5000\times$$

### 2.3 Headroom & Voltage Mapping Across the Operating Range

| Sensor Operating Condition | Sensor Output ($V_{\text{sensor}}$) | ESP32 Pin Voltage ($V_{\text{pin}} = V_{\text{sensor}} \times 0.4$) | Margin below 2.45V Linear Ceiling | Margin below 3.6V Abs Max Rating | Operating Region Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Neutral pH Buffer (pH 7.00)** | $2.500\,\text{V}$ | $1.000\,\text{V}$ | $+1450\,\text{mV}$ | $+2.600\,\text{V}$ | Optimal linear accuracy sweet spot |
| **Acidic Calibration (pH 4.00)** | $3.050\,\text{V}$ | $1.220\,\text{V}$ | $+1230\,\text{mV}$ | $+2.380\,\text{V}$ | Optimal linear accuracy sweet spot |
| **Extreme Acidic Limit (pH 0.00)** | $3.780\,\text{V}$ | $1.512\,\text{V}$ | $+938\,\text{mV}$ | $+2.088\,\text{V}$ | Well within linear accuracy window |
| **Clear Water Turbidity** | $4.500\,\text{V}$ | $1.800\,\text{V}$ | $+650\,\text{mV}$ | $+1.800\,\text{V}$ | Well within linear accuracy window |
| **5.0V Sensor Rail Saturation Fault** | $5.000\,\text{V}$ | $2.000\,\text{V}$ | **+450 mV** | **+1.600 V** | **PASS: Strictly linear; safe from damage** |

**Key Takeaways**:
1. Even under a catastrophic $5.00\,\text{V}$ module short or rail saturation, the pin voltage reaches only $2.000\,\text{V}$, preserving **$+450\,\text{mV}$ of linear headroom** below the $2.45\,\text{V}$ ceiling.
2. The pin voltage provides **$+1.60\,\text{V}$ of electrical safety margin** below the $3.60\,\text{V}$ damage threshold.
3. Signal resolution remains high: the optical turbidity dynamic span ($2.50\,\text{V}$ to $4.50\,\text{V}$) maps to an $800\,\text{mV}$ window on the ADC, yielding approximately $990$ discrete counts under 12-bit acquisition.

---

## 3. Calibrated ADC Wording & Electrical Domain Audit

### 3.1 Terminology Correction
The documentation and code comments have been rigorously audited:
- **Implementation Reality**: The firmware queries `analogReadMilliVolts(pin)`, which provides calibrated millivolt readings according to the installed Arduino-ESP32 core library implementation.
- **No Unsubstantiated eFuse Claims**: Documentation does not assert that a specific hardware eFuse calibration scheme (such as "Two Point" or "eFuse Vref") was burned into the specific target ESP32 board. Classic ESP32 chips come from various foundry batches and silicon revisions with varying eFuse states; true hardware eFuse interrogation requires running `espefuse.py` on the physical silicon.

### 3.2 Four Distinct Electrical Domains

```
0.0 V           0.15 V                     2.00 V     2.45 V           3.10 V        3.60 V
  |---|------------|--------------------------|----------|----------------|-------------|--> Volts
     Dead Zone   Effective Accuracy Range      Opt D     Saturation Knee  Full Scale    ABS MAX
     (< 150 mV)  (150 mV to 2450 mV)         Fault Max   (2.45V - 3.10V)  (Clamped)     (Damage)
```

1. **Measurable Input Range**: $0\,\text{mV}$ to $\approx 3100\,\text{mV}$ under `ADC_11db` attenuation. Inputs above $3100\,\text{mV}$ clamp to full scale (4095 counts).
2. **Effective Accuracy Range**: $150\,\text{mV}$ to $2450\,\text{mV}$ under `ADC_11db` attenuation. This is the manufacturer-recommended linear region where calibration tables minimize conversion error.
3. **Absolute Maximum Electrical Rating**: $3.60\,\text{V}$ (or $V_{DD} + 0.3\,\text{V}$). Exceeding this threshold causes silicon gate-oxide breakdown and CMOS latch-up.
4. **Multimeter-Verified Bench Accuracy**: Software eFuse calibration provides an algorithmic correction. True voltage accuracy must be verified on the physical workbench using a calibrated Digital Multimeter (DMM).

---

## 4. Multi-Sensor Conversion Domain Verification

### 4.1 pH Probe Domain Consistency
- **Authoritative Rule**: All calibration constants (`CALIB_PH_V_NEUTRAL = 2.50 V`, `CALIB_PH_V_ACIDIC = 3.05 V`) are defined strictly in the **sensor-side voltage domain**.
- **Multiplier Execution**: The Option D multiplier ($M = 2.500\times$) is applied **exactly once** during sensor-side voltage reconstruction before calling `convertVoltageToPH()`.
- **Domain Invariance Proof**: Verified in unit test `test_convert_voltage_to_ph_domain_consistency` that evaluating in the sensor domain or the scaled ADC domain yields mathematically identical pH values when calibration references are aligned to the respective domain.

### 4.2 TDS Probe Direct Measurement & Temperature Compensation
- Powered from the $3.3\,\text{V}$ rail; output naturally bounded below $2.30\,\text{V}$. Direct connection ($M = 1.0\times$).
- Reconstructed voltage equals pin voltage.
- Temperature compensation formula: $V_{\text{comp}} = V / (1.0 + 0.02 \times (T - 25.0))$.
- Golden vector tests verified: $1.0\,\text{V}$ at $25^\circ\text{C} = 367.47\,\text{ppm}$; $1.2\,\text{V}$ at $35^\circ\text{C} = 367.47\,\text{ppm}$.
- Labeled **PROVISIONAL** with $K_{\text{cell}} = 1.00$ pending certified $1413\,\mu\text{S/cm}$ calibration solution.

### 4.3 Turbidity Root Normalization & Provisional Status
- Transfer polynomial: $\text{NTU}(V) = -1120.4 \times V^2 + 5742.3 \times V - 4352.9$.
- Mathematical upper root: $V_{\text{root}} \approx 4.20025\,\text{V}$.
- Continuous normalization eliminates the previous $181.56\,\text{NTU}$ discontinuity at clear water threshold ($4.15\,\text{V}$):
  $$V_{\text{norm}} = V_{\text{sensor}} \times \left(\frac{4.20025}{V_{\text{clear}}}\right)$$
- Delta across threshold is bounded by the mathematical derivative ($\approx 3.7\,\text{NTU}$ per mV).
- Explicitly labeled **PROVISIONAL and UNCALIBRATED** pending certified Formazin standards.

### 4.4 DS18B20 Dallas CRC-8 & Startup Verification
- Upgraded bit-banged 1-Wire driver queries complete 9-byte scratchpad buffer.
- Computes Dallas/Maxim CRC-8 checksum ($X^8 + X^5 + X^4 + 1$, reflected `0x8C`).
- **Power-On Reset State (85.0°C)**:
  - If read during startup without verified conversion completion $\implies$ rejected as `DS18B20_FAULT_RESET` (startup fault).
  - If read and verified across consecutive 750 ms conversion cycles $\implies$ accepted as genuine $85.0^\circ\text{C}$ boiling water.

---

## 5. Corrected Pre-Power-On Protocol

### 5.1 Why the Blanket $> 100\,\text{k}\Omega$ Rule was Flawed
Previous documentation stated that resistance between any power rail and ground must exceed $100\,\text{k}\Omega$ on an unpowered board. This was electrically invalid:
1. **Semiconductor Paths**: Unpowered linear regulators (AMS1117-3.3), microcontrollers, and sensor front-ends contain internal semiconductor junctions, diodes, and ESD protection networks that conduct slightly when probed by a multimeter.
2. **Capacitive Inrush**: Decoupling capacitors ($10\,\mu\text{F}$ electrolytic, $100\,\text{nF}$ ceramic) charge from the multimeter's internal constant-current source, causing readings to start near zero ohms and sweep upward.

### 5.2 The Corrected Two-Phase Protocol
1. **Phase A: Power-Disconnected Inspection (Unpowered)**:
   - DMM in continuity/low-resistance mode ($200\,\Omega$ range).
   - Verify **NO DIRECT DEAD SHORT** ($R < 1.0\,\Omega$ or continuous tone) between:
     - 5.0 V rail and Common GND
     - 3.3 V rail and Common GND
     - 5.0 V rail and 3.3 V rail
   - Verify common ground continuity ($0.0\,\Omega$) across all module GND pins.
   - Verify resistor values ($R_1, R_3 = 15.0\,\text{k}\Omega \pm 1\%$; $R_2, R_4 = 10.0\,\text{k}\Omega \pm 1\%$; $R_5 = 4.7\,\text{k}\Omega \pm 5\%$).
   - Disconnect sensor signal wires from GPIO 32, 34, 35 before initial power-on.
2. **Phase B: Controlled Powered Rail Verification (Powered)**:
   - Apply USB power to ESP32 board.
   - Measure 5.0 V rail with DMM: confirm $4.75\,\text{V} - 5.25\,\text{V}$.
   - Measure 3.3 V rail with DMM: confirm $3.20\,\text{V} - 3.40\,\text{V}$.
   - Measure divider output nodes: confirm $V_{\text{node}} \le 2.00\,\text{V}$ and never exceeds $2.45\,\text{V}$.
   - Once voltages are confirmed safe, disconnect power, seat signal wires into GPIO 32, 34, 35, and re-energize.

---

## 6. Verification & Compilation Metrics

### 6.1 Automated Test Suite Execution
- **Command**: `python -m unittest discover tests`
- **Total Tests**: **55 passing out of 55 tests** (20 API, 17 Explainability, 18 IoT Integration).
- **Execution Time**: **0.342 seconds**.
- **Coverage**: Full coverage of Option D resistor ratios, headroom margins, conversion continuity, CRC-8 validation, and defensive payload building.

### 6.2 Static Type Checking
- **Command**: `npx --yes pyright`
- **Result**: **0 errors, 0 warnings, 0 informations**.

### 6.3 Arduino Firmware Compilation
- **Compiler**: `arduino-cli` v1.5.1
- **Target FQBN**: `esp32:esp32:esp32` (ESP32 Arduino Core v3.3.11)
- **Sketch**: `firmware/cloudguard_esp32/cloudguard_esp32.ino`
- **Exit Code**: **0 (SUCCESS)**
- **Flash Memory**: **1,036,300 bytes (79.06%)** of 1,310,720 bytes maximum.
- **Dynamic RAM**: **48,544 bytes (14.81%)** of 327,680 bytes maximum.
- **Available Headroom**: **279,136 bytes** available for stack and heap.

### 6.4 Jupyter Notebook Execution
- **Notebook**: `notebooks/07c_phase7_final_readiness.ipynb`
- **Execution**: Generated programmatically and executed top-to-bottom with 0 cell errors.

---

## 7. Master Hardware Readiness & Scope Separation Matrix

| Subsystem / Test Item | Verification Method | Status | Confirmed Evidence / Pending Requirement |
| :--- | :--- | :---: | :--- |
| **Option D Headroom (15k:10k)** | Mathematical derivation & Python unit tests | **CONFIRMED PASS** | Ratio 0.400, Multiplier 2.50x, 5.0V maps to 2.00V (+450 mV linear margin). |
| **Domain Consistency (pH)** | Two-point calibration unit tests | **CONFIRMED PASS** | Identical pH evaluated across Option A and Option D dividers. |
| **TDS Golden Vectors & Temp Comp** | Polynomial golden vectors | **CONFIRMED PASS** | 1.0V=367.47 ppm, 2.0V=879.35 ppm; 35°C compensation verified. |
| **Turbidity $C^0$ Continuity** | Derivative boundary unit tests | **CONFIRMED PASS** | Discontinuity eliminated at root (4.20025V); labeled provisional. |
| **DS18B20 Dallas CRC-8 & Startup** | 9-byte scratchpad validation | **CONFIRMED PASS** | Dallas CRC-8 verified; 85°C POR startup distinguished from genuine boiling. |
| **Defensive Payload Guards** | Schema validation & fault rejection | **CONFIRMED PASS** | Rejects -127°C, -126°C, NaNs, and negative voltages. |
| **Regression Suite** | Python `unittest discover tests` | **CONFIRMED PASS** | **55/55 unit tests passed** in 0.342s. |
| **Static Code Analysis** | Pyright type checker | **CONFIRMED PASS** | **0 errors, 0 warnings** across repository. |
| **Firmware Compilation** | Arduino CLI v1.5.1 (`esp32:esp32:esp32`) | **CONFIRMED PASS** | Compiled cleanly; Flash: 79.1% (1.03 MB), RAM: 14.8% (48.5 KB). |
| **Physical Multimeter Rail Checks** | Bench DMM DC voltage measurements | **PENDING HARDWARE** | Awaiting physical prototype bench assembly and multimeter test. |
| **Probe Chemical Calibration** | Buffer solutions (pH 7.00, pH 4.00, 1413 µS/cm) | **PENDING HARDWARE** | Glass probe hydration and physical calibration pending lab access. |
| **Formazin Turbidity Standards** | Certified NTU reference solutions | **PENDING HARDWARE** | Mathematical curve labeled provisional until reference test. |
| **ESP32 Firmware Flash & LAN Test** | Physical USB upload & Wi-Fi transmission | **PENDING HARDWARE** | Requires physical ESP32 board connected to LAN. |

---

## 8. Conclusion & Sign-Off

Phase 7.3 completes all necessary firmware, mathematical, electrical, and documentation preparations for the CloudGuard-Water embedded subsystem:
- Option D ($15\,\text{k}\Omega : 10\,\text{k}\Omega$) eliminates all risk of ADC non-linear compression and overvoltage damage.
- Calibrated ADC wording is accurate and technically sound.
- All conversion domains are verified consistent across C++, Python, and documentation.
- The pre-power checklist is electrically rigorous.
- 55 unit tests pass, Pyright reports 0 errors, and the firmware compiles with clean margins.

**Phase 7 is officially complete. We await user authorization before beginning Phase 8.**
