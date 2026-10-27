# Phase 7.3 Engineering Reference: ESP32 Calibrated ADC Accuracy, Option D Headroom & Hardware Commissioning Readiness

**Project Title**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Course**: Machine Learning for Cloud Services  
**Document**: Calibrated ADC Electrical Architecture, Option D Headroom Optimization & Hardware Safety  
**Target Hardware**: ESP32 Development Board (ESP32-WROOM-32 / DevKitC / NodeMCU-32S)  
**Toolchain**: Arduino CLI v1.5.1 / ESP32 Arduino Core v3.3.11 (`esp32:esp32:esp32`)  
**Date**: October 2026  

---

## 1. Electrical Architecture & Safety Overview

The CloudGuard-Water physical prototype interfaces four water quality probes with an ESP32 microcontroller. Because the ESP32 operates at **3.3 V CMOS logic** with sensitive internal Successive Approximation Register (SAR) Analog-to-Digital Converters (ADCs), connecting 5 V analog sensor modules directly to the ESP32 GPIOs presents a critical risk of permanent chip destruction via latch-up or gate-oxide breakdown.

This document establishes the electrical constraints, pin allocations, calibrated ADC sampling protocols, voltage divider headroom calculations, pull-up resistor specifications, and pre-power-on checklists required for safe, reproducible operation.

![CloudGuard-Water IoT Circuit Wiring Diagram](file:///d:/E0224033/ML_for_Cl_Ser/CL/LB_Proj/reports/figures/phase7_iot_wiring_diagram.png)

---

## 2. Power Rail Isolation Architecture

The prototype features **three strictly separated electrical nets**:

1. **5.0 V / VIN Net (Color: ORANGE)**:
   - Sourced directly from the USB connector via the ESP32 `VIN` pin (or external 5V regulated bench supply).
   - Powers the Optical Turbidity Sensor (TS-300B / SEN0189) which requires 5.0 V for optimal infrared emitter intensity.
   - Powers the Analog pH Sensor Module (PH-4502C / SEN0161) when running on 5V supply.
   - **CRITICAL**: No 5.0 V signal may connect directly to any ESP32 GPIO.

2. **3.3 V Net (Color: RED)**:
   - Sourced strictly from the ESP32 `3V3` pin (derived from the onboard AMS1117-3.3 low-dropout linear regulator).
   - Powers the Analog TDS Sensor Module (DFRobot SEN0244). When powered at 3.3 V, its maximum output is naturally clamped to $\le 2.3\,\text{V}$, making it 100% safe for direct ADC input.
   - Powers the DS18B20 digital temperature probe ($V_{DD}$).
   - Supplies the $4.7\,\text{k}\Omega$ pull-up resistor ($R_5$) on the DS18B20 1-Wire DATA bus.
   - **CRITICAL RULE**: The 3.3 V net connects **only** to sensor power pins, pull-up resistors, and the ESP32 `3V3` power pin. It NEVER connects to non-power GPIO pins.

3. **Common System Ground (GND Net, Color: BLACK / DARK SLATE)**:
   - Shared common ground reference linking the ESP32 `GND` pin, all 4 sensor modules, the DS18B20 probe shield/ground, and the lower terminals of both voltage divider resistors ($R_2, R_4$).
   - A single, unbroken ground plane is mandatory to prevent ground loops and measurement offsets.

---

## 3. ESP32 ADC Attenuation, Calibrated ADC Wording & Electrical Domain Audit

### 3.1 Limitations of Raw 12-Bit Linear Count Mapping
A naive linear conversion formula:
$$V_{\text{pin}} = \text{counts} \times \left(\frac{3.30\,\text{V}}{4095}\right)$$
assumes an idealized linear response across all 4096 discrete steps. However, real classic ESP32 SAR ADCs exhibit physical non-linearities:
- Dead zone near ground: Non-linear compression below $\approx 100 - 150\,\text{mV}$.
- Saturation knee: Non-linear compression above $\approx 2450 - 2600\,\text{mV}$.
- Chip-to-chip reference variation: Internal bandgap references can vary between chips from $1.0\,\text{V}$ to $1.2\,\text{V}$ ($\pm 10\%$).

### 3.2 Calibrated Millivolt Reading with `analogReadMilliVolts()`
The production firmware uses `analogReadMilliVolts()` from the installed Arduino-ESP32 core:
```cpp
uint32_t milliVolts = analogReadMilliVolts(pin);
float pinVoltage = milliVoltsToVolts(milliVolts); // milliVolts / 1000.0f
```

**Rigorous Terminology & Scope Separation**:
- **Implementation Reality**: The firmware queries `analogReadMilliVolts()` as implemented by the installed Arduino-ESP32 core library.
- **No Unverified Promises**: Documentation must NOT assert that a specific hardware eFuse calibration scheme (such as "Two Point" or "eFuse Vref") was burned into the specific target ESP32 unless the physical chip has been directly interrogated using `espefuse.py` or runtime chip registers. Different batches and silicon revisions of classic ESP32 chips feature varying factory calibration states (eFuse Two Point, eFuse Vref, or default software lookup tables).
- **Domain Definitions**:
  1. **Measurable Input Range**: $\approx 0\,\text{mV}$ to $\approx 3100\,\text{mV}$ at `ADC_11db` attenuation. Signals above $3100\,\text{mV}$ saturate the converter to full scale (4095 counts).
  2. **Effective Accuracy Range**: $\approx 150\,\text{mV}$ to $\approx 2450\,\text{mV}$ at `ADC_11db` attenuation. This is the recommended linear operating range specified by Espressif where calibration correction provides optimal fidelity.
  3. **Absolute Maximum Electrical Rating**: $3.60\,\text{V}$ (or $V_{DD} + 0.3\,\text{V}$). Voltages exceeding this threshold risk permanent silicon destruction (gate oxide breakdown and CMOS latch-up).
  4. **Multimeter-Verified Bench Accuracy**: Software eFuse calibration provides an algorithmic estimate. Absolute measurement accuracy must be verified on the physical workbench using a calibrated Digital Multimeter (DMM).

---

## 4. Hardware Sensor Variants & Maximum Output Analysis

| Sensor Subsystem | Hardware Variants | Supply ($V_{CC}$) | Nominal Aqueous Output | Fault / Rail Saturation Output | Notes & Constraints |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **pH Sensor** | PH-4502C | 5.0 V | $1.20\,\text{V} - 3.78\,\text{V}$ (pH 14 to 0) | $3.50\,\text{V}$ (LM358) or $5.00\,\text{V}$ (Rail-to-Rail) | LM358 op-amp positive saturation is $V_{CC}-1.5\text{V} \approx 3.5\text{V}$. Rail-to-rail op-amp clones can reach $5.0\text{V}$. |
| **pH Sensor (Alt)** | DFRobot SEN0161 / SEN0169 | 3.3 V or 5.0 V | $0.50\,\text{V} - 3.00\,\text{V}$ | $5.00\,\text{V}$ (if 5V $V_{CC}$) | Can operate natively at 3.3V (no divider needed if 3.3V jumper set). |
| **TDS Sensor** | DFRobot SEN0244 | 3.3 V (3V3 rail) | $0.00\,\text{V} - 2.00\,\text{V}$ | $\le 2.30\,\text{V}$ (saturated saline) | Powered at 3.3V; output physically constrained below $2.3\text{V}$. Direct connection safe. |
| **Turbidity Sensor** | TS-300B / SEN0189 | 5.0 V strictly | $2.50\,\text{V} - 4.50\,\text{V}$ (turbid to clear) | $4.80\,\text{V} - 5.00\,\text{V}$ (open circuit / bright light) | IR LED requires 5.0V for rated light output. Clear water output is $4.15 - 4.50\text{V}$. |
| **Temperature** | Dallas DS18B20 | 3.3 V (3V3 rail) | 1-Wire Digital | 3.3 V Logic | Open-drain digital output. Requires $4.7\,\text{k}\Omega$ pull-up to 3.3V rail. |

---

## 5. Voltage Divider Headroom Evaluation & Option D Primary Design

For 5.0 V powered analog sensor modules whose outputs can reach $5.0\,\text{V}$, passive resistive dividers step down the voltage prior to entering the ESP32 ADC:
$$\text{Ratio} = \frac{R_{\text{lower}}}{R_{\text{upper}} + R_{\text{lower}}}, \quad M = \frac{R_{\text{upper}} + R_{\text{lower}}}{R_{\text{lower}}}$$

### 5.1 Headroom Comparison Across Evaluated Configurations

| Configuration | $R_{\text{upper}}$ ($R_1, R_3$) | $R_{\text{lower}}$ ($R_2, R_4$) | Divider Ratio | Multiplier ($M$) | Pin Voltage at 4.5V (Clear Turbidity) | Pin Voltage at 5.0V (Fault Saturation) | Margin below 2.45V Effective Ceiling | Margin below 3.6V Abs Max Rating | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option A** | $10\,\text{k}\Omega$ | $20\,\text{k}\Omega$ | 0.6667 | $1.500\times$ | $3.000\,\text{V}$ | $3.333\,\text{V}$ | **-883 mV** (in compression) | $+267\,\text{mV}$ | Marginal. Exceeds 2.45V linear ceiling; leaves thin margin to 3.6V. |
| **Option B** | $10\,\text{k}\Omega$ | $15\,\text{k}\Omega$ | 0.6000 | $1.667\times$ | $2.700\,\text{V}$ | $3.000\,\text{V}$ | **-550 mV** (in compression) | $+600\,\text{mV}$ | Better, but still enters non-linear compression above 2.45V. |
| **Option C** | $10\,\text{k}\Omega$ | $10\,\text{k}\Omega$ | 0.5000 | $2.000\times$ | $2.250\,\text{V}$ | $2.500\,\text{V}$ | **-50 mV** (slight overflow) | $+1.100\,\text{V}$ | Substantially better; 5.0V maps to 2.50V (slightly above 2.45V linear limit). |
| **Option D (PRIMARY)** | **$15\,\text{k}\Omega$** | **$10\,\text{k}\Omega$** | **0.4000** | **$2.500\times$** | **$1.800\,\text{V}$** | **$2.000\,\text{V}$** | **+450 mV (OPTIMAL)** | **+1.600 V (OPTIMAL)** | **PRIMARY PRODUCTION CHOICE**. 5.0V maps to 2.00V, guaranteeing 450 mV margin below linear ceiling! |

### 5.2 Why Option D is the Primary Design Choice
1. **Linear Compliance**: Under Option D, a worst-case $5.00\,\text{V}$ rail saturation fault produces $2.00\,\text{V}$ at the ESP32 pin. This is $+450\,\text{mV}$ below the $2.45\,\text{V}$ effective accuracy ceiling, keeping every operating and fault condition strictly within the linear region.
2. **Safe from Electrical Damage**: $2.00\,\text{V}$ provides a $+1.60\,\text{V}$ safety margin below the $3.60\,\text{V}$ absolute maximum rating.
3. **Signal Resolution Preservation**:
   - Clear water turbidity ($4.50\,\text{V}$) maps to $1.80\,\text{V}$.
   - Opaque water turbidity ($2.50\,\text{V}$) maps to $1.00\,\text{V}$.
   - Dynamic span of $800\,\text{mV}$ across the active turbidity range provides $\approx 990$ discrete ADC counts, ensuring fine granularity.
   - Neutral pH buffer ($2.50\,\text{V}$) maps to $1.00\,\text{V}$.
   - Extreme acidic pH 0 ($3.78\,\text{V}$) maps to $1.51\,\text{V}$.
4. **Standard Component Availability**: Both $15\,\text{k}\Omega$ and $10\,\text{k}\Omega$ are standard E24 $1\%$ metal-film resistors.
5. **Firmware Consistency**: `config.h`, `config.h.example`, `sensor_conversions.h`, `src/iot_simulation.py`, and `tests/test_iot_integration.py` are fully synchronized with Option D parameters ($R_1 = 15\,\text{k}\Omega, R_2 = 10\,\text{k}\Omega, M = 2.50\times$).

---

## 6. The ESP32 ADC & Wi-Fi Restriction (ADC1 vs ADC2)

> [!CAUTION]
> **CRITICAL HARDWARE RULE: NEVER USE ADC2 PINS FOR ANALOG SENSORS.**
> The ESP32 integrates two Successive Approximation Register (SAR) ADCs:
> - **ADC1**: 8 analog channels (GPIOs 32, 33, 34, 35, 36, 37, 38, 39).
> - **ADC2**: 10 analog channels (GPIOs 0, 2, 4, 12, 13, 14, 15, 25, 26, 27).
> 
> **When the ESP32 Wi-Fi driver is active, the Wi-Fi subsystem monopolizes ADC2**. Any attempt by user firmware to call `analogRead()` or `analogReadMilliVolts()` on ADC2 pins while transmitting over Wi-Fi will return either an error code or corrupt, floating values.
> 
> **Resolution**: All 3 analog sensors (`pH`, `TDS`, `Turbidity`) are assigned **strictly to ADC1 channels**. The DS18B20 uses GPIO 4 in digital open-drain mode (not analog mode).

---

## 7. Master Verified Wiring Table (Option D Architecture)

| Component Pin | Wire Color | Connects To | Pin Function / Circuit | Electrical Constraint & Notes |
| :--- | :---: | :--- | :--- | :--- |
| **ESP32 Micro-USB** | - | Host PC / USB Power | Power In (5.0 V) | Supplies clean regulated power to board and sensors. |
| **ESP32 VIN Pin** | **Orange** | Breadboard 5.0 V Rail | Raw 5.0 V USB Bus | Powers Turbidity IR LED and pH module (if 5V required). |
| **ESP32 3V3 Pin** | **Red** | Breadboard 3.3 V Rail | Regulated 3.3 V Bus | Powers TDS module, DS18B20 VDD, and pull-up resistor. |
| **ESP32 GND Pins** | **Slate/Black** | Breadboard Common Ground | Shared Common Ground | **MANDATORY**: All modules must share a unified ground bus. |
| **pH Sensor $V_{CC}$** | Orange | 5.0 V Rail | Module Power | Standard 5V module. |
| **pH Sensor GND** | Black | Common Ground | Power Ground | Reference ground for BNC probe shield. |
| **pH Sensor AOUT** | Purple | $R_1$ ($15\,\text{k}\Omega$) input | Signal Out | Connects to upper terminal of Option D voltage divider. |
| **Divider Node (pH)** | Purple | **GPIO 34 (ADC1_CH6)** | Calibrated Analog Input | Input-only pin. $V_{\text{pin}} = V_{\text{sensor}} \times 0.400$. Operating span: $0.48\,\text{V} - 1.51\,\text{V} \le 2.00\,\text{V}$. |
| **pH Lower Resistor ($R_2$)** | - | GPIO 34 node to GND | $10\,\text{k}\Omega$ Divider Base | Connected between ADC node and Common Ground. |
| **TDS Sensor $V_{CC}$** | Red | 3.3 V Rail | Module Power | Powered at 3.3 V to constrain output below 2.3 V. |
| **TDS Sensor GND** | Black | Common Ground | Power Ground | Common ground reference. |
| **TDS Sensor AOUT** | Green | **GPIO 35 (ADC1_CH7)** | Calibrated Analog Input | Input-only pin. Safe directly on 3.3 V rail ($0 - 2.3\,\text{V}$, Multiplier $1.0\times$). |
| **Turbidity $V_{CC}$**| Orange | 5.0 V Rail | Optical LED Power | Requires 5 V for optimal infrared emitter intensity. |
| **Turbidity GND** | Black | Common Ground | Power Ground | Common ground reference. |
| **Turbidity AOUT** | Cyan | $R_3$ ($15\,\text{k}\Omega$) input | Analog Voltage Out | Maximum output reaches 4.2V–4.5V in clear water. |
| **Divider Node (Turb)**| Cyan | **GPIO 32 (ADC1_CH4)** | Scaled Calibrated Input | $V_{\text{pin}} = V_{\text{sensor}} \times 0.400$. Clear water: $1.80\,\text{V}$; Saturation: $2.00\,\text{V}$. |
| **Turb Lower Resistor ($R_4$)** | - | GPIO 32 node to GND | $10\,\text{k}\Omega$ Divider Base | Connected between ADC node and Common Ground. |
| **DS18B20 Red ($V_{DD}$)**| Red | 3.3 V Rail | Sensor Power | 3.0 V to 5.5 V compliant. |
| **DS18B20 Black (GND)**| Black | Common Ground | Sensor Ground | Waterproof probe ground. |
| **DS18B20 Yellow (DATA)**| Amber | **GPIO 4** | 1-Wire Digital Bus | Bidirectional open-drain digital bus. |
| **Pull-Up Resistor ($R_5$)**| - | 3.3 V Rail to GPIO 4 | $4.7\,\text{k}\Omega$ Pull-Up | Pulls 1-Wire bus HIGH to 3.3 V (protects GPIO 4 from 5V). |

---

## 8. Sensor Calibration Domain & Mathematical Conventions

### 8.1 pH Two-Point Linear Calibration (Sensor-Side Domain)
- **Authoritative Domain**: All calibration constants (`CALIB_PH_V_NEUTRAL = 2.50 V`, `CALIB_PH_V_ACIDIC = 3.05 V`) are defined in the **SENSOR-SIDE voltage domain**.
- **Execution Order**:
  1. Read calibrated millivolts from ADC1 pin using `analogReadMilliVolts(PIN_PH_ADC)`.
  2. Convert to Volts: $V_{\text{ADC}} = \text{milliVolts} / 1000.0f$.
  3. Reconstruct sensor-side voltage: $V_{\text{sensor}} = \text{scaleVoltageDivider}(V_{\text{ADC}}, R_1, R_2)$ with $R_1=15000, R_2=10000$ ($M = 2.50\times$).
  4. Compute physical pH: $\text{convertVoltageToPH}(V_{\text{sensor}}, V_{\text{neutral}}, V_{\text{acidic}})$.
  5. Clamped to $[0.00, 14.00]$. Non-finite inputs return `NAN` and halt transmission.

### 8.2 TDS Conversion (Direct Connection Domain)
- Powered at 3.3 V; no divider is used ($M = 1.0\times$).
- Reconstructed voltage is identical to ADC pin voltage.
- Temperature compensation adjusts electrical conductivity to $25.0^\circ\text{C}$ ($\alpha = 0.02\,/^\circ\text{C}$).
- Multiplied by $0.5$ NaCl factor and provisional $K_{\text{cell}} = 1.00$.

### 8.3 Turbidity Conversion ($C^0$ Continuous Root Normalization)
- Mathematical root at $V_{\text{root}} \approx 4.20025\,\text{V}$.
- Normalization: $V_{\text{norm}} = V_{\text{sensor}} \times \left(\frac{4.20025}{V_{\text{clear}}}\right)$ with sensor-side reconstruction $M = 2.50\times$.
- Continuous across clear water threshold; capped at $3000.0\,\text{NTU}$ at high opacity ($V_{\text{norm}} \le 2.50\,\text{V}$).
- Explicitly labeled **PROVISIONAL and UNCALIBRATED** pending certified Formazin standards.

### 8.4 DS18B20 Dallas CRC-8 & Startup Power-On State
- Upgraded driver reads complete 9-byte scratchpad and calculates Dallas/Maxim CRC-8 ($X^8 + X^5 + X^4 + 1$, reflected `0x8C`).
- **Power-On Reset State (85.0°C)**:
  When reading $85.0^\circ\text{C}$ (`0x0550`), the firmware triggers a second complete 750 ms conversion. If the second conversion still reads 85.0°C, it is accepted as genuine boiling water. If conversion was not completed or failed, it is rejected as a startup fault (`DS18B20_FAULT_RESET`).

---

## 9. Pre-Power-On Inspection & Controlled Power-Up Protocol

> [!NOTE]
> **Engineering Correction on Resistance Measurements**:
> Unpowered resistance measurements between power rails and ground can be significantly affected by unpowered semiconductor junctions (linear regulator internals, ESD diodes) and bulk electrolytic/ceramic decoupling capacitors charging from the multimeter's test current. A blanket requirement of $> 100\,\text{k}\Omega$ is electrically inappropriate.
> 
> The corrected protocol separates **power-disconnected short-circuit checks** from **controlled powered rail voltage verification**.
> 
> **CRITICAL DISCLAIMER**: Physical multimeter rail verification remains **PENDING** until the physical circuit is assembled and tested in the laboratory.

### Phase A: Power-Disconnected Inspection (Unpowered)
Before connecting USB power:
1. [ ] **Short Circuit Check**: With DMM in continuity / low-resistance mode ($200\,\Omega$ range), verify that there is **NO DIRECT SHORT** ($R < 1\,\Omega$ or continuous beep) between:
   - 5.0 V rail and Common GND
   - 3.3 V rail and Common GND
   - 5.0 V rail and 3.3 V rail
2. [ ] **Common Ground Continuity**: Verify $0.0\,\Omega$ resistance between ESP32 GND, breadboard GND rails, all sensor module GND pins, and lower divider resistors ($R_2, R_4$).
3. [ ] **Resistor Value Verification**: With DMM in resistance mode, verify:
   - Upper divider resistors ($R_1, R_3$): measured at $\approx 15.0\,\text{k}\Omega \pm 1\%$.
   - Lower divider resistors ($R_2, R_4$): measured at $\approx 10.0\,\text{k}\Omega \pm 1\%$.
   - DS18B20 pull-up resistor ($R_5$): measured at $\approx 4.7\,\text{k}\Omega \pm 5\%$.
4. [ ] **Signal Pin Disconnection**: Temporarily disconnect sensor signal wires from ESP32 pins (GPIO 32, 34, 35) prior to initial power-up.

### Phase B: Controlled Power-Up & Rail Verification (Powered)
1. [ ] **Energize ESP32**: Plug in USB cable to ESP32 board.
2. [ ] **Rail Voltage Verification with DMM (DC Volts)**:
   - Measure 5.0 V rail to GND: confirm $4.75\,\text{V} - 5.25\,\text{V}$.
   - Measure 3.3 V rail to GND: confirm $3.20\,\text{V} - 3.40\,\text{V}$.
3. [ ] **Divider Node Voltage Verification**:
   - Measure pH divider node (junction of $R_1, R_2$): confirm $V \le 2.00\,\text{V}$.
   - Measure turbidity divider node (junction of $R_3, R_4$): confirm $V \le 2.00\,\text{V}$.
   - Confirm neither node exceeds $2.45\,\text{V}$.
4. [ ] **Seat Signal Wires**: Once rail and node voltages are verified within limits, disconnect power, seat signal wires into GPIO 32, 34, 35, and re-energize.
5. [ ] **Liquid Isolation**: Maintain electronics at least 30 cm away from water beakers. Submerge probe tips only; never submerge signal conditioning boards.

---

## 10. Master Verification Matrix: Confirmed vs. Pending

| Subsystem / Test Item | Verification Method | Status | Confirmed Evidence / Pending Requirement |
| :--- | :--- | :---: | :--- |
| **Option D Headroom (15k:10k)** | Mathematical derivation & Python unit tests | **CONFIRMED PASS** | Ratio 0.400, Multiplier 2.50x, 5.0V maps to 2.00V (+450 mV linear margin). |
| **Domain Consistency (pH)** | Two-point calibration unit tests | **CONFIRMED PASS** | Identical pH evaluated across Option A and Option D dividers. |
| **TDS Golden Vectors & Temp Comp** | Polynomial golden vectors | **CONFIRMED PASS** | 1.0V=367.47 ppm, 2.0V=879.35 ppm; 35°C compensation verified. |
| **Turbidity $C^0$ Continuity** | Derivative boundary unit tests | **CONFIRMED PASS** | Discontinuity eliminated at root (4.20025V); labeled provisional. |
| **DS18B20 Dallas CRC-8 & Startup** | 9-byte scratchpad validation | **CONFIRMED PASS** | Dallas CRC-8 verified; 85°C POR startup distinguished from genuine boiling. |
| **Defensive Payload Guards** | Schema validation & fault rejection | **CONFIRMED PASS** | Rejects -127°C, -126°C, NaNs, and negative voltages. |
| **Regression Suite** | Python `unittest discover tests` | **CONFIRMED PASS** | **55/55 unit tests passed** in 0.380s. |
| **Static Code Analysis** | Pyright type checker | **CONFIRMED PASS** | **0 errors, 0 warnings** across repository. |
| **Firmware Compilation** | Arduino CLI v1.5.1 (`esp32:esp32:esp32`) | **CONFIRMED PASS** | Compiled cleanly; Flash: 79.1% (1.03 MB), RAM: 14.8% (48.5 KB). |
| **Physical Multimeter Rail Checks** | Bench DMM DC voltage measurements | **PENDING HARDWARE** | Awaiting physical prototype bench assembly and multimeter test. |
| **Probe Chemical Calibration** | Buffer solutions (pH 7.00, pH 4.00, 1413 µS/cm) | **PENDING HARDWARE** | Glass probe hydration and physical calibration pending lab access. |
| **Formazin Turbidity Standards** | Certified NTU reference solutions | **PENDING HARDWARE** | Mathematical curve labeled provisional until reference test. |
| **ESP32 Firmware Flash & LAN Test** | Physical USB upload & Wi-Fi transmission | **PENDING HARDWARE** | Requires physical ESP32 board connected to LAN. |
