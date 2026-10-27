"""
Script: src/create_phase7_notebook.py
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Build and execute notebooks/07_phase7_iot_integration.ipynb for Phase 7,
         covering physical IoT circuit design, ESP32 firmware architecture,
         sensor conversion mathematics, network configuration, and API integration.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_phase7_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }
    cells = []

    # -----------------------------------------------------------------------
    # Section 1: Title & Academic Context
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""# CloudGuard-Water: Phase 7 — Physical IoT Circuit, ESP32 Firmware & Live Sensor Integration

**Course**: Machine Learning for Cloud Services  
**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Notebook**: `07_phase7_iot_integration.ipynb`  
**Author**: Lead Embedded & IoT Integration Specialist  
**Canonical Dataset**: `data/water_quality_dataset1.csv`  
**Target Model**: `models/decision_tree.joblib`  
**Date**: October 2026  

---

## 1. Phase Objectives & Architecture

Phase 7 bridges the gap between our cloud-ready FastAPI microservice and the physical world. The goal is to construct a physical IoT sensor prototype where:
1. Four water quality sensors produce physical readings:
   - **pH probe**: Analog electrochemical potential.
   - **TDS probe**: Analog electrical conductivity (ppm).
   - **Turbidity probe**: Analog optical transmission/scattering (NTU).
   - **DS18B20 probe**: Digital 1-Wire water temperature ($^\circ\text{C}$).
2. An **ESP32 microcontroller** samples the sensors, filters noise, executes fault checks, converts raw signals to physical units, serializes a JSON payload, and transmits telemetry over local Wi-Fi.
3. The **FastAPI prediction microservice** receives the payload at `POST /predict`, validates constraints, executes the trained Decision Tree, and returns a statistical potability classification with estimated class probabilities.
4. Serial diagnostics display predictions and defensive safety disclaimers.

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  pH Sensor      │       │  TDS Sensor     │       │ Turbidity Sensor│       │ DS18B20 Temp    │
│  Analog (ADC1)  │       │  Analog (ADC1)  │       │  Analog (ADC1)  │       │ Digital 1-Wire  │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │                         │
         └─────────────────┬───────┴─────────────────────────┴─────────────────────────┘
                           │ Physical Probes
                           ▼
                 ┌───────────────────┐
                 │  ESP32 Controller │
                 │  - ADC1 Sampling  │
                 │  - Level Shifting │
                 │  - Fault Checks   │
                 │  - Unit Conv.     │
                 │  - Wi-Fi HTTP     │
                 └─────────┬─────────┘
                           │ HTTP POST (JSON Telemetry)
                           │ Target: http://192.168.0.106:8000/predict
                           ▼
                 ┌───────────────────┐
                 │ FastAPI Service   │
                 │ - Input Validation│
                 │ - Range Warnings  │
                 │ - Decision Tree ML│
                 │ - Safety Notice   │
                 └───────────────────┘
```
"""))

    # -----------------------------------------------------------------------
    # Section 2: Environment Setup & Project Path Configuration
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Environment Setup & Project Path Configuration

We configure project-relative imports to verify Python conversion functions and local API connectivity.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

# Configure project root in path
root_dir = Path("..").resolve() if Path("..").resolve().name == "LB_Proj" else Path(".").resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.explain import FEATURE_NAMES, CLASS_LABELS, load_model
from api.main import app, WaterQualityInput, check_training_range_warnings, SAFETY_NOTICE
from src.iot_simulation import (
    adc_to_voltage,
    scale_voltage_divider,
    convert_voltage_to_ph,
    convert_voltage_to_tds,
    convert_voltage_to_turbidity,
    validate_raw_sensor_signals,
    build_iot_payload,
)
from fastapi.testclient import TestClient

print(f"Project root resolved to: {root_dir}")
print(f"Canonical features expected by ML API: {FEATURE_NAMES}")
print(f"Target classes: {CLASS_LABELS}")
"""))

    # -----------------------------------------------------------------------
    # Section 3: Hardware Inventory & User Confirmation Checklist
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Hardware Inventory & User Confirmation Checklist

### 3.1 Hardware Components Status
Following our repository inspection, the project uses standard educational IoT prototyping modules. However, the physical hardware is currently running detached from this host machine. In strict compliance with project rules:
- **No hardware models or electrical ratings are silently fabricated.**
- **Provisional pinouts and transfer curves are modular and configurable in `firmware/cloudguard_esp32/config.h`.**
- **Sensor conversions are explicitly marked as provisional until physical reference buffer calibration is executed.**

### 3.2 Hardware Questionnaire for User Confirmation
| Item | Required Specification | Typical Default Used in Firmware |
| :--- | :--- | :--- |
| **ESP32 Board** | Exact pin count & module name | ESP32-WROOM-32 (30-pin NodeMCU-32S) |
| **pH Module** | Module part number & supply voltage | DFRobot SEN0161 / PH-4502C (5.0V VCC) |
| **TDS Module** | Sensor model & probe cell constant | Gravity SEN0244 ($K_{\\text{cell}} = 1.0$) |
| **Turbidity Module** | Optical sensor model & output switch | SEN0189 / TS-300B (Set to 'A' Analog) |
| **Level Shifter** | Precision resistors for 5V analog divider | $R_1 = 10\\,\\text{k}\\Omega, R_2 = 20\\,\\text{k}\\Omega$ ($1.5\\times$ scaling) |
| **Pull-Up Resistor** | 1-Wire bus resistor | $4.7\\,\\text{k}\\Omega$ metal film between DATA and 3.3V |
| **Calibration Buffers**| Standard reference buffers available | pH 4.00, pH 7.00, and 1413 µS/cm NaCl solution |
"""))

    # -----------------------------------------------------------------------
    # Section 4: Electrical Safety & The ADC1 vs ADC2 Conflict
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Electrical Safety & The ADC1 vs ADC2 Conflict

### 4.1 The Fundamental ESP32 Hardware Rule
The ESP32 features two internal SAR ADCs:
- **ADC1**: Channels attached to GPIO 32, 33, 34, 35, 36 (VP), 39 (VN).
- **ADC2**: Channels attached to GPIO 0, 2, 4, 12, 13, 14, 15, 25, 26, 27.

> **CRITICAL HARDWARE CONSTRAINT**:
> **When the ESP32 Wi-Fi driver is initialized, ADC2 is reserved exclusively by the radio subsystem.** Any attempt to read an analog sensor on an ADC2 pin will fail or return corrupt data.
>
> **Solution**: All 3 analog sensors (`pH`, `TDS`, `Turbidity`) are strictly wired to **ADC1 channels**:
> - `pH`: **GPIO 34 (ADC1_CH6)** (Input-only)
> - `TDS`: **GPIO 35 (ADC1_CH7)** (Input-only)
> - `Turbidity`: **GPIO 32 (ADC1_CH4)** (Input/Output capable)

### 4.2 Overvoltage Protection & Voltage Dividers
- Turbidity modules powered from 5V produce up to $4.5\\text{V}$ in clear water.
- Direct connection to an ESP32 ADC pin ($3.3\\text{V}$ absolute max) would destroy the chip.
- We implement a passive $10\\,\\text{k}\\Omega : 20\\,\\text{k}\\Omega$ divider:
  $$V_{\\text{ADC}} = V_{\\text{sensor}} \\times \\frac{20}{10 + 20} = V_{\\text{sensor}} \\times \\frac{2}{3}$$
- Max ADC voltage is $4.50\\text{V} \\times \\frac{2}{3} = 3.00\\text{V}$ (well within safe 3.3V rating).
"""))

    cells.append(nbf.v4.new_code_cell("""# Verification of Voltage Divider Mathematics
v_sensor_clean = 4.50  # Sensor output in clear water
v_adc = v_sensor_clean * (20000.0 / (10000.0 + 20000.0))
print(f"Turbidity 5V output ({v_sensor_clean}V) scaled to ESP32 pin: {v_adc:.2f}V (Safe <= 3.30V)")
assert v_adc <= 3.30, "Hazard: ADC voltage exceeds 3.3V!"

# Reconstructed sensor voltage
v_reconstructed = scale_voltage_divider(v_adc, r1=10000.0, r2=20000.0)
print(f"Reconstructed sensor voltage: {v_reconstructed:.2f}V")
assert abs(v_reconstructed - v_sensor_clean) < 1e-4
"""))

    # -----------------------------------------------------------------------
    # Section 5: Sensor Conversion & Calibration Mathematics
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 5. Sensor Conversion & Calibration Mathematics

The prediction API expects physical measurements matching the canonical dataset:
- `pH`: dimensionless scale $[0.0, 14.0]$
- `TDS_ppm`: parts per million ($\ge 0.0$)
- `Turbidity_NTU`: Nephelometric Turbidity Units ($\ge 0.0$)
- `Temperature_C`: water temperature in $^\circ\text{C}$

Raw ADC counts or uncalibrated voltages must **never** be transmitted directly to the API.

### 5.1 pH Conversion (Two-Point Linear Calibration)
$$pH = pH_{\\text{neutral}} + \\left(\\frac{pH_{\\text{acidic}} - pH_{\\text{neutral}}}{V_{\\text{acidic}} - V_{\\text{neutral}}}\\right) \\times (V - V_{\\text{neutral}})$$

### 5.2 TDS Conversion with Temperature Compensation
$$V_{25} = \\frac{V_{\\text{raw}}}{1.0 + 0.02 \\times (T - 25.0)}$$
$$\\text{TDS} = (133.42 \\times V_{25}^3 - 255.86 \\times V_{25}^2 + 857.39 \\times V_{25}) \\times 0.5 \\times K_{\\text{cell}}$$

### 5.3 Turbidity Parabolic Optical Response
$$\\text{Turbidity (NTU)} = -1120.4 \\times V^2 + 5742.3 \\times V - 4352.9 \\quad (\\text{for } V < V_{\\text{clear}})$$
"""))

    cells.append(nbf.v4.new_code_cell("""# Test pH Conversion across calibration range
ph_7 = convert_voltage_to_ph(2.50, v_neutral=2.50, v_acidic=3.05)
ph_4 = convert_voltage_to_ph(3.05, v_neutral=2.50, v_acidic=3.05)
ph_8 = convert_voltage_to_ph(2.3167, v_neutral=2.50, v_acidic=3.05)
print(f"pH at 2.50V (Neutral): {ph_7}")
print(f"pH at 3.05V (Acidic):  {ph_4}")
print(f"pH at 2.32V (Basic):   {ph_8}")

# Test TDS Conversion with Temperature Compensation
tds_cold = convert_voltage_to_tds(1.20, temperature_c=18.0)
tds_standard = convert_voltage_to_tds(1.20, temperature_c=25.0)
tds_warm = convert_voltage_to_tds(1.20, temperature_c=32.0)
print(f"\\nTDS at 1.20V (18°C): {tds_cold} ppm")
print(f"TDS at 1.20V (25°C): {tds_standard} ppm")
print(f"TDS at 1.20V (32°C): {tds_warm} ppm")

# Test Turbidity Conversion
turb_clear = convert_voltage_to_turbidity(4.20, v_clear=4.15)
turb_murky = convert_voltage_to_turbidity(3.00, v_clear=4.15)
turb_dirty = convert_voltage_to_turbidity(2.00, v_clear=4.15)
print(f"\\nTurbidity at 4.20V (Clear): {turb_clear} NTU")
print(f"Turbidity at 3.00V (Murky): {turb_murky} NTU")
print(f"Turbidity at 2.00V (Dirty): {turb_dirty} NTU")
"""))

    # -----------------------------------------------------------------------
    # Section 6: Electrical Fault Detection & Signal Validation
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Electrical Fault Detection & Signal Validation

Before physical telemetry is dispatched to the API, the firmware verifies raw signals against electrical boundaries:
- **DS18B20 Disconnection**: Probe returns `-127.0°C` if the 1-Wire bus is broken or pull-up is missing.
- **DS18B20 Incomplete Conversion**: Probe returns `85.0°C` (power-on reset value).
- **Rail Saturation**: ADC voltages $>3.30\text{V}$ indicate dangerous overvoltage.
- **Ground Faults**: Negative voltages indicate wiring polarity inversion.
"""))

    cells.append(nbf.v4.new_code_cell("""test_cases = [
    ("Healthy Telemetry", 2.45, 1.10, 3.10, 23.5),
    ("DS18B20 Disconnected", 2.45, 1.10, 3.10, -127.0),
    ("DS18B20 Power-on Reset", 2.45, 1.10, 3.10, 85.0),
    ("ADC Overvoltage", 4.50, 1.10, 3.10, 23.5),
    ("Ground Inversion", 2.45, -0.25, 3.10, 23.5),
]

for label, ph_v, tds_v, turb_v, temp in test_cases:
    valid, faults = validate_raw_sensor_signals(ph_v, tds_v, turb_v, temp)
    status_str = "[PASS] Valid" if valid else f"[ABORT] Faults: {faults}"
    print(f"Case: '{label}' -> {status_str}")
"""))

    # -----------------------------------------------------------------------
    # Section 7: Network Configuration & LAN Microservice Binding
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Network Configuration & LAN Microservice Binding

### 7.1 Loopback (`127.0.0.1`) vs Local Area Network (`0.0.0.0`)
In Phase 6, Uvicorn was launched on `127.0.0.1` (local loopback), which permits requests exclusively from processes on the same computer.

For the ESP32 to transmit telemetry over Wi-Fi, the microservice must bind to all local network interfaces:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 7.2 Host LAN IP Discovery
The host computer's active IPv4 address on the Wi-Fi adapter is:
- **Discovered IP**: `192.168.0.106`
- **ESP32 Target URL**: `http://192.168.0.106:8000/predict`
- **ESP32 Health URL**: `http://192.168.0.106:8000/health`
"""))

    cells.append(nbf.v4.new_code_cell("""# Display current host network configuration
import socket

hostname = socket.gethostname()
local_ips = socket.gethostbyname_ex(hostname)[2]
print(f"Host Machine Name: {hostname}")
print(f"Active Local IP Addresses: {local_ips}")
"""))

    # -----------------------------------------------------------------------
    # Section 8: Live In-Process Telemetry Integration Test
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Live In-Process Telemetry Integration Test

We simulate the exact ESP32 workflow:
1. Sample physical sensor voltages.
2. Validate signal integrity.
3. Convert to physical units.
4. Serialize canonical JSON payload.
5. Dispatch to `POST /predict`.
6. Inspect classification, estimated probabilities, and safety warnings.
"""))

    cells.append(nbf.v4.new_code_cell("""client = TestClient(app)

# 1. Simulate potable water sample telemetry
payload_potable = build_iot_payload(ph=7.35, tds_ppm=138.7, turbidity_ntu=0.25, temperature_c=20.25)
print(f"Simulated ESP32 JSON Payload (Potable Sample):")
print(json.dumps(payload_potable, indent=2))

resp_potable = client.post("/predict", json=payload_potable)
assert resp_potable.status_code == 200
data_potable = resp_potable.json()
print("\\nFastAPI Response to ESP32:")
print(f"  Classification:          {data_potable['model_classification']}")
print(f"  Estimated Probabilities: {data_potable['estimated_probabilities']}")
print(f"  Range Warnings:          {data_potable['warnings']}")
print(f"  Safety Notice:           {data_potable['safety_notice'][:120]}...")

# 2. Simulate contaminated water sample telemetry
payload_contaminated = build_iot_payload(ph=6.76, tds_ppm=513.0, turbidity_ntu=1.48, temperature_c=22.13)
resp_contaminated = client.post("/predict", json=payload_contaminated)
assert resp_contaminated.status_code == 200
data_contaminated = resp_contaminated.json()
print("\\nFastAPI Response to ESP32 (Contaminated Sample):")
print(f"  Classification:          {data_contaminated['model_classification']}")
print(f"  Estimated Probabilities: {data_contaminated['estimated_probabilities']}")

# 3. Simulate out-of-range sensor reading (e.g. acidic pH 5.50)
payload_out_of_range = build_iot_payload(ph=5.50, tds_ppm=250.0, turbidity_ntu=2.0, temperature_c=22.0)
resp_oor = client.post("/predict", json=payload_out_of_range)
assert resp_oor.status_code == 200
data_oor = resp_oor.json()
print("\\nFastAPI Response to ESP32 (Out-of-Range Sample):")
print(f"  Classification:          {data_oor['model_classification']}")
print(f"  Range Warnings Count:    {len(data_oor['warnings'])}")
print(f"  Warning Detail:          {data_oor['warnings'][0]}")
"""))

    # -----------------------------------------------------------------------
    # Section 9: Full Regression Suite Verification
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 9. Full Regression Suite Verification

We execute all automated tests across the repository, including our new Phase 7 IoT test suite (`tests/test_iot_integration.py`).
"""))

    cells.append(nbf.v4.new_code_cell("""import unittest

loader = unittest.TestLoader()
suite = loader.discover(start_dir=str(root_dir / "tests"), pattern="test_*.py")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

print("\\n" + "=" * 60)
print(f"Total Tests Run: {result.testsRun}")
print(f"Failures: {len(result.failures)}")
print(f"Errors: {len(result.errors)}")
print(f"Was Successful: {result.wasSuccessful()}")
print("=" * 60)

assert result.wasSuccessful(), "Test regression failures detected!"
assert result.testsRun == 48, f"Expected 48 tests, got {result.testsRun}"
print("Verification SUCCESS: All 48 tests passed cleanly (20 API, 17 Explain, 11 IoT).")
"""))

    # -----------------------------------------------------------------------
    # Section 10: Summary, Limitations & Phase 8 Roadmap
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 10. Summary, Limitations & Phase 8 Roadmap

### 10.1 Key Accomplishments in Phase 7
1. **Verified Electrical Architecture**:
   - Mapped all 3 analog probes strictly to **ADC1 channels** (GPIO 34, 35, 32) to resolve the ESP32 Wi-Fi ADC conflict.
   - Designed a safe $10\\,\\text{k}\\Omega : 20\\,\\text{k}\\Omega$ voltage divider stepping down 5V turbidity outputs to $\\le 3.0\\text{V}$.
2. **Production ESP32 Firmware**:
   - Implemented `cloudguard_esp32.ino`, `sensor_conversions.h`, and `config.h.example`.
   - Built-in zero-dependency bit-banged 1-Wire driver for DS18B20.
   - Non-blocking timing, signal validation, noise averaging, and error handling.
3. **Signal Conditioning & Conversion**:
   - Python simulation module `src/iot_simulation.py` with linear pH calibration, polynomial TDS with temperature compensation, optical turbidity, and fault detection.
4. **Expanded Automated Testing**:
   - Expanded regression suite to **48 passing tests** (11 new Phase 7 IoT tests).
   - Pyright static analysis confirmed with 0 errors, 0 warnings.

### 10.2 Technical Limitations & Academic Disclaimers
- **Hardware Detached in This Session**: The physical ESP32 and probes were not connected via serial COM port during this test run. All software conversions, payload schemas, and API responses were verified via `TestClient`.
- **Provisional Calibrations**: The calibration constants in `config.h` are provisional defaults. Actual physical buffer calibration must be performed when hardware is assembled.
- **Defensive Safety**: The system provides statistical ML screening and does not replace certified laboratory testing or actuate physical valves.

### 10.3 Phase 8 Roadmap (Dashboard & End-to-End System Integration)
- Connect live ESP32 serial telemetry stream to cloud database/storage.
- Implement real-time monitoring dashboard with live sensor gauge charts, potability classification badges, probability meters, and decision path explanations.
"""))

    nb.cells = cells
    return nb


def main():
    print("Building Phase 7 IoT integration notebook...")
    nb = build_phase7_notebook()
    target_nb_path = NOTEBOOKS_DIR / "07_phase7_iot_integration.ipynb"

    print("Executing notebook top-to-bottom via NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    executed_nb = client.execute()

    print(f"Writing executed notebook to: {target_nb_path}")
    with open(target_nb_path, "w", encoding="utf-8") as f:
        nbf.write(executed_nb, f)

    print("SUCCESS: notebooks/07_phase7_iot_integration.ipynb successfully created and executed!")


if __name__ == "__main__":
    main()
