"""
Module: src.iot_simulation
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Calibrated sensor conversion mathematics, signal conditioning, hardware fault detection,
         and simulated ESP32 payload construction for Phase 7.2 IoT circuit integration.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import httpx

from src.explain import FEATURE_NAMES

# Canonical ADC Characteristics (ESP32 12-bit ADC with 11dB attenuation)
ADC_RESOLUTION: int = 4096       # 12-bit ADC: 0 to 4095
V_REF: float = 3.30              # ESP32 3.3V operating voltage

# Plausibility & Fault Detection Boundaries
DS18B20_FAULT_DISCONNECTED: float = -127.0
DS18B20_FAULT_CRC_ERROR: float = -126.0
DS18B20_FAULT_RESET: float = 85.0

# Turbidity Transfer Function Constants (SEN0189 / TS-300B)
TURB_UPPER_ROOT_V: float = 4.20025  # Mathematical zero-crossing of -1120.4*V^2 + 5742.3*V - 4352.9
TURB_SATURATION_V: float = 2.50     # Voltage below which optical sensor saturates at maximum opacity
TURB_MAX_NTU: float = 3000.0        # Saturation ceiling in NTU


def adc_to_voltage(adc_counts: int, v_ref: float = V_REF, resolution: int = ADC_RESOLUTION) -> float:
    """
    IDEALIZED LINEAR SIMULATION MODEL (UNCALIBRATED):
    Convert raw 12-bit ADC integer counts to analog voltage at the ESP32 pin using a linear assumption.

    NOTE ON PHYSICAL HARDWARE:
        Classic ESP32 SAR ADCs exhibit non-linearity below ~150 mV and above ~2450 mV when operating
        at 11 dB attenuation. In production C++ firmware, the eFuse-calibrated function
        `analogReadMilliVolts(pin)` is used instead of raw count scaling to account for factory-burned
        calibration offsets (Two Point / eFuse Vref). This Python function serves as an idealized
        simulation model for algorithmic testing and vector verification.

    Args:
        adc_counts: Integer ADC count in [0, resolution - 1].
        v_ref: Reference analog voltage (typically 3.30V on ESP32).
        resolution: ADC discrete steps (4096 for 12-bit).

    Returns:
        float: Calculated analog voltage at the ESP32 pin in Volts.
    """
    if adc_counts < 0 or adc_counts >= resolution:
        raise ValueError(f"ADC count {adc_counts} outside valid 12-bit range [0, {resolution - 1}].")
    return float(adc_counts) * (v_ref / float(resolution - 1))


def millivolts_to_volts(millivolts: float) -> float:
    """
    Convert calibrated millivolts (e.g. from ESP32 analogReadMilliVolts) to Volts.

    Args:
        millivolts: Analog voltage in millivolts.

    Returns:
        float: Analog voltage in Volts.
    """
    if math.isnan(millivolts) or math.isinf(millivolts):
        raise ValueError("Millivolt reading must be a finite number.")
    return float(millivolts) / 1000.0


def scale_voltage_divider(v_measured: float, r1: float, r2: float) -> float:
    """
    Reverse a passive voltage divider transformation to reconstruct original sensor output voltage.
    Formula: V_sensor = V_measured * ((R1 + R2) / R2)

    When R1 == 0.0 (direct connection, no divider), V_sensor == V_measured.

    Args:
        v_measured: Voltage read at the ESP32 ADC pin (across R2) in Volts.
        r1: Upper resistor in Ohms (between sensor output and ADC pin).
        r2: Lower resistor in Ohms (between ADC pin and GND).

    Returns:
        float: Reconstructed original sensor output voltage in Volts.
    """
    if math.isnan(v_measured) or math.isinf(v_measured):
        raise ValueError("Measured voltage must be a finite number.")
    if r2 <= 0.0 or r1 < 0.0:
        raise ValueError("Resistor values must be non-negative, and R2 must be positive.")
    if r1 == 0.0:
        return float(v_measured)
    multiplier = (r1 + r2) / r2
    return float(v_measured) * multiplier


def convert_voltage_to_ph(
    voltage: float,
    v_neutral: float = 2.50,
    v_acidic: float = 3.05,
    ph_neutral: float = 7.00,
    ph_acidic: float = 4.00,
) -> float:
    """
    Convert analog sensor voltage to dimensionless pH using a standard 2-point linear calibration.

    DOMAIN ALIGNMENT RULE:
        All voltage parameters (`voltage`, `v_neutral`, `v_acidic`) MUST reside in the
        same voltage domain. By project specification, calibration constants are defined in the
        SENSOR-SIDE voltage domain. Any voltage divider scaling MUST be applied exactly once
        prior to calling this function.

    Formula:
        slope = (pH_acidic - pH_neutral) / (V_acidic - V_neutral)
        pH = pH_neutral + slope * (voltage - V_neutral)

    Args:
        voltage: Sensor output voltage in Volts (sensor-side domain).
        v_neutral: Measured sensor-side voltage in standard pH 7.00 buffer solution.
        v_acidic: Measured sensor-side voltage in standard pH 4.00 buffer solution.
        ph_neutral: Neutral reference pH (7.00).
        ph_acidic: Acidic reference pH (4.00).

    Returns:
        float: Computed pH value clamped to physical aqueous range [0.00, 14.00].
    """
    if math.isnan(voltage) or math.isinf(voltage):
        raise ValueError("Voltage must be a finite floating-point number.")
    if math.isnan(v_neutral) or math.isinf(v_neutral) or v_neutral <= 0.0:
        raise ValueError("Neutral calibration voltage must be a positive finite number.")
    if math.isnan(v_acidic) or math.isinf(v_acidic) or v_acidic <= 0.0:
        raise ValueError("Acidic calibration voltage must be a positive finite number.")
    if abs(v_neutral - v_acidic) < 1e-4:
        raise ValueError("Neutral and Acidic calibration voltages must be distinct.")

    slope = (ph_acidic - ph_neutral) / (v_acidic - v_neutral)
    ph = ph_neutral + slope * (voltage - v_neutral)
    clamped_ph = max(0.0, min(14.0, float(ph)))
    return round(clamped_ph, 2)


def convert_voltage_to_tds(
    voltage: float,
    temperature_c: float = 25.0,
    k_cell: float = 1.0,
) -> float:
    """
    Convert analog probe voltage to Total Dissolved Solids (ppm) with temperature compensation.
    Target Module: DFRobot Gravity Analog TDS Sensor (SEN0244).

    Derivation & Physics:
        1. Temperature compensation adjusts measured electrical conductivity back to 25°C:
           V_comp = V_sensor / (1.0 + 0.02 * (T - 25.0))
           where 0.02 / °C is the standard aqueous conductivity temperature coefficient.
        2. Empirical conductivity polynomial (yielding EC in µS/cm):
           EC = 133.42 * V_comp^3 - 255.86 * V_comp^2 + 857.39 * V_comp
        3. TDS conversion uses the standard 0.5 factor for NaCl solutions (TDS_ppm = EC * 0.5):
           TDS = EC * 0.5 * K_cell
           where K_cell is the dimensionless calibration constant (default 1.0).

    PROVISIONAL NOTICE:
        In the absence of physical calibration against certified 1413 µS/cm reference standard,
        K_cell = 1.0 is maintained as a provisional baseline factor.

    Args:
        voltage: Sensor output voltage in Volts (0.0 to ~2.3V for SEN0244 at 3.3V VCC).
        temperature_c: Water temperature in Celsius (from DS18B20 probe).
        k_cell: Dimensionless cell constant calibration factor (default 1.0).

    Returns:
        float: Total Dissolved Solids in parts per million (ppm), non-negative.
    """
    if math.isnan(voltage) or math.isinf(voltage) or voltage < 0.0:
        raise ValueError("Voltage must be a finite, non-negative number.")
    if math.isnan(temperature_c) or math.isinf(temperature_c):
        raise ValueError("Temperature must be a finite number.")
    if math.isnan(k_cell) or math.isinf(k_cell) or k_cell <= 0.0:
        raise ValueError("Cell constant K_cell must be a positive finite number.")

    temp_coef = 1.0 + 0.02 * (temperature_c - 25.0)
    if temp_coef <= 0.001:
        temp_coef = 0.001
    v_comp = voltage / temp_coef

    ec = 133.42 * (v_comp ** 3) - 255.86 * (v_comp ** 2) + 857.39 * v_comp
    tds = ec * 0.5 * k_cell
    return max(0.0, round(float(tds), 2))


def convert_voltage_to_turbidity(
    voltage: float,
    v_clear: float = 4.15,
) -> float:
    """
    Convert optical turbidity sensor voltage to Nephelometric Turbidity Units (NTU).
    Target Module: DFRobot SEN0189 / TS-300B optical infrared turbidity sensor.

    CRITICAL DISCONTINUITY ELIMINATION & ROOT ANALYSIS:
        The empirical transfer polynomial is:
            NTU(V) = -1120.4 * V^2 + 5742.3 * V - 4352.9
        The mathematical upper root where NTU == 0 is:
            V_root = (-5742.3 - sqrt(5742.3^2 - 4*(-1120.4)*(-4352.9))) / (2 * -1120.4)
                   ≈ 4.20025 V
        Arbitrarily clamping at V >= 4.15V previously caused an artificial 181.56 NTU discontinuity.
        To maintain C^0 continuity across all calibration points, voltage is normalized:
            V_norm = voltage * (TURB_UPPER_ROOT_V / v_clear)

        Transfer piecewise curve:
            1. Clear water: if V_norm >= TURB_UPPER_ROOT_V: return 0.0 NTU (continuous at root)
            2. High opacity saturation: if V_norm <= TURB_SATURATION_V (2.50V): return 3000.0 NTU
            3. Dynamic range: for 2.50V < V_norm < 4.20025V:
               NTU = -1120.4 * V_norm^2 + 5742.3 * V_norm - 4352.9

    PROVISIONAL CALIBRATION NOTICE:
        In the absence of physical Formazin turbidity calibration standards (0, 20, 100, 800 NTU),
        this estimate is strictly PROVISIONAL and UNCALIBRATED.

    Args:
        voltage: Sensor output voltage in Volts (sensor-side domain, reconstructed across divider).
        v_clear: Reference sensor-side voltage measured in distilled clear water.

    Returns:
        float: Estimated turbidity in NTU clamped to [0.0, 3000.0], C^0 continuous.
    """
    if math.isnan(voltage) or math.isinf(voltage) or voltage < 0.0:
        raise ValueError("Voltage must be a finite, non-negative number.")
    if math.isnan(v_clear) or math.isinf(v_clear) or v_clear <= 0.0:
        raise ValueError("Clear water reference voltage must be positive and finite.")

    # Normalize voltage against mathematical root so that V == v_clear maps exactly to root
    v_norm = voltage * (TURB_UPPER_ROOT_V / v_clear)

    if v_norm >= TURB_UPPER_ROOT_V:
        return 0.0
    if v_norm <= TURB_SATURATION_V:
        return TURB_MAX_NTU

    ntu = -1120.4 * (v_norm ** 2) + 5742.3 * v_norm - 4352.9
    clamped_ntu = max(0.0, min(TURB_MAX_NTU, float(ntu)))
    return round(clamped_ntu, 2)


def dallas_crc8(data: bytes) -> int:
    """
    Calculate Dallas / Maxim 1-Wire 8-bit Cyclic Redundancy Check (CRC-8).
    Polynomial: X^8 + X^5 + X^4 + 1 (reflected representation: 0x8C).

    Args:
        data: Sequence of bytes to hash.

    Returns:
        int: Computed 8-bit CRC checksum in [0, 255].
             If computed over the full 9-byte DS18B20 scratchpad, result is 0 on success.
    """
    crc = 0
    for byte in data:
        for _ in range(8):
            mix = (crc ^ byte) & 0x01
            crc >>= 1
            if mix:
                crc ^= 0x8C
            byte >>= 1
    return crc


def validate_ds18b20_scratchpad(
    scratchpad: bytes,
    conversion_completed: bool = True,
) -> Tuple[bool, float, str]:
    """
    Validate complete 9-byte DS18B20 scratchpad data against Dallas 1-Wire CRC-8,
    bus disconnection patterns (all 0xFF or 0x00), and uncompleted power-on reset state (85.0°C).

    Scratchpad structure:
        Byte 0: Temperature LSB
        Byte 1: Temperature MSB
        Byte 2: High alarm trigger (TH)
        Byte 3: Low alarm trigger (TL)
        Byte 4: Configuration register
        Bytes 5-7: Reserved
        Byte 8: Dallas CRC-8 over bytes 0-7

    Args:
        scratchpad: 9-byte raw scratchpad buffer.
        conversion_completed: True if temperature conversion completed; False if read during startup.

    Returns:
        Tuple of (is_valid: bool, temperature_c: float, diagnostic_message: str).
    """
    if len(scratchpad) != 9:
        return False, DS18B20_FAULT_DISCONNECTED, f"Invalid scratchpad length {len(scratchpad)} (expected 9 bytes)."

    # 1. Check for missing sensor or bus fault (pull-up held high -> all 0xFF, short to ground -> all 0x00)
    if scratchpad == b"\xff" * 9 or scratchpad == b"\x00" * 9:
        return False, DS18B20_FAULT_DISCONNECTED, "1-Wire bus fault: scratchpad all 0xFF or 0x00 (sensor disconnected or floating)."

    # 2. Check Dallas CRC-8 over all 9 bytes (must equal 0x00)
    if dallas_crc8(scratchpad) != 0:
        return False, DS18B20_FAULT_CRC_ERROR, "DS18B20 CRC-8 mismatch: data corrupted during 1-Wire transmission."

    # 3. Decode 16-bit two's complement temperature
    raw_temp = int.from_bytes(scratchpad[0:2], byteorder="little", signed=True)

    # 4. Check for power-on reset state (raw 0x0550 = +85.0°C)
    if raw_temp == 0x0550:
        if not conversion_completed:
            return False, DS18B20_FAULT_RESET, "DS18B20 power-on reset state (85.0°C): temperature conversion was not completed."
        # If conversion was verified completed twice, 85.0C is treated as genuine boiling temperature
        return True, 85.0, "DS18B20 reading valid (genuine 85.0°C confirmed after verified conversion)."

    temp_c = float(raw_temp) / 16.0
    if temp_c < -10.0 or temp_c > 100.0:
        return False, temp_c, f"Temperature {temp_c:.2f}°C violates aqueous physical bounds [-10.0, 100.0]°C."

    return True, round(temp_c, 2), "DS18B20 reading valid and CRC verified."


def validate_raw_sensor_signals(
    v_ph: float,
    v_tds: float,
    v_turb: float,
    temp_c: float,
    conversion_completed: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Inspect raw electrical and digital signals from physical sensors for hardware faults,
    disconnections, rail saturation, floating pins, or CRC failures prior to physical unit conversion.

    Args:
        v_ph: Analog voltage at pH channel in Volts (measured at ESP32 pin).
        v_tds: Analog voltage at TDS channel in Volts (measured at ESP32 pin).
        v_turb: Analog voltage at Turbidity channel in Volts (measured at ESP32 pin).
        temp_c: Digital temperature in Celsius from DS18B20.
        conversion_completed: Boolean indicating whether DS18B20 conversion completed.

    Returns:
        Tuple of (is_valid: bool, fault_messages: List[str]).
    """
    faults: List[str] = []

    # 1. DS18B20 Digital Sensor Faults
    if abs(temp_c - DS18B20_FAULT_DISCONNECTED) < 0.5:
        faults.append(
            "DS18B20 hardware fault: sensor returned -127.0°C (1-Wire bus disconnected or pull-up missing)."
        )
    elif abs(temp_c - DS18B20_FAULT_CRC_ERROR) < 0.5:
        faults.append(
            "DS18B20 communication fault: sensor returned -126.0°C (CRC-8 transmission checksum mismatch)."
        )
    elif abs(temp_c - DS18B20_FAULT_RESET) < 0.5 and not conversion_completed:
        faults.append(
            "DS18B20 startup fault: sensor returned 85.0°C (power-on reset value, conversion uncompleted)."
        )
    elif temp_c < -10.0 or temp_c > 100.0:
        faults.append(f"Temperature reading {temp_c}°C violates aqueous physical bounds [-10.0, 100.0]°C.")

    # 2. Electrical Rail Saturation & Floating Checks (ESP32 3.3V ADC)
    for name, v in [("pH", v_ph), ("TDS", v_tds), ("Turbidity", v_turb)]:
        if math.isnan(v) or math.isinf(v):
            faults.append(f"Channel '{name}' contains non-finite voltage reading ({v}).")
        elif v < 0.0:
            faults.append(f"Channel '{name}' has negative voltage ({v:.3f}V), indicating wiring ground fault.")
        elif v > 3.30:
            faults.append(
                f"Channel '{name}' voltage ({v:.3f}V) exceeds ESP32 3.3V maximum rating (hazard: overvoltage)."
            )

    is_valid = (len(faults) == 0)
    return is_valid, faults


def build_iot_payload(
    ph: float,
    tds_ppm: float,
    turbidity_ntu: float,
    temperature_c: float,
) -> Dict[str, float]:
    """
    Construct a validated JSON dictionary matching the canonical 4-feature API schema.
    Defensively rejects failed, stale, or fault-code measurements so they cannot silently
    enter the prediction microservice.

    Args:
        ph: Dimensionless pH reading.
        tds_ppm: Total Dissolved Solids in ppm.
        turbidity_ntu: Turbidity in NTU.
        temperature_c: Water temperature in Celsius.

    Returns:
        Dict[str, float]: Formatted payload ready for POST /predict.

    Raises:
        ValueError: If any measurement is null, NaN, infinite, or physically invalid.
    """
    values = [ph, tds_ppm, turbidity_ntu, temperature_c]
    for name, val in zip(FEATURE_NAMES, values):
        if val is None or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Feature '{name}' contains invalid non-finite value: {val}")

    # Check for known hardware fault codes entering payload
    for fault_code, fault_name in [
        (DS18B20_FAULT_DISCONNECTED, "disconnected (-127°C)"),
        (DS18B20_FAULT_CRC_ERROR, "CRC error (-126°C)"),
    ]:
        if abs(temperature_c - fault_code) < 0.5:
            raise ValueError(f"Temperature contains hardware fault code {fault_name}; cannot send stale telemetry.")

    if ph < 0.0 or ph > 14.0:
        raise ValueError(f"pH {ph} violates physical scale [0.0, 14.0].")
    if tds_ppm < 0.0:
        raise ValueError(f"TDS {tds_ppm} cannot be negative.")
    if turbidity_ntu < 0.0:
        raise ValueError(f"Turbidity {turbidity_ntu} cannot be negative.")
    if temperature_c < -10.0 or temperature_c > 100.0:
        raise ValueError(f"Temperature {temperature_c}°C violates physical domain [-10.0, 100.0].")

    return {
        "pH": round(float(ph), 2),
        "TDS_ppm": round(float(tds_ppm), 2),
        "Turbidity_NTU": round(float(turbidity_ntu), 2),
        "Temperature_C": round(float(temperature_c), 2),
    }


def transmit_payload_to_api(
    payload: Dict[str, float],
    api_url: str = "http://127.0.0.1:8000/predict",
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Transmit the sensor payload to the FastAPI microservice and parse the response.

    Args:
        payload: 4-feature dictionary generated by build_iot_payload.
        api_url: Target REST endpoint URL (e.g. http://192.168.0.106:8000/predict).
        timeout_sec: Network timeout in seconds.

    Returns:
        Dict[str, Any]: Parsed API response.

    Raises:
        httpx.HTTPError: On communication failure or non-200 HTTP status.
    """
    headers = {"Content-Type": "application/json"}
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
