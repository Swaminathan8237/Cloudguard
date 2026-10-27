"""
Module: tests.test_iot_integration
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Unit, boundary-value, golden-vector, and integration regression tests for Phase 7.3
         calibrated ADC conversions, Option D divider headroom, conversion validations, and fault handling.
"""

import math
import unittest
from fastapi.testclient import TestClient

from api.main import app
from src.explain import FEATURE_NAMES
from src.iot_simulation import (
    adc_to_voltage,
    millivolts_to_volts,
    scale_voltage_divider,
    convert_voltage_to_ph,
    convert_voltage_to_tds,
    convert_voltage_to_turbidity,
    dallas_crc8,
    validate_ds18b20_scratchpad,
    validate_raw_sensor_signals,
    build_iot_payload,
    DS18B20_FAULT_DISCONNECTED,
    DS18B20_FAULT_CRC_ERROR,
    DS18B20_FAULT_RESET,
    TURB_MAX_NTU,
    TURB_UPPER_ROOT_V,
)


class TestIoTIntegration(unittest.TestCase):
    """Test suite for Phase 7.3 IoT calibrated sensor conversions, fault detections, and payload formatting."""

    def setUp(self):
        self.client = TestClient(app)

    # -----------------------------------------------------------------------
    # 1. ADC Conversion & Voltage Divider Tests
    # -----------------------------------------------------------------------

    def test_adc_to_voltage_discrete_bounds(self):
        """Verify ADC count to voltage conversion across discrete 12-bit boundaries (ideal model)."""
        self.assertAlmostEqual(adc_to_voltage(0), 0.0, places=3)
        self.assertAlmostEqual(adc_to_voltage(4095), 3.30, places=2)
        self.assertAlmostEqual(adc_to_voltage(2047), 1.649, places=2)

        with self.assertRaises(ValueError):
            adc_to_voltage(-1)
        with self.assertRaises(ValueError):
            adc_to_voltage(4096)

    def test_millivolts_to_volts_conversion(self):
        """Verify millivolt to Volt conversion and non-finite rejection."""
        self.assertEqual(millivolts_to_volts(0.0), 0.0)
        self.assertAlmostEqual(millivolts_to_volts(2450.0), 2.450, places=3)
        self.assertAlmostEqual(millivolts_to_volts(3300.0), 3.300, places=3)

        with self.assertRaises(ValueError):
            millivolts_to_volts(float("nan"))
        with self.assertRaises(ValueError):
            millivolts_to_volts(float("inf"))

    def test_scale_voltage_divider_ratios_and_headroom_options(self):
        """Verify voltage divider reconstruction across baseline and conservative headroom options."""
        # Baseline Option A: 10k upper (R1), 20k lower (R2) -> Multiplier 1.50x
        v_measured = 2.0
        v_sensor = scale_voltage_divider(v_measured, r1=10000.0, r2=20000.0)
        self.assertAlmostEqual(v_sensor, 3.0, places=3)

        # Option B: 10k upper, 15k lower -> Multiplier 1.6667x
        v_sensor_opt_b = scale_voltage_divider(1.8, r1=10000.0, r2=15000.0)
        self.assertAlmostEqual(v_sensor_opt_b, 3.0, places=3)

        # Option C: 10k upper, 10k lower -> Multiplier 2.0x
        v_sensor_half = scale_voltage_divider(1.5, r1=10000.0, r2=10000.0)
        self.assertAlmostEqual(v_sensor_half, 3.0, places=3)

        # Primary Option D (Phase 7.3): 15k upper (R1), 10k lower (R2) -> Divider ratio 0.40, Multiplier 2.50x
        # Maps 5.0V maximum sensor rail to 2.00V at ADC pin (450 mV headroom below 2.45V linear ceiling!)
        v_sensor_opt_d = scale_voltage_divider(2.0, r1=15000.0, r2=10000.0)
        self.assertAlmostEqual(v_sensor_opt_d, 5.0, places=3)

        # Clear water turbidity (4.50V) maps to 1.80V at ADC pin
        v_turb_clear_sensor = scale_voltage_divider(1.80, r1=15000.0, r2=10000.0)
        self.assertAlmostEqual(v_turb_clear_sensor, 4.50, places=3)

        # Neutral pH buffer (2.50V) maps to 1.00V at ADC pin
        v_ph_neutral_sensor = scale_voltage_divider(1.00, r1=15000.0, r2=10000.0)
        self.assertAlmostEqual(v_ph_neutral_sensor, 2.50, places=3)

        # Direct connection (R1 == 0.0) -> Multiplier is 1.0 (used for TDS probe)
        v_direct = scale_voltage_divider(2.25, r1=0.0, r2=10000.0)
        self.assertAlmostEqual(v_direct, 2.25, places=3)

        # Non-finite and invalid parameter handling
        with self.assertRaises(ValueError):
            scale_voltage_divider(float("nan"), r1=10000.0, r2=20000.0)
        with self.assertRaises(ValueError):
            scale_voltage_divider(2.0, r1=1000.0, r2=0.0)
        with self.assertRaises(ValueError):
            scale_voltage_divider(2.0, r1=-100.0, r2=1000.0)

    def test_adc_headroom_and_effective_accuracy_range_margins(self):
        """
        Verify Option D electrical headroom against:
        1. Effective accuracy range linear ceiling (2.450V for ESP32 at ADC_11db)
        2. Measurable input range ceiling (~3.100V)
        3. Absolute maximum electrical rating (3.600V)
        """
        v_max_sensor_rail = 5.00  # Maximum possible sensor rail fault voltage
        r1 = 15000.0  # 15 kΩ
        r2 = 10000.0  # 10 kΩ
        divider_ratio = r2 / (r1 + r2)  # 0.400
        reconstruction_multiplier = (r1 + r2) / r2  # 2.500x

        self.assertAlmostEqual(divider_ratio, 0.400, places=3)
        self.assertAlmostEqual(reconstruction_multiplier, 2.500, places=3)

        v_pin_max_fault = v_max_sensor_rail * divider_ratio  # 2.000V
        self.assertAlmostEqual(v_pin_max_fault, 2.000, places=3)

        # 1. Effective accuracy range margin (linear region <= 2.45V)
        effective_linear_ceiling = 2.450
        linear_headroom_mv = (effective_linear_ceiling - v_pin_max_fault) * 1000.0
        self.assertAlmostEqual(linear_headroom_mv, 450.0, places=1)
        self.assertGreater(linear_headroom_mv, 0.0, "Fault voltage must remain below 2.45V linear ceiling")

        # 2. Measurable range margin (up to 3.10V)
        measurable_ceiling = 3.100
        measurable_margin_mv = (measurable_ceiling - v_pin_max_fault) * 1000.0
        self.assertAlmostEqual(measurable_margin_mv, 1100.0, places=1)

        # 3. Absolute maximum rating margin (3.60V)
        abs_max_rating = 3.600
        abs_max_margin_v = abs_max_rating - v_pin_max_fault
        self.assertAlmostEqual(abs_max_margin_v, 1.600, places=3)

    # -----------------------------------------------------------------------
    # 2. pH Domain Consistency & Calibration Tests
    # -----------------------------------------------------------------------

    def test_convert_voltage_to_ph_domain_consistency(self):
        """
        Verify that 2-point linear calibration gives identical pH regardless of whether
        evaluated in sensor domain or ADC domain, provided calibration constants match the domain.
        Tested across Option A (10k:20k) and primary Option D (15k:10k).
        """
        # Sensor-side domain: Neutral = 2.50V, Acidic = 3.05V, probe voltage = 2.775V
        v_sensor_neutral = 2.50
        v_sensor_acidic = 3.05
        v_sensor_test = 2.775
        ph_sensor_domain = convert_voltage_to_ph(
            v_sensor_test, v_neutral=v_sensor_neutral, v_acidic=v_sensor_acidic
        )

        # Case 1: Scaled to ADC pin with Option A 10k:20k divider (ratio 2/3 = 0.66667)
        ratio_a = 20000.0 / (10000.0 + 20000.0)
        v_adc_neutral_a = v_sensor_neutral * ratio_a
        v_adc_acidic_a = v_sensor_acidic * ratio_a
        v_adc_test_a = v_sensor_test * ratio_a

        ph_adc_domain_a = convert_voltage_to_ph(
            v_adc_test_a, v_neutral=v_adc_neutral_a, v_acidic=v_adc_acidic_a
        )
        self.assertEqual(ph_sensor_domain, ph_adc_domain_a)

        v_reconstructed_a = scale_voltage_divider(v_adc_test_a, r1=10000.0, r2=20000.0)
        ph_reconstructed_a = convert_voltage_to_ph(
            v_reconstructed_a, v_neutral=v_sensor_neutral, v_acidic=v_sensor_acidic
        )
        self.assertEqual(ph_sensor_domain, ph_reconstructed_a)

        # Case 2: Scaled to ADC pin with Option D 15k:10k divider (ratio 0.400, multiplier 2.50x)
        ratio_d = 10000.0 / (15000.0 + 10000.0)
        v_adc_neutral_d = v_sensor_neutral * ratio_d  # 1.00V
        v_adc_acidic_d = v_sensor_acidic * ratio_d   # 1.22V
        v_adc_test_d = v_sensor_test * ratio_d       # 1.11V

        ph_adc_domain_d = convert_voltage_to_ph(
            v_adc_test_d, v_neutral=v_adc_neutral_d, v_acidic=v_adc_acidic_d
        )
        self.assertEqual(ph_sensor_domain, ph_adc_domain_d)

        v_reconstructed_d = scale_voltage_divider(v_adc_test_d, r1=15000.0, r2=10000.0)
        ph_reconstructed_d = convert_voltage_to_ph(
            v_reconstructed_d, v_neutral=v_sensor_neutral, v_acidic=v_sensor_acidic
        )
        self.assertEqual(ph_sensor_domain, ph_reconstructed_d)

    def test_convert_voltage_to_ph_bounds_and_invalid_settings(self):
        """Verify boundary clamping [0.0, 14.0] and rejection of invalid calibration settings."""
        # Extreme acidic voltage -> clamped to 0.0
        self.assertEqual(convert_voltage_to_ph(5.0, v_neutral=2.50, v_acidic=3.05), 0.0)
        # Extreme alkaline voltage -> clamped to 14.0
        self.assertEqual(convert_voltage_to_ph(1.0, v_neutral=2.50, v_acidic=3.05), 14.0)

        with self.assertRaises(ValueError):
            convert_voltage_to_ph(float("nan"))
        with self.assertRaises(ValueError):
            convert_voltage_to_ph(2.5, v_neutral=2.5, v_acidic=2.5)  # Equal voltages
        with self.assertRaises(ValueError):
            convert_voltage_to_ph(2.5, v_neutral=-1.0, v_acidic=3.05)  # Negative calibration voltage
        with self.assertRaises(ValueError):
            convert_voltage_to_ph(2.5, v_neutral=2.50, v_acidic=float("nan"))

    # -----------------------------------------------------------------------
    # 3. TDS Conversion Golden Vectors & Temperature Compensation
    # -----------------------------------------------------------------------

    def test_convert_voltage_to_tds_golden_vectors(self):
        """Verify TDS conversion against exact theoretical golden vectors."""
        # Golden Vector 1: At 25°C, V = 1.0V, K_cell = 1.0
        # EC = 133.42*(1)^3 - 255.86*(1)^2 + 857.39*(1) = 734.95 µS/cm
        # TDS = EC * 0.5 = 367.475 -> 367.47 ppm (IEEE 754 half-to-even)
        tds_1v = convert_voltage_to_tds(1.0, temperature_c=25.0, k_cell=1.0)
        self.assertAlmostEqual(tds_1v, 367.47, places=2)

        # Golden Vector 2: At 25°C, V = 2.0V, K_cell = 1.0
        # EC = 133.42*(8) - 255.86*(4) + 857.39*(2) = 1067.36 - 1023.44 + 1714.78 = 1758.70 µS/cm
        # TDS = 1758.70 * 0.5 = 879.35 ppm
        tds_2v = convert_voltage_to_tds(2.0, temperature_c=25.0, k_cell=1.0)
        self.assertAlmostEqual(tds_2v, 879.35, places=2)

        # Golden Vector 3: K_cell factor scaling (879.35 * 1.1 = 967.285 -> 967.28)
        tds_kcell = convert_voltage_to_tds(2.0, temperature_c=25.0, k_cell=1.1)
        self.assertAlmostEqual(tds_kcell, 967.28, places=2)

        # Golden Vector 4: Temperature compensation at 35°C (temp_coef = 1.0 + 0.02*(35-25) = 1.2)
        # V_comp = 1.2 / 1.2 = 1.0V -> TDS should be exactly 367.47 ppm
        tds_temp_comp = convert_voltage_to_tds(1.2, temperature_c=35.0, k_cell=1.0)
        self.assertAlmostEqual(tds_temp_comp, 367.47, places=2)

        # Zero voltage -> 0 ppm
        self.assertEqual(convert_voltage_to_tds(0.0), 0.0)

    def test_convert_voltage_to_tds_invalid_inputs(self):
        """Verify exception handling on invalid TDS voltages, temperatures, and K_cell."""
        with self.assertRaises(ValueError):
            convert_voltage_to_tds(-0.5)
        with self.assertRaises(ValueError):
            convert_voltage_to_tds(float("nan"))
        with self.assertRaises(ValueError):
            convert_voltage_to_tds(1.5, temperature_c=float("inf"))
        with self.assertRaises(ValueError):
            convert_voltage_to_tds(1.5, k_cell=0.0)
        with self.assertRaises(ValueError):
            convert_voltage_to_tds(1.5, k_cell=-1.0)

    # -----------------------------------------------------------------------
    # 4. Turbidity Continuity & Discontinuity Fix Verification
    # -----------------------------------------------------------------------

    def test_turbidity_continuity_across_threshold(self):
        """
        CRITICAL TEST: Verify that the 181.56 NTU discontinuity at V_clear is completely eliminated.
        Check delta between V_clear - 0.001 and V_clear.
        With slope dNTU/dV ≈ -3670 NTU/V, a 1 mV delta yields ~3.7 NTU (vs 181.56 NTU jump previously!).
        """
        v_clear = 4.15
        ntu_below = convert_voltage_to_turbidity(v_clear - 0.001, v_clear=v_clear)
        ntu_at = convert_voltage_to_turbidity(v_clear, v_clear=v_clear)
        ntu_above = convert_voltage_to_turbidity(v_clear + 0.001, v_clear=v_clear)

        # Both at and above v_clear must be 0.0
        self.assertEqual(ntu_at, 0.0)
        self.assertEqual(ntu_above, 0.0)

        # Delta across threshold must be bounded by the mathematical derivative (~3.7 NTU for 1 mV), NOT 181.56 NTU!
        self.assertLess(abs(ntu_below - ntu_at), 5.0)

    def test_turbidity_saturation_and_invalid_settings(self):
        """Verify turbidity saturation at complete opacity (<= 2.50V) and rejection of invalid settings."""
        # Low voltages (muddy/opaque) saturate at 3000 NTU
        self.assertEqual(convert_voltage_to_turbidity(2.50, v_clear=4.15), TURB_MAX_NTU)
        self.assertEqual(convert_voltage_to_turbidity(2.00, v_clear=4.15), TURB_MAX_NTU)
        self.assertEqual(convert_voltage_to_turbidity(0.50, v_clear=4.15), TURB_MAX_NTU)

        # Mid-range voltage gives reasonable intermediate NTU
        ntu_mid = convert_voltage_to_turbidity(3.50, v_clear=4.15)
        self.assertGreater(ntu_mid, 0.0)
        self.assertLess(ntu_mid, TURB_MAX_NTU)

        with self.assertRaises(ValueError):
            convert_voltage_to_turbidity(float("nan"))
        with self.assertRaises(ValueError):
            convert_voltage_to_turbidity(-1.0)
        with self.assertRaises(ValueError):
            convert_voltage_to_turbidity(3.0, v_clear=0.0)
        with self.assertRaises(ValueError):
            convert_voltage_to_turbidity(3.0, v_clear=-4.15)

    # -----------------------------------------------------------------------
    # 5. Dallas 1-Wire CRC-8 & DS18B20 Scratchpad Validation Tests
    # -----------------------------------------------------------------------

    def test_dallas_crc8_golden_vectors(self):
        """Verify Dallas 1-Wire CRC-8 against known Maxim/Dallas test vectors."""
        # Test Vector 1: Valid scratchpad for +25.0625°C
        # LSB=0x91, MSB=0x01, TH=0x4B, TL=0x46, Conf=0x7F, Res=0xFF, Res=0x00, Res=0x10 -> CRC=0x3D
        scratchpad_25c = bytes([0x91, 0x01, 0x4B, 0x46, 0x7F, 0xFF, 0x00, 0x10])
        computed_crc = dallas_crc8(scratchpad_25c)
        self.assertEqual(computed_crc, 0x3D)

        # Hashing all 9 bytes (including CRC) must yield 0x00
        full_scratchpad = scratchpad_25c + bytes([0x3D])
        self.assertEqual(dallas_crc8(full_scratchpad), 0x00)

        # Test Vector 2: Power-on reset default scratchpad (+85.0°C)
        por_scratchpad = bytes([0x50, 0x05, 0x4B, 0x46, 0x7F, 0xFF, 0x0C, 0x10, 0x1C])
        self.assertEqual(dallas_crc8(por_scratchpad), 0x00)

    def test_validate_ds18b20_scratchpad_scenarios(self):
        """Verify DS18B20 scratchpad validation across normal, CRC error, POR, and bus fault."""
        # 1. Valid +25.06°C reading
        valid_sp = bytes([0x91, 0x01, 0x4B, 0x46, 0x7F, 0xFF, 0x00, 0x10, 0x3D])
        is_val, temp, msg = validate_ds18b20_scratchpad(valid_sp, conversion_completed=True)
        self.assertTrue(is_val)
        self.assertAlmostEqual(temp, 25.06, places=2)
        self.assertIn("CRC verified", msg)

        # 2. Corrupted CRC (single bit flip)
        corrupted_sp = bytes([0x91, 0x01, 0x4B, 0x46, 0x7F, 0xFF, 0x00, 0x10, 0x3E])
        is_val_c, temp_c, msg_c = validate_ds18b20_scratchpad(corrupted_sp)
        self.assertFalse(is_val_c)
        self.assertEqual(temp_c, DS18B20_FAULT_CRC_ERROR)
        self.assertIn("CRC-8 mismatch", msg_c)

        # 3. Power-on reset (+85.0°C) with conversion NOT completed -> Startup fault
        por_sp = bytes([0x50, 0x05, 0x4B, 0x46, 0x7F, 0xFF, 0x0C, 0x10, 0x1C])
        is_val_p, temp_p, msg_p = validate_ds18b20_scratchpad(por_sp, conversion_completed=False)
        self.assertFalse(is_val_p)
        self.assertEqual(temp_p, DS18B20_FAULT_RESET)
        self.assertIn("power-on reset", msg_p)

        # 4. Power-on reset (+85.0°C) with conversion VERIFIED completed -> Valid genuine boiling temp
        is_val_p2, temp_p2, msg_p2 = validate_ds18b20_scratchpad(por_sp, conversion_completed=True)
        self.assertTrue(is_val_p2)
        self.assertEqual(temp_p2, 85.0)
        self.assertIn("genuine 85.0°C", msg_p2)

        # 5. Disconnected bus (all 0xFF or all 0x00)
        disc_sp = b"\xff" * 9
        is_val_d, temp_d, msg_d = validate_ds18b20_scratchpad(disc_sp)
        self.assertFalse(is_val_d)
        self.assertEqual(temp_d, DS18B20_FAULT_DISCONNECTED)

    # -----------------------------------------------------------------------
    # 6. Hardware Fault Detection & Signal Validation Tests
    # -----------------------------------------------------------------------

    def test_validate_raw_signals_healthy(self):
        """Verify healthy analog signals and digital temperature pass validation."""
        valid, faults = validate_raw_sensor_signals(v_ph=2.45, v_tds=1.20, v_turb=2.80, temp_c=22.5)
        self.assertTrue(valid)
        self.assertEqual(len(faults), 0)

    def test_validate_raw_signals_ds18b20_faults(self):
        """Verify detection of DS18B20 disconnection (-127°C), CRC failure (-126°C), and uncompleted POR (85°C)."""
        valid_disc, faults_disc = validate_raw_sensor_signals(v_ph=2.5, v_tds=1.0, v_turb=2.5, temp_c=-127.0)
        self.assertFalse(valid_disc)
        self.assertTrue(any("-127.0°C" in f for f in faults_disc))

        valid_crc, faults_crc = validate_raw_sensor_signals(v_ph=2.5, v_tds=1.0, v_turb=2.5, temp_c=-126.0)
        self.assertFalse(valid_crc)
        self.assertTrue(any("CRC-8" in f for f in faults_crc))

        # Uncompleted conversion -> treated as fault
        valid_reset, faults_reset = validate_raw_sensor_signals(
            v_ph=2.5, v_tds=1.0, v_turb=2.5, temp_c=85.0, conversion_completed=False
        )
        self.assertFalse(valid_reset)
        self.assertTrue(any("85.0°C" in f for f in faults_reset))

    def test_validate_raw_signals_nonfinite_and_out_of_bounds(self):
        """Verify detection of non-finite voltages, hazardous overvoltage (>3.3V), and negative wiring faults."""
        valid_nan, faults_nan = validate_raw_sensor_signals(v_ph=float("nan"), v_tds=1.0, v_turb=2.5, temp_c=22.0)
        self.assertFalse(valid_nan)
        self.assertTrue(any("non-finite" in f for f in faults_nan))

        valid_over, faults_over = validate_raw_sensor_signals(v_ph=4.5, v_tds=1.0, v_turb=2.5, temp_c=22.0)
        self.assertFalse(valid_over)
        self.assertTrue(any("exceeds ESP32 3.3V" in f for f in faults_over))

        valid_neg, faults_neg = validate_raw_sensor_signals(v_ph=2.5, v_tds=-0.2, v_turb=2.5, temp_c=22.0)
        self.assertFalse(valid_neg)
        self.assertTrue(any("negative voltage" in f for f in faults_neg))

    # -----------------------------------------------------------------------
    # 7. Payload Construction & Defensive Stale/Fault Prevention Tests
    # -----------------------------------------------------------------------

    def test_build_iot_payload_valid(self):
        """Verify building a valid payload matching canonical feature names."""
        payload = build_iot_payload(ph=7.35, tds_ppm=185.0, turbidity_ntu=2.1, temperature_c=22.5)
        self.assertEqual(list(payload.keys()), FEATURE_NAMES)
        self.assertEqual(payload["pH"], 7.35)
        self.assertEqual(payload["TDS_ppm"], 185.0)

    def test_build_iot_payload_rejects_fault_codes_and_stale_telemetry(self):
        """Verify failed or stale sensor values cannot silently enter prediction payload."""
        # Disconnected DS18B20 (-127°C) must be rejected
        with self.assertRaises(ValueError):
            build_iot_payload(ph=7.0, tds_ppm=200.0, turbidity_ntu=1.0, temperature_c=DS18B20_FAULT_DISCONNECTED)

        # Corrupted CRC (-126°C) must be rejected
        with self.assertRaises(ValueError):
            build_iot_payload(ph=7.0, tds_ppm=200.0, turbidity_ntu=1.0, temperature_c=DS18B20_FAULT_CRC_ERROR)

        # Non-finite values must be rejected
        with self.assertRaises(ValueError):
            build_iot_payload(ph=float("nan"), tds_ppm=200.0, turbidity_ntu=1.0, temperature_c=25.0)
        with self.assertRaises(ValueError):
            build_iot_payload(ph=7.0, tds_ppm=float("inf"), turbidity_ntu=1.0, temperature_c=25.0)

        # Unphysical out-of-scale values must be rejected
        with self.assertRaises(ValueError):
            build_iot_payload(ph=15.0, tds_ppm=100.0, turbidity_ntu=1.0, temperature_c=20.0)
        with self.assertRaises(ValueError):
            build_iot_payload(ph=7.0, tds_ppm=-20.0, turbidity_ntu=1.0, temperature_c=20.0)

    def test_iot_payload_e2e_api_dispatch(self):
        """Verify simulated ESP32 payload reaches live FastAPI /predict endpoint successfully."""
        payload = build_iot_payload(ph=7.35, tds_ppm=138.7, turbidity_ntu=0.25, temperature_c=20.25)
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["model_classification"], "Safe")
        self.assertIn("estimated_probabilities", data)
        self.assertIn("safety_notice", data)
        self.assertIn("Estimated model class probabilities from the trained Decision Tree", data["safety_notice"])


if __name__ == "__main__":
    unittest.main()
