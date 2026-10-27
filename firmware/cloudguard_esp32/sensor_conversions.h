/**
 * @file sensor_conversions.h
 * @brief CloudGuard-Water Phase 7.2: Calibrated Sensor Conversion & Signal Conditioning Header
 *
 * Provides physical unit conversions, temperature compensation, voltage divider
 * scaling, and electrical fault detection for ESP32 analog and digital probes.
 */

#ifndef SENSOR_CONVERSIONS_H
#define SENSOR_CONVERSIONS_H

#if defined(ARDUINO) || (defined(__has_include) && __has_include(<Arduino.h>))
#include <Arduino.h>
#else
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <string>
#include <cstdio>
class String : public std::string {
public:
    String() : std::string() {}
    String(const char* s) : std::string(s ? s : "") {}
    String(const std::string& s) : std::string(s) {}
    String(float val, int decimals = 2) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.*f", decimals, val);
        assign(buf);
    }
    String(int val) {
        assign(std::to_string(val));
    }
};
#endif
#include <math.h>

// Plausibility & Fault Detection Boundaries
#define DS18B20_FAULT_DISCONNECTED  -127.0f
#define DS18B20_FAULT_CRC_ERROR     -126.0f
#define DS18B20_FAULT_RESET         85.0f

// Turbidity Transfer Function Constants (SEN0189 / TS-300B)
#define TURB_UPPER_ROOT_V           4.20025f // Mathematical zero-crossing of -1120.4*V^2 + 5742.3*V - 4352.9
#define TURB_SATURATION_V           2.50f    // Voltage below which optical sensor saturates at maximum opacity
#define TURB_MAX_NTU                3000.0f  // Maximum saturation ceiling in NTU

/**
 * @brief Convert raw 12-bit ADC discrete count to pin voltage using an ideal linear model.
 *
 * SIMULATION NOTE:
 * This represents an idealized uncalibrated linear transfer model (counts / 4095.0 * 3.3V).
 * For physical ESP32 hardware, the factory eFuse-calibrated analogReadMilliVolts() function
 * must be used to eliminate chip-specific non-linearity and offset errors.
 *
 * @param adcCounts Raw ADC reading [0 to 4095].
 * @param vRef Reference analog voltage (typically 3.30V).
 * @return float Calculated voltage at ESP32 pin in Volts.
 */
inline float adcToVoltage(int adcCounts, float vRef = 3.30f) {
    if (adcCounts < 0) adcCounts = 0;
    if (adcCounts > 4095) adcCounts = 4095;
    return ((float)adcCounts / 4095.0f) * vRef;
}

/**
 * @brief Convert millivolts (e.g. from analogReadMilliVolts) to Volts.
 * @param milliVolts Voltage in millivolts [0 to ~3300].
 * @return float Voltage in Volts [0.0f to 3.30f].
 */
inline float milliVoltsToVolts(uint32_t milliVolts) {
    return (float)milliVolts / 1000.0f;
}

/**
 * @brief Reconstruct original sensor output voltage across a voltage divider.
 * Formula: V_sensor = V_measured * ((R1 + R2) / R2)
 *
 * When R1 == 0.0f (direct connection), V_sensor == V_measured.
 *
 * @param vMeasured Voltage read at ADC pin across R2 in Volts.
 * @param r1 Upper resistor (Ohms).
 * @param r2 Lower resistor (Ohms).
 * @return float Original sensor output voltage in Volts, or NAN if parameters are invalid.
 */
inline float scaleVoltageDivider(float vMeasured, float r1, float r2) {
    if (isnan(vMeasured) || isinf(vMeasured)) return NAN;
    if (r2 <= 0.0f || r1 < 0.0f) return NAN;
    if (r1 == 0.0f) return vMeasured;
    float multiplier = (r1 + r2) / r2;
    return vMeasured * multiplier;
}

/**
 * @brief Convert pH probe voltage to dimensionless pH using 2-point linear calibration.
 *
 * DOMAIN ALIGNMENT RULE:
 * All voltage parameters (voltage, vNeutral, vAcidic) MUST reside in the same
 * voltage domain (authoritatively: SENSOR-SIDE voltage in Volts). Any voltage divider
 * reconstruction MUST be applied exactly once prior to calling this function.
 *
 * Formula: pH = pH_neutral + ((pH_acidic - pH_neutral) / (vAcidic - vNeutral)) * (voltage - vNeutral)
 *
 * @param voltage Measured sensor-side voltage in Volts.
 * @param vNeutral Voltage measured in pH 7.00 buffer.
 * @param vAcidic Voltage measured in pH 4.00 buffer.
 * @param phNeutral Neutral buffer value (default 7.00).
 * @param phAcidic Acidic buffer value (default 4.00).
 * @return float Computed pH value clamped to physical range [0.00 to 14.00], or NAN on error.
 */
inline float convertVoltageToPH(
    float voltage,
    float vNeutral = 2.50f,
    float vAcidic = 3.05f,
    float phNeutral = 7.00f,
    float phAcidic = 4.00f
) {
    if (isnan(voltage) || isinf(voltage)) return NAN;
    if (isnan(vNeutral) || isinf(vNeutral) || vNeutral <= 0.0f) return NAN;
    if (isnan(vAcidic) || isinf(vAcidic) || vAcidic <= 0.0f) return NAN;
    if (fabs(vAcidic - vNeutral) < 0.001f) return NAN;

    float slope = (phAcidic - phNeutral) / (vAcidic - vNeutral);
    float ph = phNeutral + slope * (voltage - vNeutral);
    if (ph < 0.0f) ph = 0.0f;
    if (ph > 14.0f) ph = 14.0f;
    return ph;
}

/**
 * @brief Convert TDS probe voltage to parts per million (ppm) with temperature compensation.
 * Target Module: DFRobot Gravity Analog TDS Sensor (SEN0244).
 *
 * Compensation: V_comp = V_raw / (1.0 + 0.02 * (T - 25.0))
 * Electrical Conductivity: EC = 133.42 * V^3 - 255.86 * V^2 + 857.39 * V
 * TDS Conversion: TDS = EC * 0.5 * K_cell (0.5 is NaCl standard conversion factor)
 *
 * PROVISIONAL NOTICE:
 * In the absence of physical calibration with standard 1413 µS/cm conductivity reference solution,
 * K_cell is maintained at 1.00 as a provisional baseline.
 *
 * @param voltage Sensor output voltage in Volts.
 * @param temperatureC Water temperature in Celsius from DS18B20.
 * @param kCell Probe cell constant calibration factor (default 1.0).
 * @return float Total Dissolved Solids in ppm, non-negative, or NAN on error.
 */
inline float convertVoltageToTDS(
    float voltage,
    float temperatureC = 25.0f,
    float kCell = 1.0f
) {
    if (isnan(voltage) || isinf(voltage) || voltage < 0.0f) return NAN;
    if (isnan(temperatureC) || isinf(temperatureC)) return NAN;
    if (isnan(kCell) || isinf(kCell) || kCell <= 0.0f) return NAN;

    float tempCoef = 1.0f + 0.02f * (temperatureC - 25.0f);
    if (tempCoef <= 0.001f) tempCoef = 0.001f;
    float vComp = voltage / tempCoef;

    float ec = 133.42f * (vComp * vComp * vComp) - 255.86f * (vComp * vComp) + 857.39f * vComp;
    float tds = ec * 0.5f * kCell;
    if (tds < 0.0f) tds = 0.0f;
    return tds;
}

/**
 * @brief Convert optical turbidity sensor voltage to NTU.
 * Target Module: DFRobot SEN0189 / TS-300B optical infrared turbidity sensor.
 *
 * Continuous root normalization eliminates the previous 181.56 NTU jump at 4.15V.
 * Mathematical root where NTU == 0 is at V = 4.20025V.
 *
 * PROVISIONAL CALIBRATION NOTICE:
 * In the absence of physical Formazin standards (0, 20, 100, 800 NTU), this NTU estimate is strictly
 * PROVISIONAL and UNCALIBRATED.
 *
 * @param voltage Sensor output voltage in Volts (sensor-side domain).
 * @param vClear Clear water reference voltage (~4.15V measured in distilled water).
 * @return float Turbidity in Nephelometric Turbidity Units (NTU) in [0.0, 3000.0], or NAN on error.
 */
inline float convertVoltageToTurbidity(float voltage, float vClear = 4.15f) {
    if (isnan(voltage) || isinf(voltage) || voltage < 0.0f) return NAN;
    if (isnan(vClear) || isinf(vClear) || vClear <= 0.0f) return NAN;

    // Normalize voltage so that vClear maps to the mathematical root
    float vNorm = voltage * (TURB_UPPER_ROOT_V / vClear);

    if (vNorm >= TURB_UPPER_ROOT_V) return 0.0f;
    if (vNorm <= TURB_SATURATION_V) return TURB_MAX_NTU;

    float ntu = -1120.4f * (vNorm * vNorm) + 5742.3f * vNorm - 4352.9f;
    if (ntu < 0.0f) ntu = 0.0f;
    if (ntu > TURB_MAX_NTU) ntu = TURB_MAX_NTU;
    return ntu;
}

/**
 * @brief Calculate Dallas / Maxim 1-Wire 8-bit Cyclic Redundancy Check (CRC-8).
 * Polynomial: X^8 + X^5 + X^4 + 1 (reflected representation: 0x8C).
 *
 * @param data Pointer to byte buffer.
 * @param len Number of bytes.
 * @return uint8_t Computed 8-bit CRC.
 */
inline uint8_t dallasCrc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t byteVal = data[i];
        for (uint8_t j = 0; j < 8; j++) {
            uint8_t mix = (crc ^ byteVal) & 0x01;
            crc >>= 1;
            if (mix) {
                crc ^= 0x8C;
            }
            byteVal >>= 1;
        }
    }
    return crc;
}

/**
 * @brief Check raw sensor electrical signals for disconnects, rail saturation, CRC errors, or noise.
 *
 * @param vPh Voltage measured at pH pin in Volts.
 * @param vTds Voltage measured at TDS pin in Volts.
 * @param vTurb Voltage measured at Turbidity pin in Volts.
 * @param tempC Temperature from DS18B20 probe in Celsius.
 * @param conversionCompleted True if DS18B20 conversion cycle was completed; False if read at startup.
 * @param outFaultMsg Output string receiving diagnostic description if fault detected.
 * @return bool True if all signals pass electrical health checks; False if fault detected.
 */
inline bool checkSignalHealth(
    float vPh,
    float vTds,
    float vTurb,
    float tempC,
    bool conversionCompleted,
    String &outFaultMsg
) {
    outFaultMsg = "";

    // 1. Non-finite signal checks
    if (isnan(vPh) || isinf(vPh)) {
        outFaultMsg = "pH channel contains non-finite reading (NaN or Inf). Check ADC acquisition.";
        return false;
    }
    if (isnan(vTds) || isinf(vTds)) {
        outFaultMsg = "TDS channel contains non-finite reading (NaN or Inf). Check ADC acquisition.";
        return false;
    }
    if (isnan(vTurb) || isinf(vTurb)) {
        outFaultMsg = "Turbidity channel contains non-finite reading (NaN or Inf). Check ADC acquisition.";
        return false;
    }
    if (isnan(tempC) || isinf(tempC)) {
        outFaultMsg = "Temperature channel contains non-finite reading (NaN or Inf). Check 1-Wire driver.";
        return false;
    }

    // 2. DS18B20 specific fault codes
    if (fabs(tempC - DS18B20_FAULT_DISCONNECTED) < 0.5f) {
        outFaultMsg = "DS18B20 disconnected (-127.0C returned). Check 4.7k pull-up and wiring.";
        return false;
    }
    if (fabs(tempC - DS18B20_FAULT_CRC_ERROR) < 0.5f) {
        outFaultMsg = "DS18B20 CRC-8 mismatch (-126.0C returned). Transmission corrupted by noise or jitter.";
        return false;
    }
    // Startup fault check: Treat 85.0C as fault ONLY if conversion was not completed
    if (fabs(tempC - DS18B20_FAULT_RESET) < 0.5f && !conversionCompleted) {
        outFaultMsg = "DS18B20 power-on reset state (85.0C): temperature conversion was not completed.";
        return false;
    }

    if (tempC < -10.0f || tempC > 100.0f) {
        outFaultMsg = "Temperature (" + String(tempC, 1) + "C) outside physical aqueous range [-10, 100]C.";
        return false;
    }

    // 3. Electrical rail checks (ESP32 ADC accepts 0.0V to 3.30V)
    if (vPh < 0.0f || vPh > 3.30f) {
        outFaultMsg = "pH ADC voltage (" + String(vPh, 2) + "V) outside permitted range [0, 3.3V].";
        return false;
    }
    if (vTds < 0.0f || vTds > 3.30f) {
        outFaultMsg = "TDS ADC voltage (" + String(vTds, 2) + "V) outside permitted range [0, 3.3V].";
        return false;
    }
    if (vTurb < 0.0f || vTurb > 3.30f) {
        outFaultMsg = "Turbidity ADC voltage (" + String(vTurb, 2) + "V) outside permitted range [0, 3.3V].";
        return false;
    }

    return true;
}

#endif // SENSOR_CONVERSIONS_H
