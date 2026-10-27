/**
 * @file cloudguard_esp32.ino
 * @brief CloudGuard-Water Phase 7.2: Production ESP32 Telemetry & Calibrated ADC Firmware
 *
 * Microcontroller firmware reading 4 physical water quality sensors (pH, TDS,
 * Turbidity, DS18B20 temperature), performing calibrated ADC sampling, signal conditioning,
 * defensive fault checks, formatting canonical JSON payloads, transmitting telemetry to the
 * CloudGuard-Water FastAPI prediction microservice over Wi-Fi, and displaying classification results.
 *
 * Target Hardware: ESP32 Development Board (NodeMCU-32S / DevKitC)
 * Development Framework: Arduino C++ / ESP32 Core (v3.x / v2.x compatible)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

#include "config.h"
#include "sensor_conversions.h"

// ===========================================================================
// Bit-Banged 1-Wire Implementation for DS18B20 (Zero-Dependency)
// ===========================================================================

static inline void oneWireLow(uint8_t pin) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
}

static inline void oneWireRelease(uint8_t pin) {
    pinMode(pin, INPUT_PULLUP);
}

static bool oneWireReset(uint8_t pin) {
    oneWireLow(pin);
    delayMicroseconds(480);
    oneWireRelease(pin);
    delayMicroseconds(70);
    int presence = digitalRead(pin);
    delayMicroseconds(410);
    return (presence == LOW);
}

static void oneWireWriteBit(uint8_t pin, uint8_t bit) {
    if (bit) {
        oneWireLow(pin);
        delayMicroseconds(6);
        oneWireRelease(pin);
        delayMicroseconds(64);
    } else {
        oneWireLow(pin);
        delayMicroseconds(60);
        oneWireRelease(pin);
        delayMicroseconds(10);
    }
}

static uint8_t oneWireReadBit(uint8_t pin) {
    oneWireLow(pin);
    delayMicroseconds(2);
    oneWireRelease(pin);
    delayMicroseconds(10);
    uint8_t bit = digitalRead(pin);
    delayMicroseconds(55);
    return bit;
}

static void oneWireWriteByte(uint8_t pin, uint8_t byteVal) {
    for (uint8_t i = 0; i < 8; i++) {
        oneWireWriteBit(pin, byteVal & 0x01);
        byteVal >>= 1;
    }
}

static uint8_t oneWireReadByte(uint8_t pin) {
    uint8_t byteVal = 0;
    for (uint8_t i = 0; i < 8; i++) {
        if (oneWireReadBit(pin)) {
            byteVal |= (1 << i);
        }
    }
    return byteVal;
}

/**
 * @brief Retrieve temperature reading from DS18B20 probe over 1-Wire with 9-byte scratchpad CRC-8 validation.
 *
 * STARTUP / POWER-ON RESET VERIFICATION:
 * When DS18B20 powers up, scratchpad temperature defaults to 0x0550 (+85.0°C).
 * If 85.0°C is read on the first attempt, a second complete conversion cycle (~750ms)
 * is triggered to verify whether the reading was an uncompleted startup reset state
 * or genuine +85.0°C boiling water.
 *
 * @param pin GPIO pin connected to DS18B20 DATA line.
 * @param outConversionCompleted Reference boolean set to true if conversion successfully completed.
 * @return float Temperature in Celsius, or error code (-127.0 disconnected, -126.0 CRC mismatch, 85.0 reset state).
 */
float readDS18B20Temperature(uint8_t pin, bool &outConversionCompleted) {
    outConversionCompleted = false;

    // 1. Reset bus & issue Skip ROM (0xCC) -> Convert T (0x44)
    if (!oneWireReset(pin)) {
        return DS18B20_FAULT_DISCONNECTED; // -127.0f: Disconnected or pull-up missing
    }
    oneWireWriteByte(pin, 0xCC); // Skip ROM
    oneWireWriteByte(pin, 0x44); // Convert T

    // Allow conversion time for 12-bit resolution (~750ms max)
    delay(750);
    outConversionCompleted = true;

    // 2. Reset bus & issue Skip ROM (0xCC) -> Read Scratchpad (0xBE)
    if (!oneWireReset(pin)) {
        outConversionCompleted = false;
        return DS18B20_FAULT_DISCONNECTED;
    }
    oneWireWriteByte(pin, 0xCC);
    oneWireWriteByte(pin, 0xBE);

    uint8_t scratchpad[9];
    for (uint8_t i = 0; i < 9; i++) {
        scratchpad[i] = oneWireReadByte(pin);
    }

    // 3. Check for bus fault (all 0xFF or all 0x00 indicates floating or shorted bus)
    bool allFF = true;
    bool all00 = true;
    for (uint8_t i = 0; i < 9; i++) {
        if (scratchpad[i] != 0xFF) allFF = false;
        if (scratchpad[i] != 0x00) all00 = false;
    }
    if (allFF || all00) {
        outConversionCompleted = false;
        return DS18B20_FAULT_DISCONNECTED;
    }

    // 4. Verify Dallas CRC-8 over all 9 bytes (must equal 0x00)
    if (dallasCrc8(scratchpad, 9) != 0) {
        outConversionCompleted = false;
        return DS18B20_FAULT_CRC_ERROR; // -126.0f: Checksum mismatch
    }

    // 5. Check for power-on reset state (raw 0x0550 = +85.0°C)
    int16_t rawTemp = (int16_t)((scratchpad[1] << 8) | scratchpad[0]);
    if (rawTemp == 0x0550) {
        // Trigger a second verified conversion to determine if genuine 85.0C or power-on state
        if (!oneWireReset(pin)) {
            outConversionCompleted = false;
            return DS18B20_FAULT_RESET;
        }
        oneWireWriteByte(pin, 0xCC);
        oneWireWriteByte(pin, 0x44);
        delay(750);

        if (!oneWireReset(pin)) {
            outConversionCompleted = false;
            return DS18B20_FAULT_RESET;
        }
        oneWireWriteByte(pin, 0xCC);
        oneWireWriteByte(pin, 0xBE);
        for (uint8_t i = 0; i < 9; i++) {
            scratchpad[i] = oneWireReadByte(pin);
        }
        if (dallasCrc8(scratchpad, 9) != 0) {
            outConversionCompleted = false;
            return DS18B20_FAULT_CRC_ERROR;
        }
        rawTemp = (int16_t)((scratchpad[1] << 8) | scratchpad[0]);
        if (rawTemp == 0x0550) {
            outConversionCompleted = true; // Genuine 85.0C confirmed after 2 full cycles
        } else {
            outConversionCompleted = true; // Resolved to true ambient temperature
        }
    }

    return (float)rawTemp / 16.0f;
}

// ===========================================================================
// Global State & Timing Variables
// ===========================================================================

unsigned long lastTelemetryTime = 0;
unsigned long telemetryCycleCount = 0;
int consecutiveSensorErrors = 0;

// ===========================================================================
// Helper: Wi-Fi Reconnection Logic
// ===========================================================================

void connectToWiFi() {
    Serial.println();
    Serial.print("[WiFi] Connecting to SSID: ");
    Serial.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 25) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("[WiFi] SUCCESS: Connected. Local IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("[WiFi] RSSI Signal Strength: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
    } else {
        Serial.println("[WiFi] WARNING: Connection timeout. Will retry during telemetry loop.");
    }
}

// ===========================================================================
// Helper: Filtered Calibrated Analog ADC Reading
// ===========================================================================

/**
 * @brief Acquire calibrated analog voltage at an ESP32 ADC pin.
 *
 * Uses analogReadMilliVolts() to query eFuse factory calibration (Two Point / Vref),
 * eliminating the non-linearity of raw 12-bit integer count mappings.
 *
 * @param pin ADC1 GPIO pin (e.g. GPIO 32, 34, 35).
 * @param sampleCount Number of samples to average for noise reduction.
 * @param delayMs Delay between consecutive samples in milliseconds.
 * @return float Calibrated voltage at the pin in Volts [0.00f to ~3.30f].
 */
float readAverageADCVoltage(uint8_t pin, int sampleCount, int delayMs) {
    uint32_t sumMilliVolts = 0;
    for (int i = 0; i < sampleCount; i++) {
        sumMilliVolts += analogReadMilliVolts(pin);
        if (delayMs > 0) delay(delayMs);
    }
    float avgMilliVolts = (float)sumMilliVolts / (float)sampleCount;
    return milliVoltsToVolts((uint32_t)avgMilliVolts);
}

// ===========================================================================
// Arduino Setup
// ===========================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("==================================================================");
    Serial.println("  CloudGuard-Water: Intelligent Water Quality Classification");
    Serial.println("  Phase 7.2: Calibrated ESP32 Telemetry & ADC Verification Client");
    Serial.println("==================================================================");

    // Configure ADC attenuation (11dB provides ~150mV - 2450mV linear, up to ~3100mV calibrated)
    analogSetAttenuation(ADC_11db);

    // Configure analog input pins (strictly ADC1 channels)
    pinMode(PIN_PH_ADC, INPUT);
    pinMode(PIN_TDS_ADC, INPUT);
    pinMode(PIN_TURB_ADC, INPUT);

    // Initialize 1-Wire pin for DS18B20
    oneWireRelease(PIN_DS18B20);

    Serial.println("[Hardware] Configured ADC1 pins: pH=34, TDS=35, Turb=32, DS18B20=4.");
    Serial.println("[Calibration] analogReadMilliVolts() factory eFuse calibration ACTIVE.");
    Serial.println("[Security] Defensive safety mode ACTIVE. Automatic physical actuation DISABLED.");

    // Connect to local Wi-Fi
    connectToWiFi();
}

// ===========================================================================
// Arduino Main Loop
// ===========================================================================

void loop() {
    unsigned long currentMillis = millis();

    // Check transmission timer
    if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL_MS || lastTelemetryTime == 0) {
        lastTelemetryTime = currentMillis;
        telemetryCycleCount++;

        Serial.println();
        Serial.println("------------------------------------------------------------------");
        Serial.print("[Cycle #");
        Serial.print(telemetryCycleCount);
        Serial.println("] Sampling physical sensors (eFuse calibrated ADC)...");

        // 1. Read water temperature from DS18B20 probe
        bool conversionCompleted = false;
        float tempC = readDS18B20Temperature(PIN_DS18B20, conversionCompleted);

        // 2. Read averaged, eFuse-calibrated analog voltages on ADC1 pins
        float vPhPin = readAverageADCVoltage(PIN_PH_ADC, SAMPLE_AVERAGE_COUNT, SAMPLE_DELAY_MS);
        float vTdsPin = readAverageADCVoltage(PIN_TDS_ADC, SAMPLE_AVERAGE_COUNT, SAMPLE_DELAY_MS);
        float vTurbPin = readAverageADCVoltage(PIN_TURB_ADC, SAMPLE_AVERAGE_COUNT, SAMPLE_DELAY_MS);

        // 3. Reconstruct original sensor voltages across voltage dividers
        // Note: TDS probe is powered at 3.3V and connected directly (multiplier = 1.0)
        float vPhSensor = scaleVoltageDivider(vPhPin, PH_DIVIDER_R1, PH_DIVIDER_R2);
        float vTdsSensor = vTdsPin; // Direct 3.3V connection, no divider reconstruction
        float vTurbSensor = scaleVoltageDivider(vTurbPin, TURB_DIVIDER_R1, TURB_DIVIDER_R2);

        Serial.print("  [Calibrated Pins] ADC Voltages: pH=");
        Serial.print(vPhPin, 3);
        Serial.print("V, TDS=");
        Serial.print(vTdsPin, 3);
        Serial.print("V, Turb=");
        Serial.print(vTurbPin, 3);
        Serial.print("V, Temp=");
        Serial.print(tempC, 2);
        Serial.println(" C");

        // 4. Validate electrical signal integrity and non-finite guards
        String faultMsg;
        bool isHealthy = checkSignalHealth(vPhPin, vTdsPin, vTurbPin, tempC, conversionCompleted, faultMsg);
        if (!isHealthy) {
            consecutiveSensorErrors++;
            Serial.print("  [FAULT DETECTED] ");
            Serial.println(faultMsg);
            if (consecutiveSensorErrors >= 3) {
                Serial.println("  [STALE / FAULT WARNING] Sensor bus failure sustained across >= 3 cycles.");
            }
            Serial.println("  [ABORT] Telemetry transmission skipped due to sensor fault. Stale data rejected.");
            return;
        }

        // 5. Convert voltages to canonical physical units
        float physPH = convertVoltageToPH(vPhSensor, CALIB_PH_V_NEUTRAL, CALIB_PH_V_ACIDIC);
        float physTDS = convertVoltageToTDS(vTdsSensor, tempC, CALIB_TDS_K_CELL);
        float physTurb = convertVoltageToTurbidity(vTurbSensor, CALIB_TURB_V_CLEAR);

        // Defensive check: Ensure failed, non-finite, or stale calculations never enter payload
        if (isnan(physPH) || isnan(physTDS) || isnan(physTurb) || isnan(tempC)) {
            consecutiveSensorErrors++;
            Serial.println("  [ABORT] Physical unit conversion produced NaN or invalid output. Telemetry aborted.");
            return;
        }

        // Reset error tracking on successful validated reading
        consecutiveSensorErrors = 0;

        Serial.print("  [Physical Units] pH=");
        Serial.print(physPH, 2);
        Serial.print(", TDS=");
        Serial.print(physTDS, 1);
        Serial.print(" ppm, Turbidity=");
        Serial.print(physTurb, 2);
        Serial.print(" NTU, Temp=");
        Serial.print(tempC, 2);
        Serial.println(" C");

        // 6. Ensure Wi-Fi connection is active
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("  [WiFi] Not connected. Attempting reconnection...");
            connectToWiFi();
            if (WiFi.status() != WL_CONNECTED) {
                Serial.println("  [WiFi ERROR] Reconnection failed. Aborting transmission.");
                return;
            }
        }

        // 7. Format JSON payload matching FastAPI schema
        String jsonPayload = "{";
        jsonPayload += "\"pH\":" + String(physPH, 2) + ",";
        jsonPayload += "\"TDS_ppm\":" + String(physTDS, 2) + ",";
        jsonPayload += "\"Turbidity_NTU\":" + String(physTurb, 2) + ",";
        jsonPayload += "\"Temperature_C\":" + String(tempC, 2);
        jsonPayload += "}";

        Serial.print("  [HTTP POST] Target: ");
        Serial.println(API_PREDICT_URL);
        Serial.print("  [JSON Payload] ");
        Serial.println(jsonPayload);

        // 8. Transmit HTTP POST request to FastAPI microservice
        HTTPClient http;
        http.begin(API_PREDICT_URL);
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(HTTP_TIMEOUT_MS);

        int httpResponseCode = http.POST(jsonPayload);

        if (httpResponseCode > 0) {
            Serial.print("  [HTTP STATUS] ");
            Serial.println(httpResponseCode);

            String responseString = http.getString();
            Serial.print("  [API Response] ");
            Serial.println(responseString);

            // Parse response fields cleanly
            int classIdx = responseString.indexOf("\"model_classification\":\"");
            if (classIdx != -1) {
                int endClass = responseString.indexOf("\"", classIdx + 24);
                String classification = responseString.substring(classIdx + 24, endClass);

                Serial.println();
                Serial.println("  ========================================================");
                Serial.print("  >>> PREDICTED WATER STATUS: [ ");
                Serial.print(classification);
                Serial.println(" ] <<<");
                Serial.println("  ========================================================");
            }

            // Extract estimated probabilities
            int safeIdx = responseString.indexOf("\"Safe\":");
            int unsafeIdx = responseString.indexOf("\"Unsafe\":");
            if (safeIdx != -1 && unsafeIdx != -1) {
                int endSafe = responseString.indexOf(",", safeIdx + 7);
                int endUnsafe = responseString.indexOf("}", unsafeIdx + 9);
                String pSafe = responseString.substring(safeIdx + 7, endSafe);
                String pUnsafe = responseString.substring(unsafeIdx + 9, endUnsafe);

                Serial.print("  Estimated Probabilities: P(Safe)=");
                Serial.print(pSafe);
                Serial.print(", P(Unsafe)=");
                Serial.println(pUnsafe);
            }

            // Check if per-feature warnings were generated
            int warnIdx = responseString.indexOf("\"warnings\":[");
            if (warnIdx != -1) {
                int endWarn = responseString.indexOf("]", warnIdx + 11);
                String warnContent = responseString.substring(warnIdx + 11, endWarn);
                if (warnContent.length() > 2) {
                    Serial.print("  [DATA WARNINGS] ");
                    Serial.println(warnContent);
                } else {
                    Serial.println("  [DATA WARNINGS] None (inputs within observed training bounds).");
                }
            }

            Serial.println("  [NOTICE] Prediction is statistical model output, not potable certification.");
        } else {
            Serial.print("  [HTTP ERROR] POST failed, error code: ");
            Serial.print(httpResponseCode);
            Serial.print(" (");
            Serial.print(http.errorToString(httpResponseCode));
            Serial.println(")");
            Serial.println("  [DIAGNOSTIC] Check host IP, Uvicorn binding (--host 0.0.0.0), and firewall.");
        }

        http.end();
    }

    // Short loop yield to allow ESP32 background Wi-Fi tasks to run smoothly
    delay(50);
}
