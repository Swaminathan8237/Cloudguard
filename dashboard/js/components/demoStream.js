/**
 * CloudGuard-Water Dashboard: Demo Telemetry Stream & Preset Scenarios
 * Module: demoStream.js
 * 
 * Provides deterministic and semi-dynamic simulated water quality telemetry
 * for testing, verification, and academic viva demonstrations.
 * All data produced by this generator is strictly identified as DEMO DATA.
 */

export const PRESET_SCENARIOS = {
  safe_potable: {
    id: "safe_potable",
    name: "Standard Potable Water",
    description: "Conforming tap/treated water well within physical drinking standards.",
    expected_class: "Safe",
    payload: {
      pH: 7.25,
      TDS_ppm: 185.0,
      Turbidity_NTU: 0.45,
      Temperature_C: 22.0
    }
  },
  high_tds_contaminated: {
    id: "high_tds_contaminated",
    name: "Contaminated High-TDS Runoff",
    description: "Saline/industrial runoff with high dissolved solids exceeding potability.",
    expected_class: "Unsafe",
    payload: {
      pH: 7.45,
      TDS_ppm: 1850.0,
      Turbidity_NTU: 1.80,
      Temperature_C: 24.5
    }
  },
  acidic_industrial: {
    id: "acidic_industrial",
    name: "Acidic Chemical Runoff",
    description: "Low pH chemical discharge severely outside safe human ingestion envelope.",
    expected_class: "Unsafe",
    payload: {
      pH: 3.85,
      TDS_ppm: 820.0,
      Turbidity_NTU: 15.50,
      Temperature_C: 21.0
    }
  },
  muddy_stormwater: {
    id: "muddy_stormwater",
    name: "Muddy Stormwater (High Turbidity)",
    description: "Post-rain stormwater carrying heavy suspended colloidal particulates.",
    expected_class: "Unsafe",
    payload: {
      pH: 6.80,
      TDS_ppm: 240.0,
      Turbidity_NTU: 55.00,
      Temperature_C: 19.5
    }
  },
  false_negative_55037: {
    id: "false_negative_55037",
    name: "Test Set Case #55037 (Model False Negative)",
    description: "Verified historical test sample labeled Unsafe but predicted Safe by Decision Tree.",
    expected_class: "Safe (Model Error)",
    is_audit_case: true,
    payload: {
      pH: 7.05,
      TDS_ppm: 192.3,
      Turbidity_NTU: 3.95,
      Temperature_C: 24.26
    }
  },
  out_of_distribution: {
    id: "out_of_distribution",
    name: "Out-of-Distribution Warning Case",
    description: "Extreme measurements exceeding canonical training ranges to trigger warnings.",
    expected_class: "Unsafe (With Warnings)",
    payload: {
      pH: 11.20,
      TDS_ppm: 3150.0,
      Turbidity_NTU: 88.50,
      Temperature_C: 36.5
    }
  }
};

/**
 * Generator producing dynamic stream fluctuations around an active baseline.
 */
export class DemoStreamManager {
  constructor() {
    this.activePresetKey = "safe_potable";
    this.sequenceNumber = 0;
  }

  setPreset(key) {
    if (PRESET_SCENARIOS[key]) {
      this.activePresetKey = key;
    }
  }

  getCurrentPreset() {
    return PRESET_SCENARIOS[this.activePresetKey];
  }

  /**
   * Produce next simulated observation with realistic sensor noise.
   */
  nextSample(addNoise = true) {
    this.sequenceNumber++;
    const base = PRESET_SCENARIOS[this.activePresetKey].payload;
    
    if (!addNoise) {
      return {
        ...base,
        timestamp: new Date().toISOString(),
        seq: this.sequenceNumber,
        is_demo: true,
        scenario_id: this.activePresetKey
      };
    }

    // Add mild Gaussian-like random jitter
    const jitter = (mag) => (Math.random() - 0.5) * 2 * mag;
    
    const noisyPH = Math.max(0.0, Math.min(14.0, base.pH + jitter(0.08)));
    const noisyTDS = Math.max(0.0, base.TDS_ppm + jitter(12.0));
    const noisyTurb = Math.max(0.0, base.Turbidity_NTU + jitter(0.15));
    const noisyTemp = Math.max(-5.0, Math.min(60.0, base.Temperature_C + jitter(0.25)));

    return {
      pH: Number(noisyPH.toFixed(2)),
      TDS_ppm: Number(noisyTDS.toFixed(1)),
      Turbidity_NTU: Number(noisyTurb.toFixed(2)),
      Temperature_C: Number(noisyTemp.toFixed(2)),
      timestamp: new Date().toISOString(),
      seq: this.sequenceNumber,
      is_demo: true,
      scenario_id: this.activePresetKey
    };
  }
}
