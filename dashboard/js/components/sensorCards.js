/**
 * CloudGuard-Water Dashboard: Sensor Cards Component
 * Module: sensorCards.js
 * 
 * Renders and updates 4 telemetry cards: pH, TDS, Turbidity, and Temperature.
 * Features animated gauge fills, canonical training-range tracking, and WHO/EPA reference boundaries.
 */

export const SENSOR_CONFIGS = {
  pH: {
    key: "pH",
    label: "pH Level",
    unit: "",
    min: 0.0,
    max: 14.0,
    training_min: 6.09,
    training_max: 9.96,
    guideline_min: 6.5,
    guideline_max: 8.5,
    guideline_desc: "WHO Potable: 6.5 - 8.5",
    icon: "🧪"
  },
  TDS_ppm: {
    key: "TDS_ppm",
    label: "Total Dissolved Solids",
    unit: "ppm",
    min: 0.0,
    max: 3000.0,
    training_min: 28.6,
    training_max: 2806.1,
    guideline_max: 500.0,
    guideline_desc: "WHO Desirable: < 500 ppm",
    icon: "💧"
  },
  Turbidity_NTU: {
    key: "Turbidity_NTU",
    label: "Turbidity",
    unit: "NTU",
    min: 0.0,
    max: 80.0,
    training_min: 0.0,
    training_max: 72.75,
    guideline_max: 1.0,
    guideline_desc: "WHO Desirable: < 1.0 NTU",
    icon: "🌫️"
  },
  Temperature_C: {
    key: "Temperature_C",
    label: "Water Temperature",
    unit: "°C",
    min: 0.0,
    max: 45.0,
    training_min: 16.56,
    training_max: 31.51,
    guideline_min: 15.0,
    guideline_max: 25.0,
    guideline_desc: "Typical Potable: 15 - 25 °C",
    icon: "🌡️"
  }
};

/**
 * Render initial card markup into a container.
 */
export function renderSensorCards(container) {
  if (typeof container === "string") {
    container = document.getElementById(container);
  }
  if (!container) return;

  container.innerHTML = Object.values(SENSOR_CONFIGS).map(cfg => `
    <article class="card sensor-card" id="sensor-card-${cfg.key}" data-sensor="${cfg.key}">
      <header class="card-header">
        <div class="sensor-title-row">
          <span class="sensor-icon" aria-hidden="true">${cfg.icon}</span>
          <h3 class="sensor-name">${cfg.label}</h3>
        </div>
        <span class="badge badge-neutral" id="badge-${cfg.key}">Awaiting Telemetry</span>
      </header>
      
      <div class="sensor-value-row">
        <span class="sensor-value" id="val-${cfg.key}">--</span>
        <span class="sensor-unit">${cfg.unit}</span>
      </div>

      <div class="sensor-gauge-track" aria-hidden="true">
        <div class="sensor-gauge-fill" id="gauge-${cfg.key}" style="width: 0%"></div>
      </div>

      <footer class="sensor-meta">
        <div class="meta-row">
          <span class="meta-label">WHO/BIS Guideline:</span>
          <span class="meta-val">${cfg.guideline_desc}</span>
        </div>
        <div class="meta-row">
          <span class="meta-label">Training Range:</span>
          <span class="meta-val" id="training-range-${cfg.key}">[${cfg.training_min} – ${cfg.training_max} ${cfg.unit}]</span>
        </div>
      </footer>
    </article>
  `).join("");
}

/**
 * Update card values, gauges, and status badges.
 * @param {Object} readings - { pH, TDS_ppm, Turbidity_NTU, Temperature_C }
 * @param {Array<string>} warnings - List of training-range warning strings from API
 */
export function updateSensorCards(readings, warnings = []) {
  if (!readings) return;

  // Determine which features triggered warnings
  const warnedFeatures = new Set();
  warnings.forEach(w => {
    Object.keys(SENSOR_CONFIGS).forEach(key => {
      if (w.includes(`Feature '${key}'`) || w.includes(`'${key}'`)) {
        warnedFeatures.add(key);
      }
    });
  });

  Object.values(SENSOR_CONFIGS).forEach(cfg => {
    const rawVal = readings[cfg.key];
    const valEl = document.getElementById(`val-${cfg.key}`);
    const gaugeEl = document.getElementById(`gauge-${cfg.key}`);
    const badgeEl = document.getElementById(`badge-${cfg.key}`);
    const cardEl = document.getElementById(`sensor-card-${cfg.key}`);

    if (rawVal === undefined || rawVal === null || isNaN(rawVal)) {
      if (valEl) valEl.textContent = "--";
      if (gaugeEl) gaugeEl.style.width = "0%";
      if (badgeEl) {
        badgeEl.textContent = "Offline";
        badgeEl.className = "badge badge-neutral";
      }
      return;
    }

    const num = Number(rawVal);
    if (valEl) valEl.textContent = num.toFixed(cfg.key === "TDS_ppm" ? 1 : 2);

    // Calculate gauge percentage
    const span = cfg.max - cfg.min;
    const clamped = Math.max(cfg.min, Math.min(cfg.max, num));
    const pct = ((clamped - cfg.min) / span) * 100;
    if (gaugeEl) gaugeEl.style.width = `${pct.toFixed(1)}%`;

    // Determine badge and color states
    const isWarned = warnedFeatures.has(cfg.key) || num < cfg.training_min || num > cfg.training_max;
    
    if (badgeEl) {
      if (isWarned) {
        badgeEl.textContent = "Out of Training Range";
        badgeEl.className = "badge badge-warning";
        if (gaugeEl) gaugeEl.className = "sensor-gauge-fill fill-warning";
      } else {
        // Evaluate guideline compliance
        let guidelineCompliant = true;
        if (cfg.guideline_min !== undefined && num < cfg.guideline_min) guidelineCompliant = false;
        if (cfg.guideline_max !== undefined && num > cfg.guideline_max) guidelineCompliant = false;

        if (guidelineCompliant) {
          badgeEl.textContent = "Guideline Potable";
          badgeEl.className = "badge badge-safe";
          if (gaugeEl) gaugeEl.className = "sensor-gauge-fill fill-safe";
        } else {
          badgeEl.textContent = "Exceeds Standard";
          badgeEl.className = "badge badge-unsafe";
          if (gaugeEl) gaugeEl.className = "sensor-gauge-fill fill-unsafe";
        }
      }
    }
  });
}
