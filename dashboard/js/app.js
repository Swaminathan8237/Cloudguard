/**
 * CloudGuard-Water Dashboard: Main Frontend Application Controller
 * Module: app.js
 * 
 * Orchestrates telemetry modes (Demo Stream, Live Standby, Interactive What-If),
 * polling intervals, API request lifecycles, UI state transitions, and canvas charts.
 */

import { renderSensorCards, updateSensorCards } from "./components/sensorCards.js";
import { TelemetryChart } from "./components/charts.js";
import { renderExplanation, resetExplanationView } from "./components/explanation.js";
import { DemoStreamManager, PRESET_SCENARIOS } from "./components/demoStream.js";

// Determine API Base URL safely
const getApiBase = () => {
  if (window.location.protocol.startsWith("http")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
};

class DashboardApp {
  constructor() {
    this.apiBase = getApiBase();
    this.streamManager = new DemoStreamManager();
    this.chart = null;
    this.mode = "demo_stream"; // "demo_stream", "live_standby", "interactive"
    this.pollInterval = 3000;
    this.timerId = null;
    this.isPaused = false;
    this.apiHealthy = false;
    this.lastLatency = null;
    this.lastUpdateTime = null;

    this.init();
  }

  async init() {
    // 1. Initialize Sensor Cards Container
    renderSensorCards("sensor-cards-grid");

    // 2. Initialize Canvas Chart
    const canvas = document.getElementById("telemetry-chart");
    if (canvas) {
      this.chart = new TelemetryChart(canvas, 25);
    }

    // 3. Populate Scenario Dropdown
    this.populateScenarios();

    // 4. Attach UI Event Listeners
    this.attachEventListeners();

    // 5. Initial Health Check
    await this.checkApiHealth();
    setInterval(() => this.checkApiHealth(), 10000);

    // 6. Start in Demo Stream Mode
    this.setMode("demo_stream");
  }

  populateScenarios() {
    const select = document.getElementById("scenario-select");
    if (!select) return;

    select.innerHTML = Object.values(PRESET_SCENARIOS).map(sc => `
      <option value="${sc.id}">${sc.name} (${sc.expected_class})</option>
    `).join("");
  }

  attachEventListeners() {
    // Mode Switcher Tabs
    const modeTabs = document.querySelectorAll(".mode-tab");
    modeTabs.forEach(tab => {
      tab.addEventListener("click", () => {
        const targetMode = tab.dataset.mode;
        this.setMode(targetMode);
      });
    });

    // Scenario Dropdown Change
    const scenarioSelect = document.getElementById("scenario-select");
    if (scenarioSelect) {
      scenarioSelect.addEventListener("change", (e) => {
        this.streamManager.setPreset(e.target.value);
        if (this.mode === "demo_stream") {
          this.fetchNextObservation();
        }
      });
    }

    // Stream Pause / Resume Button
    const pauseBtn = document.getElementById("btn-pause-stream");
    if (pauseBtn) {
      pauseBtn.addEventListener("click", () => {
        this.isPaused = !this.isPaused;
        pauseBtn.textContent = this.isPaused ? "▶ Resume Stream" : "⏸ Pause Stream";
        pauseBtn.className = this.isPaused ? "btn btn-outline btn-paused" : "btn btn-outline";
      });
    }

    // Chart Mode Selector
    const chartModeSelect = document.getElementById("chart-metric-select");
    if (chartModeSelect && this.chart) {
      chartModeSelect.addEventListener("change", (e) => {
        this.chart.setMode(e.target.value);
      });
    }

    // Interactive Form Submit ("What-If" Analysis)
    const interactiveForm = document.getElementById("interactive-form");
    if (interactiveForm) {
      interactiveForm.addEventListener("submit", (e) => {
        e.preventDefault();
        this.runInteractiveInference();
      });
    }

    // Interactive Form Reset
    const resetBtn = document.getElementById("btn-reset-interactive");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        document.getElementById("input-ph").value = "7.20";
        document.getElementById("input-tds").value = "180.0";
        document.getElementById("input-turb").value = "0.50";
        document.getElementById("input-temp").value = "22.0";
      });
    }
  }

  setMode(mode) {
    this.mode = mode;

    // Update Tab Styles
    document.querySelectorAll(".mode-tab").forEach(tab => {
      tab.classList.toggle("active", tab.dataset.mode === mode);
    });

    // Stop existing polling
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }

    const demoBanner = document.getElementById("demo-mode-banner");
    const liveBanner = document.getElementById("live-standby-banner");
    const demoControls = document.getElementById("demo-controls-panel");
    const interactivePanel = document.getElementById("interactive-panel");

    if (mode === "demo_stream") {
      if (demoBanner) demoBanner.style.display = "flex";
      if (liveBanner) liveBanner.style.display = "none";
      if (demoControls) demoControls.style.display = "flex";
      if (interactivePanel) interactivePanel.style.display = "none";

      this.fetchNextObservation();
      this.timerId = setInterval(() => {
        if (!this.isPaused) {
          this.fetchNextObservation();
        }
      }, this.pollInterval);

    } else if (mode === "live_standby") {
      if (demoBanner) demoBanner.style.display = "none";
      if (liveBanner) liveBanner.style.display = "flex";
      if (demoControls) demoControls.style.display = "none";
      if (interactivePanel) interactivePanel.style.display = "none";

      // Standby state: clearly distinguish that live hardware is pending
      updateSensorCards(null);
      resetExplanationView("STANDBY: No physical ESP32 connected. Hardware deployment pending calibration & commissioning.");

    } else if (mode === "interactive") {
      if (demoBanner) demoBanner.style.display = "none";
      if (liveBanner) liveBanner.style.display = "none";
      if (demoControls) demoControls.style.display = "none";
      if (interactivePanel) interactivePanel.style.display = "block";
    }
  }

  async checkApiHealth() {
    const healthBadge = document.getElementById("api-health-badge");
    const latencyEl = document.getElementById("api-latency-val");
    const t0 = performance.now();

    try {
      const res = await fetch(`${this.apiBase}/health`, {
        signal: AbortSignal.timeout(4000)
      });
      const t1 = performance.now();
      this.lastLatency = Math.round(t1 - t0);

      if (res.ok) {
        const data = await res.json();
        this.apiHealthy = data.model_loaded === true;
        if (healthBadge) {
          healthBadge.textContent = "API ONLINE";
          healthBadge.className = "badge badge-safe";
        }
        if (latencyEl) latencyEl.textContent = `${this.lastLatency} ms`;
      } else {
        this.apiHealthy = false;
        if (healthBadge) {
          healthBadge.textContent = "API DEGRADED";
          healthBadge.className = "badge badge-warning";
        }
      }
    } catch (err) {
      this.apiHealthy = false;
      if (healthBadge) {
        healthBadge.textContent = "API DISCONNECTED";
        healthBadge.className = "badge badge-unsafe";
      }
      if (latencyEl) latencyEl.textContent = "-- ms";
    }
  }

  async fetchNextObservation() {
    if (!this.apiHealthy && this.mode === "demo_stream") {
      await this.checkApiHealth();
      if (!this.apiHealthy) return;
    }

    const sample = this.streamManager.nextSample(true);
    await this.processObservation(sample);
  }

  async runInteractiveInference() {
    const phVal = parseFloat(document.getElementById("input-ph")?.value);
    const tdsVal = parseFloat(document.getElementById("input-tds")?.value);
    const turbVal = parseFloat(document.getElementById("input-turb")?.value);
    const tempVal = parseFloat(document.getElementById("input-temp")?.value);

    const formErrorEl = document.getElementById("interactive-form-error");
    if (formErrorEl) formErrorEl.style.display = "none";

    // Client-side domain check
    if (isNaN(phVal) || isNaN(tdsVal) || isNaN(turbVal) || isNaN(tempVal)) {
      if (formErrorEl) {
        formErrorEl.textContent = "Validation error: All four physical parameters must be finite numbers.";
        formErrorEl.style.display = "block";
      }
      return;
    }

    const payload = {
      pH: phVal,
      TDS_ppm: tdsVal,
      Turbidity_NTU: turbVal,
      Temperature_C: tempVal,
      timestamp: new Date().toISOString(),
      is_interactive: true
    };

    await this.processObservation(payload);
  }

  async processObservation(observation) {
    const payload = {
      pH: observation.pH,
      TDS_ppm: observation.TDS_ppm,
      Turbidity_NTU: observation.Turbidity_NTU,
      Temperature_C: observation.Temperature_C
    };

    const t0 = performance.now();

    try {
      const res = await fetch(`${this.apiBase}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(5000)
      });

      const t1 = performance.now();
      this.lastLatency = Math.round(t1 - t0);
      const latencyEl = document.getElementById("api-latency-val");
      if (latencyEl) latencyEl.textContent = `${this.lastLatency} ms`;

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        this.handleApiError(res.status, errorData);
        return;
      }

      const exp = await res.json();

      // 1. Update Sensor Cards
      updateSensorCards(observation, exp.warnings || []);

      // 2. Update Explainability View
      renderExplanation(exp);

      // 3. Update Canvas Chart
      if (this.chart) {
        this.chart.pushReading({
          ...observation,
          classification: exp.model_classification
        });
      }

      // 4. Update Timestamp
      this.lastUpdateTime = new Date();
      const lastUpdateEl = document.getElementById("last-update-time");
      if (lastUpdateEl) {
        lastUpdateEl.textContent = this.lastUpdateTime.toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        });
      }

    } catch (err) {
      this.handleNetworkError(err);
    }
  }

  handleApiError(statusCode, errorData) {
    console.error(`API Error ${statusCode}:`, errorData);
    const formErrorEl = document.getElementById("interactive-form-error");
    let msg = `HTTP ${statusCode} Error: Unable to complete prediction request.`;

    if (statusCode === 422 && errorData.detail) {
      const detailStr = Array.isArray(errorData.detail) 
        ? errorData.detail.map(d => `${d.loc?.slice(-1)[0] || 'field'}: ${d.msg}`).join("; ")
        : JSON.stringify(errorData.detail);
      msg = `Pydantic Validation Error (HTTP 422): ${detailStr}`;
    } else if (statusCode === 503) {
      msg = `Service Unavailable (HTTP 503): Model artifact not loaded.`;
    }

    if (formErrorEl && this.mode === "interactive") {
      formErrorEl.textContent = msg;
      formErrorEl.style.display = "block";
    }

    resetExplanationView(msg);
  }

  handleNetworkError(err) {
    console.error("Network or Timeout Error:", err);
    resetExplanationView(`API Connectivity Error: ${err.message || 'Connection refused / timed out'}`);
  }
}

// Bootstrap application on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  window.dashboardApp = new DashboardApp();
});
