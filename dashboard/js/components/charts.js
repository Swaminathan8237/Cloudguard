/**
 * CloudGuard-Water Dashboard: HTML5 Canvas Multi-Trace Time-Series Chart
 * Module: charts.js
 * 
 * High-performance, zero-dependency canvas renderer for rolling telemetry history.
 * Displays normalized or raw physical telemetry traces with thresholds and legends.
 */

export class TelemetryChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} maxPoints - Maximum history buffer length (default: 30)
   */
  constructor(canvas, maxPoints = 30) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.maxPoints = maxPoints;
    this.history = []; // Array of { timestamp, pH, TDS_ppm, Turbidity_NTU, Temperature_C, classification }
    this.activeMode = "normalized"; // "normalized", "pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"

    this.colors = {
      pH: "#38bdf8",            // Sky blue
      TDS_ppm: "#a78bfa",       // Purple
      Turbidity_NTU: "#fbbf24", // Amber
      Temperature_C: "#f43f5e", // Rose
      grid: "rgba(148, 163, 184, 0.12)",
      text: "#94a3b8",
      safe: "#10b981",
      unsafe: "#f43f5e"
    };

    this.scales = {
      pH: { min: 4.0, max: 12.0, unit: "" },
      TDS_ppm: { min: 0.0, max: 2500.0, unit: "ppm" },
      Turbidity_NTU: { min: 0.0, max: 60.0, unit: "NTU" },
      Temperature_C: { min: 10.0, max: 40.0, unit: "°C" }
    };

    this.handleResize = this.handleResize.bind(this);
    window.addEventListener("resize", this.handleResize);
    this.handleResize();
  }

  handleResize() {
    if (!this.canvas) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = (rect.height || 260) * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height || 260}px`;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height || 260;
    this.render();
  }

  setMode(mode) {
    this.activeMode = mode;
    this.render();
  }

  pushReading(sample) {
    if (!sample) return;
    this.history.push({
      timestamp: sample.timestamp || new Date().toISOString(),
      pH: Number(sample.pH),
      TDS_ppm: Number(sample.TDS_ppm),
      Turbidity_NTU: Number(sample.Turbidity_NTU),
      Temperature_C: Number(sample.Temperature_C),
      classification: sample.classification || "Unknown"
    });

    if (this.history.length > this.maxPoints) {
      this.history.shift();
    }

    this.render();
  }

  clear() {
    this.history = [];
    this.render();
  }

  render() {
    if (!this.ctx || !this.width || !this.height) return;
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Clear background
    ctx.clearRect(0, 0, w, h);

    // Padding
    const pad = { top: 25, right: 30, bottom: 35, left: 55 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    if (plotW <= 0 || plotH <= 0) return;

    // Draw grid lines
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const y = pad.top + (plotH / yTicks) * i;
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);

      // Y-axis labels
      ctx.fillStyle = this.colors.text;
      ctx.font = "11px Inter, system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      let label = "";
      if (this.activeMode === "normalized") {
        const pct = Math.round(100 - (100 / yTicks) * i);
        label = `${pct}%`;
      } else {
        const sc = this.scales[this.activeMode];
        const val = sc.max - ((sc.max - sc.min) / yTicks) * i;
        label = `${Math.round(val)} ${sc.unit}`;
      }
      ctx.fillText(label, pad.left - 8, y);
    }
    ctx.stroke();

    if (this.history.length < 2) {
      // Empty state watermark
      ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
      ctx.font = "13px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Waiting for telemetry observations...", pad.left + plotW / 2, pad.top + plotH / 2);
      return;
    }

    // Time points mapping
    const n = this.history.length;
    const xStep = plotW / (this.maxPoints - 1);
    const getX = (idx) => pad.left + (this.maxPoints - n + idx) * xStep;

    const keysToDraw = this.activeMode === "normalized" 
      ? ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"]
      : [this.activeMode];

    keysToDraw.forEach(key => {
      const sc = this.scales[key];
      const color = this.colors[key];

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";

      this.history.forEach((pt, idx) => {
        const raw = pt[key];
        let normY = 0;
        if (this.activeMode === "normalized") {
          normY = Math.max(0, Math.min(1, (raw - sc.min) / (sc.max - sc.min)));
        } else {
          normY = Math.max(0, Math.min(1, (raw - sc.min) / (sc.max - sc.min)));
        }
        const y = pad.top + plotH - normY * plotH;
        const x = getX(idx);

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw point markers
      this.history.forEach((pt, idx) => {
        const raw = pt[key];
        const normY = Math.max(0, Math.min(1, (raw - sc.min) / (sc.max - sc.min)));
        const y = pad.top + plotH - normY * plotH;
        const x = getX(idx);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    });

    // Time labels on bottom axis
    ctx.fillStyle = this.colors.text;
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    
    // Earliest and latest time
    if (this.history.length > 0) {
      const t0 = new Date(this.history[0].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const tEnd = new Date(this.history[this.history.length - 1].timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      ctx.fillText(t0, getX(0), pad.top + plotH + 8);
      ctx.fillText(tEnd, getX(n - 1), pad.top + plotH + 8);
    }
  }
}
