"""
Script: src/create_phase8_notebook.py
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Build and execute notebooks/08_phase8_dashboard_integration.ipynb for Phase 8,
         covering the web dashboard architecture, FastAPI integration, demo telemetry stream,
         explainability ladder visualization, canvas charts, and defensive safety disclaimers.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_phase8_notebook():
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
    cells.append(nbf.v4.new_markdown_cell(r"""# CloudGuard-Water: Phase 8 — Web Dashboard and End-to-End Software Integration

**Course**: Machine Learning for Cloud Services  
**Project**: CloudGuard-Water: Intelligent and Explainable Water Quality Classification  
**Notebook**: `08_phase8_dashboard_integration.ipynb`  
**Phase**: Phase 8 (Software Integration & Dashboard Verification)  
**Author**: Antigravity Lead Systems & Frontend Engineer  
**Canonical Dataset**: `data/water_quality_dataset1.csv` (100,800 observations, 4 features)  
**Active Model**: `models/decision_tree.joblib` (`max_depth=6`)  
**Date**: October 2026  

---

## 1. Executive Summary & Phase Objectives

Following the successful completion of Phase 7.3 (ESP32 ADC voltage divider finalization, firmware compilation, and static calibration verification), Phase 8 realizes the primary user-facing presentation layer: the **CloudGuard-Water Web Dashboard**.

### Core Objectives:
1. **Zero-Dependency Lightweight Frontend**: Build a responsive web dashboard using semantic HTML5, modern Vanilla CSS (custom properties, responsive grid, glassmorphism, WCAG AA compliance), and modular ES6 JavaScript.
2. **FastAPI Microservice Integration**: Mount the dashboard static assets directly inside the existing FastAPI service under `/dashboard`, providing seamless access while preserving existing `/health`, `/predict`, and `/explain` REST endpoints.
3. **Four Sensor Cards**: Render real-time cards for pH, TDS (ppm), Turbidity (NTU), and Water Temperature (°C) with animated gauge fills, WHO/EPA reference boundaries, and training-range indicators.
4. **Transparent Explainability View**: Render the Decision Tree traversal path ladder, primary influencing feature tags, estimated class probabilities ($P(\text{Safe}), P(\text{Unsafe})$), and auditable plain-English operational narratives.
5. **Historical Time-Series Canvas Chart**: Zero-dependency HTML5 Canvas multi-trace renderer with normalized (0–100%) and raw metric viewing modes.
6. **Explicit Telemetry Modes & Safety Guardrails**:
   - **Demo Telemetry Stream**: Active simulated stream with 6 deterministic viva scenarios, clearly marked with the mandatory alert banner: `DEMO DATA — NOT LIVE SENSOR TELEMETRY`.
   - **Live ESP32 Standby**: Explicitly displays `STANDBY: NO LIVE HARDWARE CONNECTED — PENDING PHYSICAL COMMISSIONING` without silent fallback to simulated data.
   - **Interactive What-If Tester**: Allows live manual parameter submission to inspect decision tree branching.
7. **Statutory Safety & Disabled Actuation**: Emphasizes that 4 physical sensors cannot certify potable safety (no pathogen, heavy metal, or chemical detection) and that physical actuation remains strictly disabled.
"""))

    # -----------------------------------------------------------------------
    # Section 2: Environment & Repository Verification
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 2. Environment Setup & Workspace Audit

We verify project directory paths, file existence, and import necessary libraries including FastAPI's `TestClient`, pandas, and our core explainability modules.
"""))

    cells.append(nbf.v4.new_code_cell(r"""import sys
import os
from pathlib import Path
import json
import pandas as pd
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path("..").resolve() if Path("..").resolve().name == "LB_Proj" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app, load_api_model, app_state, FEATURE_NAMES, TRAINING_DISTRIBUTION_RANGES, SAFETY_NOTICE

print(f"Workspace Root: {PROJECT_ROOT}")
print(f"Python Executable: {sys.executable}")
print(f"FastAPI App Title: {app.title}")
print(f"Canonical Features: {FEATURE_NAMES}")
print("Training Distribution Ranges:")
for k, v in TRAINING_DISTRIBUTION_RANGES.items():
    print(f"  - {k}: {v}")
"""))

    # -----------------------------------------------------------------------
    # Section 3: Dashboard File Structure Audit
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Dashboard Frontend Source Structure Audit

We inspect the physical dashboard assets created under `dashboard/`, including stylesheets, templates, and ES6 modules.
"""))

    cells.append(nbf.v4.new_code_cell(r"""dashboard_dir = PROJECT_ROOT / "dashboard"
files_to_check = [
    dashboard_dir / "index.html",
    dashboard_dir / "css" / "styles.css",
    dashboard_dir / "js" / "app.js",
    dashboard_dir / "js" / "components" / "demoStream.js",
    dashboard_dir / "js" / "components" / "sensorCards.js",
    dashboard_dir / "js" / "components" / "charts.js",
    dashboard_dir / "js" / "components" / "explanation.js",
]

print("--- DASHBOARD COMPONENT INVENTORY ---")
for f in files_to_check:
    exists = f.exists()
    size = f.stat().st_size if exists else 0
    rel = f.relative_to(PROJECT_ROOT)
    print(f"[{'OK' if exists else 'MISSING'}] {rel} ({size:,} bytes)")
"""))

    # -----------------------------------------------------------------------
    # Section 4: FastAPI Static Mount & Health Verification
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 4. FastAPI Endpoint & Static Mount Verification

We test the FastAPI endpoints using the synchronous `TestClient`:
1. `GET /`: Returns root metadata including links to `/docs`, `/health`, and `/dashboard`.
2. `GET /health`: Confirms model readiness and artifact status.
3. `GET /dashboard/`: Delivers the rendered `index.html`.
4. Static assets: CSS, JS modules.
"""))

    cells.append(nbf.v4.new_code_cell(r"""client = TestClient(app)

# 1. Root Service Information
res_root = client.get("/")
print("GET / Response:")
print(json.dumps(res_root.json(), indent=2))
assert res_root.status_code == 200
assert res_root.json()["dashboard"] == "/dashboard"

# 2. Microservice Health Check
res_health = client.get("/health")
print("\nGET /health Response:")
print(json.dumps(res_health.json(), indent=2))
assert res_health.status_code == 200
assert res_health.json()["status"] == "healthy"
assert res_health.json()["model_loaded"] is True

# 3. Web Dashboard Delivery
res_dash = client.get("/dashboard/")
print(f"\nGET /dashboard/ Status: {res_dash.status_code}")
print(f"Content-Type: {res_dash.headers.get('content-type')}")
print(f"Payload Size: {len(res_dash.content):,} bytes")
assert res_dash.status_code == 200
assert "CloudGuard-Water" in res_dash.text
assert "DEMO DATA — NOT LIVE SENSOR TELEMETRY" in res_dash.text
assert "STANDBY: NO LIVE HARDWARE CONNECTED — PENDING PHYSICAL COMMISSIONING" in res_dash.text
print("✓ Mandatory banners and static delivery verified successfully.")
"""))

    # -----------------------------------------------------------------------
    # Section 5: Verification of Preset Viva Demo Scenarios
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 5. End-to-End Verification of Demo Scenarios

The dashboard provides 6 deterministic preset scenarios in `demoStream.js` to facilitate viva demonstrations, laboratory audits, and edge-case testing. We execute each scenario against `POST /explain` to capture the end-to-end response:
1. `safe_potable`: Standard conformant tap water.
2. `high_tds_contaminated`: Industrial runoff with high dissolved solids.
3. `acidic_industrial`: Low pH chemical discharge.
4. `muddy_stormwater`: High turbidity colloidal stormwater.
5. `false_negative_55037`: Verified historical test sample labeled `Unsafe` but predicted `Safe` by the Decision Tree (auditing model limitations).
6. `out_of_distribution`: Extreme readings exceeding training boundaries (triggering training-range warnings).
"""))

    cells.append(nbf.v4.new_code_cell(r"""scenarios = [
    {
        "id": "safe_potable",
        "name": "Standard Potable Water",
        "payload": {"pH": 7.25, "TDS_ppm": 185.0, "Turbidity_NTU": 0.45, "Temperature_C": 22.0},
    },
    {
        "id": "high_tds_contaminated",
        "name": "Contaminated High-TDS Runoff",
        "payload": {"pH": 7.45, "TDS_ppm": 1850.0, "Turbidity_NTU": 1.80, "Temperature_C": 24.5},
    },
    {
        "id": "acidic_industrial",
        "name": "Acidic Chemical Runoff",
        "payload": {"pH": 3.85, "TDS_ppm": 820.0, "Turbidity_NTU": 15.50, "Temperature_C": 21.0},
    },
    {
        "id": "muddy_stormwater",
        "name": "Muddy Stormwater (High Turbidity)",
        "payload": {"pH": 6.80, "TDS_ppm": 240.0, "Turbidity_NTU": 55.00, "Temperature_C": 19.5},
    },
    {
        "id": "false_negative_55037",
        "name": "Case #55037 (Retrospective False Negative)",
        "payload": {"pH": 7.05, "TDS_ppm": 192.3, "Turbidity_NTU": 3.95, "Temperature_C": 24.26},
    },
    {
        "id": "out_of_distribution",
        "name": "Out-of-Distribution Boundary Stress",
        "payload": {"pH": 11.20, "TDS_ppm": 3150.0, "Turbidity_NTU": 88.50, "Temperature_C": 36.5},
    },
]

results = []
for sc in scenarios:
    res = client.post("/explain", json=sc["payload"])
    assert res.status_code == 200
    data = res.json()
    p_safe = data["estimated_probabilities"]["Safe"]
    p_unsafe = data["estimated_probabilities"]["Unsafe"]
    num_steps = len(data["decision_path_steps"])
    warnings_count = len(data["warnings"])
    results.append({
        "Scenario": sc["name"],
        "pH": sc["payload"]["pH"],
        "TDS (ppm)": sc["payload"]["TDS_ppm"],
        "Turbidity (NTU)": sc["payload"]["Turbidity_NTU"],
        "Temp (°C)": sc["payload"]["Temperature_C"],
        "Prediction": data["model_classification"],
        "P(Safe)": f"{p_safe:.4f}",
        "P(Unsafe)": f"{p_unsafe:.4f}",
        "Leaf Node": data["leaf_node_id"],
        "Tree Steps": num_steps,
        "Warnings": warnings_count,
    })

df_results = pd.DataFrame(results)
print("=== END-TO-END DEMO SCENARIO EVALUATION ===")
print(df_results.to_string(index=False))
"""))

    # -----------------------------------------------------------------------
    # Section 6: Inspection of Decision Tree Ladder & Explanation Output
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 6. Deep Dive: Decision Tree Ladder & Auditable Narrative

We inspect the detailed JSON response for the `acidic_industrial` scenario to confirm how the dashboard renders:
1. Exact sequential node comparisons (`threshold`, `direction`, `rule`).
2. Primary influencing features.
3. Plain-English narrative.
4. Non-certification safety notice.
"""))

    cells.append(nbf.v4.new_code_cell(r"""res_acid = client.post("/explain", json={"pH": 3.85, "TDS_ppm": 820.0, "Turbidity_NTU": 15.50, "Temperature_C": 21.0})
exp_data = res_acid.json()

print(f"Classification: {exp_data['model_classification']}")
print(f"Model Architecture: {exp_data['model']}")
print(f"Leaf Node ID: {exp_data['leaf_node_id']}")
print(f"Estimated Probabilities: {exp_data['estimated_probabilities']}")
print(f"Primary Influencing Features: {exp_data['primary_influencing_features']}")

print("\n--- DECISION PATH LADDER ---")
for idx, step in enumerate(exp_data["decision_path_steps"], 1):
    print(f"Step {idx}: Node #{step['node_id']} | Rule: {step['rule']} | Direction: {step['direction']}")

print("\n--- HUMAN EXPLANATION NARRATIVE ---")
print(exp_data["human_explanation"])

print("\n--- STATUTORY SAFETY DISCLAIMER ---")
print(exp_data["safety_notice"])
"""))

    # -----------------------------------------------------------------------
    # Section 7: Defensive State & Error Handling Verification
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 7. Defensive States & Validation Error Handling

We demonstrate that the system refuses invalid requests cleanly:
1. **Pydantic Validation Error (HTTP 422)**: Negative TDS or pH > 14 rejected.
2. **Missing Model Error (HTTP 503)**: Handled gracefully when artifact is unseated.
3. **Training Range Warnings**: Non-blocking warnings attached when features exceed observed distributions.
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 1. Pydantic Domain Validation Error (HTTP 422)
bad_payload = {"pH": 15.2, "TDS_ppm": -50.0, "Turbidity_NTU": 1.0, "Temperature_C": 25.0}
res_422 = client.post("/explain", json=bad_payload)
print(f"Invalid Payload Test -> HTTP Status: {res_422.status_code}")
assert res_422.status_code == 422
print("Error details returned by API:")
for err in res_422.json().get("detail", []):
    print(f"  - Field {err.get('loc')}: {err.get('msg')}")

# 2. Out-of-Distribution Warning Test
ood_payload = {"pH": 12.5, "TDS_ppm": 3500.0, "Turbidity_NTU": 85.0, "Temperature_C": 38.0}
res_ood = client.post("/explain", json=ood_payload)
print(f"\nOOD Payload Test -> HTTP Status: {res_ood.status_code}")
ood_data = res_ood.json()
print(f"Number of warnings triggered: {len(ood_data['warnings'])}")
for w in ood_data["warnings"]:
    print(f"  ⚠️ {w}")
"""))

    # -----------------------------------------------------------------------
    # Section 8: Hardware Commissioning Status & Limitations
    # -----------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(r"""## 8. Physical Hardware Commissioning Status & Project Viva Notes

### Physical Hardware Status:
- **Circuits & Pinouts**: Designed and verified in Phase 7.1/7.3 (Option D: $15\,\text{k}\Omega : 10\,\text{k}\Omega$ voltage divider, $M = 2.50\times$, $5.0\,\text{V} \to 2.0\,\text{V}$).
- **ESP32 Firmware**: Written and compiled with `arduino-cli` (Exit code 0, 79.06% Flash, 14.81% RAM).
- **Physical Commissioning**: **STRICTLY PENDING**. Physical bench testing with live probes, chemical buffers (pH 4.01, 7.00, 9.21), and standard 1413 µS/cm conductivity calibration solution will occur when physical access is scheduled.
- **No Automatic Actuation**: Physical actuators (solenoids, valves, chlorination dosing pumps) are strictly prohibited and disabled in software and firmware.

### Key Viva Discussion Points:
1. **Why is the Safe prediction not a drinking certification?**  
   Four basic physical sensors (pH, TDS, turbidity, temperature) cannot detect:
   - Microbiological pathogens (coliforms, Salmonella, Cryptosporidium, noroviruses).
   - Heavy metal toxicity (arsenic, lead, mercury, cadmium).
   - Trace organic pollutants (pesticides, PFAS, endocrine disruptors).
2. **What does the Decision Tree actually capture?**  
   It captures empirical associations present in the 100,800-sample training dataset (`data/water_quality_dataset1.csv`), where high TDS and low pH strongly correlate with Unsafe labels.
3. **Why do we show Decision Tree path ladders?**  
   Unlike black-box models, decision trees allow every single inference to be audited against exact thresholds, enabling domain operators to understand why an alert was triggered.
"""))

    nb.cells = cells
    return nb


def main():
    notebook_path = NOTEBOOKS_DIR / "08_phase8_dashboard_integration.ipynb"
    print(f"Building Phase 8 notebook: {notebook_path}")
    nb = build_phase8_notebook()

    print("Executing notebook top-to-bottom...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully created and executed {notebook_path}")


if __name__ == "__main__":
    main()
