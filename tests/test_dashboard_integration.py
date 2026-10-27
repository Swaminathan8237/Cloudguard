"""
Unit and Integration Test Suite: tests/test_dashboard_integration.py
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Phase: Phase 8 — Web Dashboard and End-to-End Software Integration
Purpose: Verify that the FastAPI microservice correctly mounts and serves the web dashboard,
         static assets (HTML, CSS, JS), and validates API payloads for all demo scenarios,
         training-range warnings, defensive error states, and safety disclaimers.
"""

from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from api.main import (
    app,
    load_api_model,
    app_state,
    SAFETY_NOTICE,
)


class TestDashboardIntegration(unittest.TestCase):
    """Integration test suite for the CloudGuard-Water Dashboard."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        load_api_model()

    def tearDown(self):
        if app_state["model"] is None or app_state["error"] is not None:
            load_api_model()

    # -----------------------------------------------------------------------
    # 1. Dashboard HTML & Asset Delivery Tests
    # -----------------------------------------------------------------------

    def test_dashboard_root_serves_html(self):
        """Verify GET /dashboard/ serves index.html with HTTP 200 and text/html."""
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn("<title>CloudGuard-Water", body)
        self.assertIn("sensor-cards-grid", body)
        self.assertIn("telemetry-chart", body)
        self.assertIn("decision-path-steps", body)

    def test_dashboard_contains_mandatory_demo_banner(self):
        """Verify the dashboard HTML contains the visible DEMO DATA banner."""
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("DEMO DATA — NOT LIVE SENSOR TELEMETRY", body)
        self.assertIn("demo-mode-banner", body)

    def test_dashboard_contains_live_standby_banner(self):
        """Verify the dashboard HTML contains the pending physical commissioning banner."""
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("STANDBY: NO LIVE HARDWARE CONNECTED — PENDING PHYSICAL COMMISSIONING", body)
        self.assertIn("live-standby-banner", body)

    def test_dashboard_contains_safety_and_actuation_disclaimer(self):
        """Verify statutory safety notice and disabled actuation warning in dashboard HTML."""
        response = self.client.get("/dashboard/")
        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("not a certified drinking-water potability declaration", body)
        self.assertIn("Physical equipment actuation (valves, pumps) is strictly disabled", body)

    def test_dashboard_static_assets_exist_and_accessible(self):
        """Verify CSS and JavaScript modular components are accessible via static mount."""
        assets = [
            "/dashboard/css/styles.css",
            "/dashboard/js/app.js",
            "/dashboard/js/components/demoStream.js",
            "/dashboard/js/components/sensorCards.js",
            "/dashboard/js/components/charts.js",
            "/dashboard/js/components/explanation.js",
        ]
        for asset in assets:
            res = self.client.get(asset)
            self.assertEqual(res.status_code, 200, f"Failed to load static asset: {asset}")
            self.assertGreater(len(res.content), 50, f"Asset content suspiciously empty: {asset}")

    def test_root_endpoint_includes_dashboard_link(self):
        """Verify GET / metadata includes the /dashboard link."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dashboard", data)
        self.assertEqual(data["dashboard"], "/dashboard")

    # -----------------------------------------------------------------------
    # 2. Preset Demo Scenarios Integration with /explain API
    # -----------------------------------------------------------------------

    def test_scenario_safe_potable_pipeline(self):
        """Verify potable water scenario produces Safe classification and compliant metrics."""
        payload = {"pH": 7.25, "TDS_ppm": 185.0, "Turbidity_NTU": 0.45, "Temperature_C": 22.0}
        res = self.client.post("/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_classification"], "Safe")
        self.assertGreater(data["estimated_probabilities"]["Safe"], 0.5)
        self.assertEqual(len(data["warnings"]), 0)
        self.assertIn("safety_notice", data)

    def test_scenario_high_tds_contaminated_pipeline(self):
        """Verify high TDS runoff produces Unsafe classification."""
        payload = {"pH": 7.45, "TDS_ppm": 1850.0, "Turbidity_NTU": 1.80, "Temperature_C": 24.5}
        res = self.client.post("/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_classification"], "Unsafe")
        self.assertGreater(data["estimated_probabilities"]["Unsafe"], 0.5)
        self.assertIn("TDS_ppm", data["primary_influencing_features"])

    def test_scenario_acidic_industrial_pipeline(self):
        """Verify acidic chemical discharge produces Unsafe classification and path traversal."""
        payload = {"pH": 3.85, "TDS_ppm": 820.0, "Turbidity_NTU": 15.50, "Temperature_C": 21.0}
        res = self.client.post("/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_classification"], "Unsafe")
        self.assertGreater(len(data["decision_path_steps"]), 0)

    def test_scenario_out_of_distribution_triggers_warnings(self):
        """Verify out-of-distribution inputs trigger non-blocking training range warnings."""
        payload = {"pH": 11.20, "TDS_ppm": 3150.0, "Turbidity_NTU": 88.50, "Temperature_C": 36.5}
        res = self.client.post("/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(len(data["warnings"]), 0)
        # Check that warnings flag out of range features
        warning_str = " ".join(data["warnings"])
        self.assertIn("pH", warning_str)
        self.assertIn("TDS_ppm", warning_str)
        self.assertIn("Turbidity_NTU", warning_str)
        self.assertIn("Temperature_C", warning_str)

    def test_scenario_audit_false_negative_55037(self):
        """Verify test set case #55037 is auditable and returns Safe with tree path."""
        payload = {"pH": 7.05, "TDS_ppm": 192.3, "Turbidity_NTU": 3.95, "Temperature_C": 24.26}
        res = self.client.post("/explain", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_classification"], "Safe")
        self.assertIn("decision_path_steps", data)

    # -----------------------------------------------------------------------
    # 3. Defensive Error & Timeout States
    # -----------------------------------------------------------------------

    def test_interactive_validation_error_422(self):
        """Verify invalid physical parameter returns HTTP 422 unprocessable entity."""
        invalid_payload = {"pH": 15.5, "TDS_ppm": -10.0, "Turbidity_NTU": 1.0, "Temperature_C": 20.0}
        res = self.client.post("/explain", json=invalid_payload)
        self.assertEqual(res.status_code, 422)
        errors = res.json().get("detail", [])
        fields = [e["loc"][-1] for e in errors]
        self.assertIn("pH", fields)
        self.assertIn("TDS_ppm", fields)

    def test_disconnected_model_returns_503(self):
        """Verify missing model triggers HTTP 503 Service Unavailable."""
        app_state["model"] = None
        try:
            res = self.client.post("/explain", json={"pH": 7.0, "TDS_ppm": 200.0, "Turbidity_NTU": 1.0, "Temperature_C": 20.0})
            self.assertEqual(res.status_code, 503)
        finally:
            load_api_model()

    # -----------------------------------------------------------------------
    # 4. Simulation Console & CORS Integration
    # -----------------------------------------------------------------------

    def test_simulation_dashboard_serves_html(self):
        """Verify GET /simulation/ serves index.html with HTTP 200 and text/html."""
        response = self.client.get("/simulation/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        body = response.text
        self.assertIn("CloudGuard-Water", body)
        self.assertIn("SIMULATION MODE", body)
        self.assertIn("telemetryChart", body)

    def test_cors_allows_simulation_port_5173(self):
        """Verify CORS allows origin http://127.0.0.1:5173."""
        response = self.client.options(
            "/predict",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")


if __name__ == "__main__":
    unittest.main()
