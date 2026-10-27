"""
Unit and Integration Test Suite: tests/test_api.py
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Verify FastAPI REST microservice endpoints (/health, /predict, /explain),
         Pydantic input validation, physical domain constraints, training range warnings,
         probability calibration, decision path explainability, non-certifying safety wording,
         and missing model artifact handling (HTTP 503).
"""

from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from api.main import (
    app,
    load_api_model,
    app_state,
    TRAINING_DISTRIBUTION_RANGES,
    SAFETY_NOTICE,
)


class TestCloudGuardAPI(unittest.TestCase):
    """Automated test suite for the CloudGuard-Water FastAPI REST microservice."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and ensure canonical model artifact is loaded."""
        cls.client = TestClient(app)
        load_api_model()

    def tearDown(self):
        """Ensure canonical model is always restored after every test."""
        if app_state["model"] is None or app_state["error"] is not None:
            load_api_model()

    # -----------------------------------------------------------------------
    # 1. Health and Diagnostics Tests
    # -----------------------------------------------------------------------

    def test_root_endpoint(self):
        """Verify GET / returns service metadata and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("version", data)
        self.assertIn("/docs", data["documentation"])
        self.assertIn("/health", data["endpoints"])

    def test_health_healthy_status(self):
        """Verify GET /health returns 200 and healthy readiness when model is loaded."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_name"], "DecisionTreeClassifier")
        self.assertEqual(data["api_version"], "1.0.0")
        self.assertEqual(data["classes"], ["Safe", "Unsafe"])
        self.assertEqual(data["features"], ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"])
        self.assertIsNone(data["detail"])

    def test_health_unhealthy_when_model_missing(self):
        """Verify GET /health returns HTTP 503 when the required model artifact is missing."""
        # Simulate missing model artifact
        load_api_model(Path("models/nonexistent_dummy_model.joblib"))
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "unhealthy")
        self.assertFalse(data["model_loaded"])
        self.assertIsNone(data["model_name"])
        self.assertIsNotNone(data["detail"])

    # -----------------------------------------------------------------------
    # 2. Prediction Endpoint Tests (/predict)
    # -----------------------------------------------------------------------

    def test_predict_valid_safe_observation(self):
        """Verify POST /predict correctly classifies a representative potable water sample."""
        payload = {
            "pH": 7.38,
            "TDS_ppm": 138.7,
            "Turbidity_NTU": 0.25,
            "Temperature_C": 20.25,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_classification"], "Safe")
        self.assertIn("estimated_probabilities", data)
        probas = data["estimated_probabilities"]
        self.assertAlmostEqual(probas["Safe"], 1.0, places=3)
        self.assertAlmostEqual(probas["Unsafe"], 0.0, places=3)
        self.assertAlmostEqual(probas["Safe"] + probas["Unsafe"], 1.0, places=4)
        self.assertEqual(len(data["warnings"]), 0)
        self.assertIn("statistical machine learning classification", data["safety_notice"])

    def test_predict_valid_unsafe_observation(self):
        """Verify POST /predict correctly classifies a contaminated water sample."""
        payload = {
            "pH": 6.89,
            "TDS_ppm": 370.1,
            "Turbidity_NTU": 11.13,
            "Temperature_C": 22.25,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_classification"], "Unsafe")
        probas = data["estimated_probabilities"]
        self.assertAlmostEqual(probas["Safe"], 0.0, places=3)
        self.assertAlmostEqual(probas["Unsafe"], 1.0, places=3)
        self.assertAlmostEqual(probas["Safe"] + probas["Unsafe"], 1.0, places=4)
        self.assertEqual(len(data["warnings"]), 0)

    def test_predict_borderline_sample_probabilities(self):
        """Verify POST /predict returns estimated model class probabilities from the trained Decision Tree for borderline sample (Row 55037)."""
        payload = {
            "pH": 7.05,
            "TDS_ppm": 192.3,
            "Turbidity_NTU": 3.95,
            "Temperature_C": 24.26,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_classification"], "Safe")
        probas = data["estimated_probabilities"]
        self.assertAlmostEqual(probas["Safe"], 0.9948, places=3)
        self.assertAlmostEqual(probas["Unsafe"], 0.0052, places=3)
        self.assertAlmostEqual(probas["Safe"] + probas["Unsafe"], 1.0, places=4)

    # -----------------------------------------------------------------------
    # 3. Explainability Endpoint Tests (/explain)
    # -----------------------------------------------------------------------

    def test_explain_valid_observation(self):
        """Verify POST /explain returns full decision path, steps, and human-readable explanation."""
        payload = {
            "pH": 7.35,
            "TDS_ppm": 185.0,
            "Turbidity_NTU": 2.10,
            "Temperature_C": 22.50,
        }
        response = self.client.post("/explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["model_classification"], "Safe")
        self.assertEqual(data["leaf_node_id"], 5)
        self.assertIsInstance(data["decision_path_steps"], list)
        self.assertGreater(len(data["decision_path_steps"]), 0)

        # Verify step attributes
        first_step = data["decision_path_steps"][0]
        self.assertEqual(first_step["node_id"], 0)
        self.assertEqual(first_step["feature"], "TDS_ppm")
        self.assertAlmostEqual(first_step["threshold"], 202.75, places=2)
        self.assertEqual(first_step["direction"], "left")

        # Verify human explanation narrative and safety disclaimers
        explanation = data["human_explanation"]
        self.assertIn("Model Classification: SAFE", explanation)
        self.assertIn("Estimated Model Probabilities", explanation)
        self.assertIn("Operational Recommendation & Safety Notice", explanation)
        self.assertIn("ADVISORY:", explanation)
        self.assertIn("certified laboratory testing", explanation.lower())

    # -----------------------------------------------------------------------
    # 4. Input Validation & Edge Case Rejection Tests
    # -----------------------------------------------------------------------

    def test_reject_missing_features(self):
        """Verify rejection (HTTP 422) when one or more required features are omitted."""
        # Missing pH
        payload = {
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

        # Empty body
        response_empty = self.client.post("/predict", json={})
        self.assertEqual(response_empty.status_code, 422)

    def test_reject_nonnumeric_values(self):
        """Verify rejection (HTTP 422) when non-numeric strings are submitted."""
        payload = {
            "pH": "neutral",
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_reject_null_values(self):
        """Verify rejection (HTTP 422) when null/None values are provided."""
        payload = {
            "pH": None,
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_reject_nan_and_infinity_strings(self):
        """Verify rejection (HTTP 422) of NaN and Infinity float values."""
        for bad_val in ["NaN", "nan", "Infinity", "-Infinity", "inf", "-inf"]:
            payload = {
                "pH": bad_val,
                "TDS_ppm": 150.0,
                "Turbidity_NTU": 2.0,
                "Temperature_C": 22.0,
            }
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 422, f"Failed to reject invalid float: {bad_val}")

    def test_reject_invalid_physical_domains(self):
        """Verify rejection (HTTP 422) when measurements violate basic physical domain limits."""
        test_cases = [
            ("pH below 0.0", {"pH": -1.5, "TDS_ppm": 150.0, "Turbidity_NTU": 2.0, "Temperature_C": 22.0}),
            ("pH above 14.0", {"pH": 15.5, "TDS_ppm": 150.0, "Turbidity_NTU": 2.0, "Temperature_C": 22.0}),
            ("negative TDS", {"pH": 7.0, "TDS_ppm": -50.0, "Turbidity_NTU": 2.0, "Temperature_C": 22.0}),
            ("negative Turbidity", {"pH": 7.0, "TDS_ppm": 150.0, "Turbidity_NTU": -3.5, "Temperature_C": 22.0}),
            ("temperature below -10C", {"pH": 7.0, "TDS_ppm": 150.0, "Turbidity_NTU": 2.0, "Temperature_C": -25.0}),
            ("temperature above 100C", {"pH": 7.0, "TDS_ppm": 150.0, "Turbidity_NTU": 2.0, "Temperature_C": 125.0}),
        ]
        for desc, payload in test_cases:
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 422, f"Domain check failed for case: {desc}")

    def test_reject_extra_unexpected_fields(self):
        """Verify rejection (HTTP 422) when unexpected extra fields are submitted (extra='forbid')."""
        payload = {
            "pH": 7.0,
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
            "unauthorized_field": "injected_data",
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_out_of_training_range_generates_warnings(self):
        """
        Verify that physically valid inputs that lie outside the canonical training range
        are accepted (HTTP 200) but generate explicit data-quality warnings.
        """
        payload = {
            "pH": 10.5,             # Training range: [6.09, 9.96]
            "TDS_ppm": 3200.0,        # Training range: [28.6, 2806.1]
            "Turbidity_NTU": 85.0,    # Training range: [0.0, 72.75]
            "Temperature_C": 35.0,    # Training range: [16.56, 31.51]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        warnings = data["warnings"]
        self.assertEqual(len(warnings), 4)
        for feat in ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"]:
            self.assertTrue(
                any(feat in w for w in warnings),
                f"Missing out-of-range warning for feature: {feat}",
            )

    def test_in_range_values_produce_no_warnings(self):
        """
        Verify that sensor measurements completely within observed canonical training ranges
        produce zero training-range warnings.
        """
        payload = {
            "pH": 7.35,              # Within [6.09, 9.96]
            "TDS_ppm": 250.0,         # Within [28.6, 2806.1]
            "Turbidity_NTU": 3.5,     # Within [0.0, 72.75]
            "Temperature_C": 23.5,    # Within [16.56, 31.51]
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["warnings"], [])

    def test_individual_per_feature_training_range_warnings(self):
        """
        Verify that when exactly one feature falls outside its canonical training range,
        exactly one per-feature warning is generated specifically identifying that feature,
        while other compliant features do not trigger warnings.
        """
        base_payload = {
            "pH": 7.35,
            "TDS_ppm": 250.0,
            "Turbidity_NTU": 3.5,
            "Temperature_C": 23.5,
        }
        single_deviations = [
            ("pH", 5.5, "Feature 'pH'"),                   # Below min 6.09
            ("pH", 10.5, "Feature 'pH'"),                  # Above max 9.96
            ("TDS_ppm", 15.0, "Feature 'TDS_ppm'"),        # Below min 28.6
            ("TDS_ppm", 3000.0, "Feature 'TDS_ppm'"),      # Above max 2806.1
            ("Turbidity_NTU", 80.0, "Feature 'Turbidity_NTU'"),  # Above max 72.75
            ("Temperature_C", 14.0, "Feature 'Temperature_C'"),  # Below min 16.56
            ("Temperature_C", 35.0, "Feature 'Temperature_C'"),  # Above max 31.51
        ]

        for feat, deviant_val, expected_substr in single_deviations:
            payload = dict(base_payload)
            payload[feat] = deviant_val
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            warnings = data["warnings"]
            self.assertEqual(
                len(warnings),
                1,
                f"Expected exactly 1 warning for single deviation in {feat}={deviant_val}, got {len(warnings)}: {warnings}",
            )
            self.assertIn("Per-feature training-range warning:", warnings[0])
            self.assertIn(expected_substr, warnings[0])

    def test_probability_terminology_and_disclaimer(self):
        """
        Verify that responses describe probabilities as estimated model class probabilities
        and explicitly disclaim calibration or water potability guarantees.
        """
        payload = {
            "pH": 7.2,
            "TDS_ppm": 200.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
        }
        pred_resp = self.client.post("/predict", json=payload)
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.json()

        # Check probability keys and values
        probas = pred_data["estimated_probabilities"]
        self.assertIn("Safe", probas)
        self.assertIn("Unsafe", probas)
        self.assertAlmostEqual(probas["Safe"] + probas["Unsafe"], 1.0, places=4)
        self.assertGreaterEqual(probas["Safe"], 0.0)
        self.assertLessEqual(probas["Safe"], 1.0)

        # Check safety notice content
        safety_notice = pred_data["safety_notice"]
        self.assertIn("Estimated model class probabilities from the trained Decision Tree", safety_notice)
        self.assertIn("empirical training leaf proportions and do not guarantee real-world water safety", safety_notice)

        # Ensure misleading term 'calibrated empirical probabilities' is NOT present
        self.assertNotIn("calibrated empirical probabilities", str(pred_data).lower())

    def test_in_range_values_do_not_certify_safety(self):
        """
        Verify that in-range inputs do not claim drinking-water safety or compliance.
        """
        payload = {
            "pH": 7.35,
            "TDS_ppm": 250.0,
            "Turbidity_NTU": 3.5,
            "Temperature_C": 23.5,
        }
        pred_resp = self.client.post("/predict", json=payload)
        pred_text = str(pred_resp.json()).lower()

        # Safety notice must be included even when inside training range
        self.assertIn("safety_notice", pred_text)
        self.assertIn("not a certified drinking-water potability declaration", pred_text)


    # -----------------------------------------------------------------------
    # 5. Missing Model Handling on Inference (HTTP 503)
    # -----------------------------------------------------------------------

    def test_predict_and_explain_return_503_when_model_unavailable(self):
        """Verify /predict and /explain return controlled HTTP 503 when the model is missing."""
        load_api_model(Path("models/nonexistent_model.joblib"))
        payload = {
            "pH": 7.0,
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 2.0,
            "Temperature_C": 22.0,
        }
        pred_resp = self.client.post("/predict", json=payload)
        self.assertEqual(pred_resp.status_code, 503)
        self.assertIn("Model artifact is not loaded", pred_resp.json()["detail"])

        explain_resp = self.client.post("/explain", json=payload)
        self.assertEqual(explain_resp.status_code, 503)
        self.assertIn("Model artifact is not loaded", explain_resp.json()["detail"])

    # -----------------------------------------------------------------------
    # 6. Safety Wording Verification
    # -----------------------------------------------------------------------

    def test_safety_wording_never_claims_safe_to_drink(self):
        """Verify API outputs never make unsupported claims that water is certified safe to drink."""
        payload = {
            "pH": 7.35,
            "TDS_ppm": 150.0,
            "Turbidity_NTU": 1.5,
            "Temperature_C": 21.0,
        }
        pred_resp = self.client.post("/predict", json=payload)
        explain_resp = self.client.post("/explain", json=payload)

        pred_text = str(pred_resp.json()).lower()
        explain_text = str(explain_resp.json()).lower()

        prohibited_phrases = [
            "water is safe to drink",
            "guaranteed safe",
            "certified safe",
            "parameters comply with official guidelines",
            "parameters comply with all official",
        ]

        for phrase in prohibited_phrases:
            self.assertNotIn(phrase, pred_text, f"Prohibited phrase found in /predict response: {phrase}")
            self.assertNotIn(phrase, explain_text, f"Prohibited phrase found in /explain response: {phrase}")


if __name__ == "__main__":
    unittest.main()
