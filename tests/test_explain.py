"""
Unit tests for src/explain.py
CloudGuard-Water: Intelligent and Explainable Water Quality Classification
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.explain import (
    FEATURE_NAMES,
    CLASS_LABELS,
    load_model,
    validate_input,
    predict_with_probabilities,
    extract_decision_path,
    get_feature_importances,
    generate_human_explanation,
    explain_sample,
    InvalidInputError,
    ModelLoadError,
)


class TestExplainModule(unittest.TestCase):
    """Comprehensive test suite for the explainability layer."""

    @classmethod
    def setUpClass(cls):
        """Load actual trained model and reference data once for the test suite."""
        cls.model_path = Path("models/decision_tree.joblib")
        if not cls.model_path.exists():
            cls.model_path = Path("../models/decision_tree.joblib")
        cls.dt = load_model(cls.model_path)
        cls.rf = load_model(Path("models/random_forest.joblib") if Path("models/random_forest.joblib").exists() else Path("../models/random_forest.joblib"))

        # Test observation fixtures
        cls.safe_sample_dict = {"pH": 7.38, "TDS_ppm": 138.7, "Turbidity_NTU": 0.25, "Temperature_C": 20.25}
        cls.unsafe_sample_dict = {"pH": 6.89, "TDS_ppm": 370.1, "Turbidity_NTU": 11.13, "Temperature_C": 22.25}

    def test_model_loading_success(self):
        """Verify model loads successfully and matches expected scikit-learn type."""
        self.assertIsInstance(self.dt, DecisionTreeClassifier)
        self.assertEqual(list(self.dt.classes_), [0, 1])
        self.assertEqual(self.dt.n_features_in_, 4)

    def test_model_loading_nonexistent_raises_error(self):
        """Verify ModelLoadError is raised when model file does not exist."""
        with self.assertRaises(ModelLoadError):
            load_model("nonexistent_model.joblib")

    def test_validate_input_dict(self):
        """Verify input validation on dictionary inputs."""
        df = validate_input(self.safe_sample_dict)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), FEATURE_NAMES)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["pH"].iloc[0], 7.38)
        self.assertAlmostEqual(df["TDS_ppm"].iloc[0], 138.7)

    def test_validate_input_list(self):
        """Verify input validation on list / 1D array inputs."""
        raw_list = [7.38, 138.7, 0.25, 20.25]
        df = validate_input(raw_list)
        self.assertEqual(list(df.columns), FEATURE_NAMES)
        self.assertAlmostEqual(df["Turbidity_NTU"].iloc[0], 0.25)

    def test_validate_input_dataframe(self):
        """Verify input validation on DataFrame inputs with arbitrary column order."""
        raw_df = pd.DataFrame([{
            "Temperature_C": 20.25,
            "Turbidity_NTU": 0.25,
            "pH": 7.38,
            "TDS_ppm": 138.7,
        }])
        df = validate_input(raw_df)
        self.assertEqual(list(df.columns), FEATURE_NAMES)

    def test_validate_input_missing_feature_raises_error(self):
        """Verify InvalidInputError is raised when a required feature is omitted."""
        incomplete = {"pH": 7.38, "TDS_ppm": 138.7}
        with self.assertRaises(InvalidInputError):
            validate_input(incomplete)

    def test_validate_input_nan_raises_error(self):
        """Verify InvalidInputError is raised on null or NaN values."""
        nan_sample = {"pH": np.nan, "TDS_ppm": 138.7, "Turbidity_NTU": 0.25, "Temperature_C": 20.25}
        with self.assertRaises(InvalidInputError):
            validate_input(nan_sample)

    def test_validate_input_wrong_length_list_raises_error(self):
        """Verify InvalidInputError is raised if list has wrong length."""
        with self.assertRaises(InvalidInputError):
            validate_input([7.0, 150.0])

    def test_predict_with_probabilities_agreement(self):
        """Verify prediction functions agree exactly with model.predict and model.predict_proba."""
        df = validate_input(self.safe_sample_dict)
        preds, probas, labels = predict_with_probabilities(self.dt, df)

        expected_preds = self.dt.predict(df)
        expected_probas = self.dt.predict_proba(df)

        np.testing.assert_array_equal(preds, expected_preds)
        np.testing.assert_array_almost_equal(probas, expected_probas)
        self.assertEqual(labels[0], CLASS_LABELS[int(expected_preds[0])])

    def test_extract_decision_path_safe(self):
        """Verify decision path extraction on a safe water sample."""
        df = validate_input(self.safe_sample_dict)
        res = extract_decision_path(self.dt, df)

        self.assertEqual(res["predicted_class"], "Safe")
        self.assertIn("node_indices", res)
        self.assertGreater(len(res["node_indices"]), 1)
        self.assertIn("steps", res)
        self.assertGreater(len(res["steps"]), 0)

        # Check probability bounds
        self.assertAlmostEqual(res["probabilities"]["Safe"] + res["probabilities"]["Unsafe"], 1.0)
        self.assertGreater(res["probabilities"]["Safe"], 0.9)

        # Verify root split uses TDS_ppm <= 202.75
        first_step = res["steps"][0]
        self.assertEqual(first_step["feature"], "TDS_ppm")
        self.assertEqual(first_step["direction"], "left")
        self.assertLessEqual(first_step["value"], first_step["threshold"])

    def test_extract_decision_path_unsafe(self):
        """Verify decision path extraction on an unsafe water sample."""
        df = validate_input(self.unsafe_sample_dict)
        res = extract_decision_path(self.dt, df)

        self.assertEqual(res["predicted_class"], "Unsafe")
        self.assertAlmostEqual(res["probabilities"]["Unsafe"], 1.0)

        # Verify root split uses TDS_ppm > 202.75
        first_step = res["steps"][0]
        self.assertEqual(first_step["feature"], "TDS_ppm")
        self.assertEqual(first_step["direction"], "right")
        self.assertGreater(first_step["value"], first_step["threshold"])

    def test_generate_human_explanation_safe(self):
        """Verify human explanation text for Safe predictions uses non-certifying, probabilistic wording."""
        df = validate_input(self.safe_sample_dict)
        res = extract_decision_path(self.dt, df)
        text = generate_human_explanation(res)

        # Explicitly identified as model classification, not certification
        self.assertIn("Model Classification: SAFE", text)
        self.assertIn("Statistical Prediction — Not a Potability Certification", text)

        # Probabilities described as estimated training proportions, not guarantees
        self.assertIn("Estimated Model Probabilities (Empirical Training Proportions — Not Guarantees):", text)
        self.assertIn("Estimated P(Safe)", text)

        # NEVER produce unsupported claims
        self.assertNotIn("water is safe to drink", text.lower())
        self.assertNotIn("parameters comply with official guidelines", text.lower())
        self.assertNotIn("normal:", text)

        # Explicitly state laboratory testing and statutory standards are necessary
        self.assertIn("laboratory testing", text.lower())
        self.assertIn("statutory drinking-water standards", text.lower())

    def test_generate_human_explanation_unsafe(self):
        """Verify human explanation text for Unsafe predictions produces warning and testing recommendation."""
        df = validate_input(self.unsafe_sample_dict)
        res = extract_decision_path(self.dt, df)
        text = generate_human_explanation(res)

        self.assertIn("Model Classification: UNSAFE", text)
        self.assertIn("WARNING:", text)
        self.assertIn("laboratory testing", text.lower())

    def test_known_retrospective_false_negative_explanation(self):
        """
        Verify that a known retrospective false-negative (Row 55037: actual Unsafe, predicted Safe)
        is explicitly flagged and explained as a model error rather than giving a safe advisory.
        """
        fn_sample = {"pH": 7.05, "TDS_ppm": 192.3, "Turbidity_NTU": 3.95, "Temperature_C": 24.26}
        out = explain_sample(self.dt, fn_sample, actual_label="Unsafe")

        self.assertEqual(out["predicted_class"], "Safe")
        self.assertEqual(out["actual_label"], "Unsafe")
        self.assertTrue(out["is_classification_error"])

        text = out["human_explanation"]
        self.assertIn("ALERT: KNOWN MODEL FALSE NEGATIVE", text)
        self.assertIn("Ground Truth: 'Unsafe' vs Model: 'Safe'", text)
        self.assertIn("ACTION REQUIRED: Do NOT treat this water as potable", text)
        self.assertIn("laboratory microbiological", text.lower())
        self.assertNotIn("advisory: measured physical sensor values fall within", text.lower())

    def test_explain_sample_end_to_end(self):
        """Verify end-to-end explain_sample convenience function."""
        out = explain_sample(self.dt, self.safe_sample_dict)
        self.assertEqual(out["predicted_class"], "Safe")
        self.assertIn("Safe", out["probabilities"])
        self.assertIn("Unsafe", out["probabilities"])
        self.assertIsNotNone(out["decision_path"])
        self.assertIn("Model Classification: SAFE", out["human_explanation"])

    def test_get_feature_importances(self):
        """Verify feature importances are extracted, non-negative, and sum to 100%."""
        fi_dt = get_feature_importances(self.dt)
        self.assertEqual(len(fi_dt), 4)
        self.assertAlmostEqual(fi_dt["Percentage"].sum(), 100.0, places=3)
        self.assertEqual(fi_dt.iloc[0]["Feature"], "TDS_ppm")

        fi_rf = get_feature_importances(self.rf)
        self.assertEqual(len(fi_rf), 4)
        self.assertAlmostEqual(fi_rf["Percentage"].sum(), 100.0, places=3)
        self.assertEqual(fi_rf.iloc[0]["Feature"], "TDS_ppm")

    def test_artifacts_remain_intact(self):
        """Verify that existing Phase 3 and Phase 4 artifacts remain intact."""
        self.assertTrue(Path("models/decision_tree.joblib").exists())
        self.assertTrue(Path("models/random_forest.joblib").exists())
        self.assertTrue(Path("models/logistic_regression.joblib").exists())
        self.assertTrue(Path("data/water_quality_dataset1.csv").exists())
        self.assertTrue(Path("data/water_quality_dataset1_original.csv").exists())


if __name__ == "__main__":
    unittest.main()
