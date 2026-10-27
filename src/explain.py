"""
Module: src.explain
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: Production-grade, reusable explainability layer for Decision Tree and
         tree ensemble models. Provides input validation, decision path extraction,
         probability mapping, and human-readable explanation generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TREE_LEAF: int = -1

# Canonical feature specification
FEATURE_NAMES: List[str] = ["pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C"]
CLASS_LABELS: Dict[int, str] = {0: "Safe", 1: "Unsafe"}

# Physical plausible sensor measurement ranges (for validation and warnings)
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "pH": (0.0, 14.0),
    "TDS_ppm": (0.0, 5000.0),
    "Turbidity_NTU": (0.0, 500.0),
    "Temperature_C": (0.0, 100.0),
}


class ModelLoadError(FileNotFoundError):
    """Raised when the specified model artifact cannot be found or loaded."""
    pass


class InvalidInputError(ValueError):
    """Raised when input features fail schema, presence, or type validation."""
    pass


def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Load a trained model artifact from disk with candidate path resolution.

    Args:
        model_path: Optional path to the joblib model artifact. If None,
                    defaults to 'models/decision_tree.joblib'.

    Returns:
        The loaded scikit-learn model object.

    Raises:
        ModelLoadError: If the model artifact cannot be located.
    """
    if model_path is None:
        candidates = [
            Path("models/decision_tree.joblib"),
            Path("../models/decision_tree.joblib"),
            Path("../../models/decision_tree.joblib"),
            Path(__file__).resolve().parent.parent / "models" / "decision_tree.joblib",
        ]
        target_path = None
        for p in candidates:
            if p.exists():
                target_path = p
                break
        if target_path is None:
            raise ModelLoadError("Default model 'models/decision_tree.joblib' not found in candidate paths.")
    else:
        target_path = Path(model_path)
        if not target_path.exists():
            raise ModelLoadError(f"Specified model path does not exist: {target_path}")

    return joblib.load(target_path)


def validate_input(
    features: Union[Dict[str, float], pd.DataFrame, List[float], np.ndarray]
) -> pd.DataFrame:
    """
    Validate and coerce input features into a standardized DataFrame matching
    the canonical feature names and ordering.

    Args:
        features: Input feature data as a dictionary, single-row DataFrame,
                  or 4-element list/array in [pH, TDS_ppm, Turbidity_NTU, Temperature_C] order.

    Returns:
        pd.DataFrame: A 1-row DataFrame containing the 4 ordered features.

    Raises:
        InvalidInputError: If features are missing, non-numeric, or null.
    """
    if isinstance(features, dict):
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            raise InvalidInputError(f"Missing required feature(s): {missing}. Expected: {FEATURE_NAMES}")
        data = {f: [float(features[f])] for f in FEATURE_NAMES}
        df = pd.DataFrame(data)

    elif isinstance(features, pd.DataFrame):
        missing = [f for f in FEATURE_NAMES if f not in features.columns]
        if missing:
            raise InvalidInputError(f"DataFrame missing required column(s): {missing}. Expected: {FEATURE_NAMES}")
        df = features[FEATURE_NAMES].copy()

    elif isinstance(features, (list, tuple, np.ndarray)):
        arr = np.array(features)
        if arr.size != len(FEATURE_NAMES):
            raise InvalidInputError(
                f"Expected {len(FEATURE_NAMES)} features [pH, TDS_ppm, Turbidity_NTU, Temperature_C], "
                f"got {arr.size} elements."
            )
        df = pd.DataFrame([arr.flatten()], columns=FEATURE_NAMES)
    else:
        raise InvalidInputError(f"Unsupported feature type: {type(features)}. Expected dict, DataFrame, or list/array.")

    # Check for NaNs or Infinities
    if df.isnull().any().any():
        raise InvalidInputError("Input contains null or NaN values.")
    if np.isinf(df.to_numpy()).any():
        raise InvalidInputError("Input contains infinite values.")

    # Check numeric types
    for col in FEATURE_NAMES:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except Exception as e:
            raise InvalidInputError(f"Feature '{col}' could not be converted to float: {e}")

    return df


def predict_with_probabilities(
    model: Any, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate predictions and estimated model class probabilities from the trained Decision Tree.

    Args:
        model: Trained scikit-learn classifier with predict and predict_proba methods.
        X: Validated input DataFrame.

    Returns:
        Tuple of:
          - class_indices: np.ndarray of predicted class integers (0 or 1)
          - probabilities: np.ndarray of shape (N, 2) where col 0 = P(Safe), col 1 = P(Unsafe)
                           representing estimated model class probabilities from the trained Decision Tree.
                           Note: predict_proba() outputs represent empirical leaf proportions from training
                           and do not guarantee real-world drinking-water safety.
          - class_labels: List of predicted string labels ('Safe' or 'Unsafe')
    """
    raw_preds = model.predict(X)
    raw_probas = model.predict_proba(X)

    # Validate model.classes_ ordering strictly
    classes = list(model.classes_)
    if classes != [0, 1] and classes != [0.0, 1.0]:
        raise ValueError(f"Unexpected model.classes_ ordering: {classes}. Expected [0, 1].")

    labels = [CLASS_LABELS.get(int(p), "Unknown") for p in raw_preds]
    return raw_preds, raw_probas, labels


def extract_decision_path(
    model: DecisionTreeClassifier, sample: pd.DataFrame
) -> Dict[str, Any]:
    """
    Extract the exact decision path followed by an observation through a DecisionTreeClassifier.

    Args:
        model: Trained DecisionTreeClassifier.
        sample: Validated 1-row DataFrame.

    Returns:
        Dict containing:
          - node_indices: List of integer node IDs traversed from root to leaf
          - steps: List of human-readable comparison steps
          - leaf_node_id: Integer ID of the terminal leaf
          - leaf_value: Class distribution/counts at the leaf
          - predicted_class: Predicted class label ('Safe' or 'Unsafe')
          - class_probabilities: Dict mapping class labels to probabilities
    """
    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError(f"extract_decision_path requires a DecisionTreeClassifier, got {type(model)}")

    df = validate_input(sample)
    feature_arr = df[FEATURE_NAMES].to_numpy()

    dp: Any = model.decision_path(df[FEATURE_NAMES])
    node_indices = [int(idx) for idx in dp.indices]
    tree = model.tree_

    steps = []
    for node_id in node_indices:
        if tree.children_left[node_id] == TREE_LEAF:
            break

        feat_idx = int(tree.feature[node_id])
        feat_name = FEATURE_NAMES[feat_idx]
        thresh = float(tree.threshold[node_id])
        val = float(feature_arr[0, feat_idx])

        if val <= thresh:
            comp_str = f"{feat_name} ({val:.2f}) <= {thresh:.2f}"
            direction = "left"
        else:
            comp_str = f"{feat_name} ({val:.2f}) > {thresh:.2f}"
            direction = "right"

        steps.append({
            "node_id": node_id,
            "feature": feat_name,
            "value": val,
            "threshold": thresh,
            "direction": direction,
            "rule": comp_str,
        })

    leaf_id = node_indices[-1]
    leaf_val = tree.value[leaf_id][0].tolist()
    total_leaf_samples = sum(leaf_val)
    probas = [count / total_leaf_samples if total_leaf_samples > 0 else 0.0 for count in leaf_val]

    pred_idx = int(np.argmax(leaf_val))
    pred_label = CLASS_LABELS.get(pred_idx, "Unknown")

    return {
        "node_indices": node_indices,
        "steps": steps,
        "leaf_node_id": leaf_id,
        "leaf_sample_count": total_leaf_samples,
        "leaf_distribution": leaf_val,
        "predicted_class_index": pred_idx,
        "predicted_class": pred_label,
        "probabilities": {
            "Safe": probas[0],
            "Unsafe": probas[1],
        },
    }


def get_feature_importances(model: Any) -> pd.DataFrame:
    """
    Extract and rank global feature importances from a tree-based model.

    Args:
        model: Trained tree classifier with feature_importances_ attribute.

    Returns:
        pd.DataFrame: Features sorted descending by relative importance percentage.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {type(model)} does not expose feature_importances_.")

    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
        "Percentage": (importances / importances.sum()) * 100.0 if importances.sum() > 0 else 0.0,
    })
    return df.sort_values(by="Importance", ascending=False).reset_index(drop=True)


def generate_human_explanation(
    explanation_dict: Dict[str, Any],
    actual_label: Optional[str] = None,
) -> str:
    """
    Format an extracted decision path dictionary into a structured, human-readable report.
    Guarantees rigorous scientific wording:
      - Predictions are explicitly identified as statistical model classifications, NOT safety certifications.
      - A 'Safe' prediction never claims water is safe to drink or complies with official standards.
      - Probabilities are explicitly labeled as estimated empirical training proportions, not guarantees.
      - Retrospective false negatives are clearly highlighted and explained as model classification errors.
      - Explicitly states that laboratory testing and statutory standards remain necessary to establish actual potability.

    Args:
        explanation_dict: Output dictionary from extract_decision_path.
        actual_label: Optional ground truth string ('Safe' or 'Unsafe') for retrospective auditing.

    Returns:
        str: Formatted multi-line human-readable explanation.
    """
    pred_class = explanation_dict["predicted_class"]
    probas = explanation_dict["probabilities"]
    steps = explanation_dict["steps"]
    leaf_id = explanation_dict["leaf_node_id"]

    # Identify primary triggering features
    features_used = list(dict.fromkeys([s["feature"] for s in steps]))

    lines = []
    lines.append(f"Model Classification: {pred_class.upper()} (Statistical Prediction — Not a Potability Certification)")
    lines.append("")
    lines.append("Estimated Model Probabilities (Empirical Training Proportions — Not Guarantees):")
    lines.append(f"  - Estimated P(Safe)  : {probas['Safe'] * 100:.2f}%")
    lines.append(f"  - Estimated P(Unsafe): {probas['Unsafe'] * 100:.2f}%")
    lines.append("")
    lines.append("Decision Path Traversal:")
    lines.append(f"  Traversed {len(steps)} decision node(s) to Leaf Node #{leaf_id}:")
    for i, s in enumerate(steps, 1):
        lines.append(f"    {i}. {s['rule']} (Split at node #{s['node_id']})")

    lines.append("")
    lines.append(f"Primary Influencing Features: {', '.join(features_used)}")
    lines.append("")

    # Check for retrospective ground-truth discrepancy
    if actual_label is not None:
        actual_norm = actual_label.strip().capitalize()
        pred_norm = pred_class.strip().capitalize()
        if actual_norm != pred_norm:
            lines.append("Retrospective Error Analysis:")
            if actual_norm == "Unsafe" and pred_norm == "Safe":
                lines.append(
                    f"  ALERT: KNOWN MODEL FALSE NEGATIVE (Ground Truth: '{actual_norm}' vs Model: '{pred_norm}')."
                )
                lines.append(
                    f"  Explanation: All 4 measured physical features fall within the model's learned 'Safe' envelope "
                    f"(Leaf Node #{leaf_id}, where {probas['Safe'] * 100:.2f}% of training samples were labeled Safe). "
                    f"However, the actual ground truth for this sample is Unsafe. This demonstrates a critical limitation: "
                    f"relying solely on physical sensor measurements (pH, TDS, turbidity, temperature) can produce "
                    f"dangerous false negatives when water contamination is driven by unmeasured factors (such as "
                    f"microbial pathogens, heavy metals, or dataset boundary noise)."
                )
            else:
                lines.append(
                    f"  NOTICE: MODEL FALSE POSITIVE (Ground Truth: '{actual_norm}' vs Model: '{pred_norm}')."
                )
                lines.append(
                    f"  Explanation: The model classified this observation conservatively as Unsafe based on learned "
                    f"thresholds, exceeding ground-truth label criteria."
                )
            lines.append("")

    lines.append("Operational Recommendation & Safety Notice:")
    if actual_label is not None and actual_label.strip().capitalize() == "Unsafe" and pred_class.strip().capitalize() == "Safe":
        lines.append(
            "  ACTION REQUIRED: Do NOT treat this water as potable. Because ground truth indicates contamination, "
            "the model classification is overruled. Comprehensive laboratory microbiological, toxicological, and "
            "chemical testing is strictly required before any consumption."
        )
    elif pred_class == "Unsafe":
        lines.append(
            "  WARNING: Physical sensor readings exceed the model's learned thresholds for potable classification. "
            "Do NOT consume this water. Further laboratory testing and corrective water treatment are required."
        )
    else:
        lines.append(
            "  ADVISORY: Measured physical sensor values fall within the model's learned classification envelope. "
            "However, this statistical classification does NOT certify potability or consumption safety, nor does it establish "
            "compliance with statutory drinking-water standards (such as WHO, US EPA, or BIS IS 10500). Physical sensors cannot detect "
            "microbial pathogens (e.g., E. coli, Salmonella), toxic heavy metals (arsenic, lead, mercury), or organic chemical "
            "pollutants. Comprehensive certified laboratory testing remains strictly necessary to establish actual drinking-water potability."
        )

    return "\n".join(lines)


def explain_sample(
    model: Any,
    features: Union[Dict[str, float], pd.DataFrame, List[float], np.ndarray],
    actual_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end convenience function to validate an input sample, compute model
    prediction and probabilities, extract the decision path (if Decision Tree),
    and generate the structured human-readable explanation.

    Args:
        model: Trained model (typically DecisionTreeClassifier).
        features: Input feature data.
        actual_label: Optional ground truth string ('Safe' or 'Unsafe') for retrospective auditing.

    Returns:
        Dict with keys:
          - input_features: Dict of input features
          - predicted_class: 'Safe' or 'Unsafe'
          - actual_label: Optional ground truth label
          - is_classification_error: Boolean flag indicating if prediction disagrees with actual_label
          - probabilities: Dict mapping class names to probabilities
          - decision_path: Extracted path dict (if DecisionTreeClassifier)
          - human_explanation: Formatted multi-line explanation text
    """
    validated_df = validate_input(features)
    sample_dict = validated_df.iloc[0].to_dict()

    if isinstance(model, DecisionTreeClassifier):
        path_info = extract_decision_path(model, validated_df)
        human_text = generate_human_explanation(path_info, actual_label=actual_label)
        is_error = False
        if actual_label is not None:
            is_error = (actual_label.strip().capitalize() != path_info["predicted_class"].strip().capitalize())

        return {
            "input_features": sample_dict,
            "predicted_class": path_info["predicted_class"],
            "actual_label": actual_label,
            "is_classification_error": is_error,
            "probabilities": path_info["probabilities"],
            "decision_path": path_info,
            "human_explanation": human_text,
        }
    else:
        # Non-decision tree (e.g. Random Forest, Logistic Regression)
        _, probas, labels = predict_with_probabilities(model, validated_df)
        p_safe, p_unsafe = float(probas[0, 0]), float(probas[0, 1])
        pred_label = labels[0]
        is_error = False
        if actual_label is not None:
            is_error = (actual_label.strip().capitalize() != pred_label.strip().capitalize())

        summary_lines = [
            f"Model Classification: {pred_label.upper()} (Statistical Prediction — Not a Potability Certification)",
            "",
            "Estimated Model Probabilities (Empirical Training Proportions — Not Guarantees):",
            f"  - Estimated P(Safe)  : {p_safe * 100:.2f}%",
            f"  - Estimated P(Unsafe): {p_unsafe * 100:.2f}%",
            "",
            "Operational Recommendation & Safety Notice:",
            "  ADVISORY: Statistical ensemble model classification does not certify potability or consumption safety. "
            "Certified laboratory testing remains strictly necessary to establish actual drinking-water potability.",
        ]
        return {
            "input_features": sample_dict,
            "predicted_class": pred_label,
            "actual_label": actual_label,
            "is_classification_error": is_error,
            "probabilities": {"Safe": p_safe, "Unsafe": p_unsafe},
            "decision_path": None,
            "human_explanation": "\n".join(summary_lines),
        }
