"""
Module: api.main
Project: CloudGuard-Water: Intelligent and Explainable Water Quality Classification
Purpose: FastAPI REST microservice exposing health, prediction, and explainability endpoints.
         Provides Pydantic request validation, domain range checks, scikit-learn model inference,
         decision path extraction, and defensive safety disclaimers for cloud & IoT integration.
"""

from contextlib import asynccontextmanager
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ConfigDict, Field, field_validator
from pydantic.main import BaseModel
from sklearn.tree import DecisionTreeClassifier

from src.explain import (
    CLASS_LABELS,
    FEATURE_NAMES,
    extract_decision_path,
    generate_human_explanation,
    load_model,
)

# ---------------------------------------------------------------------------
# Canonical Training & Physical Domain Boundaries
# ---------------------------------------------------------------------------

# Canonical training distribution ranges from data/water_quality_dataset1.csv (N=100,800)
# Used to flag unusual or out-of-distribution inputs without rejecting them.
TRAINING_DISTRIBUTION_RANGES: Dict[str, Tuple[float, float]] = {
    "pH": (6.09, 9.96),
    "TDS_ppm": (28.6, 2806.1),
    "Turbidity_NTU": (0.0, 72.75),
    "Temperature_C": (16.56, 31.51),
}

# Standardized defensive safety notice for all inference responses
SAFETY_NOTICE: str = (
    "This output is a statistical machine learning classification based on 4 physical sensor inputs "
    "(pH, TDS, turbidity, temperature), not a certified drinking-water potability declaration. "
    "Estimated model class probabilities from the trained Decision Tree (predict_proba values) are "
    "empirical training leaf proportions and do not guarantee real-world water safety. Physical sensors "
    "cannot detect microbiological pathogens (e.g., E. coli, Salmonella, noroviruses), toxic heavy metals "
    "(e.g., arsenic, lead, mercury), or hazardous organic chemicals. Standardized laboratory testing under "
    "statutory drinking-water standards (such as WHO guidelines or BIS IS 10500) remains strictly necessary "
    "to establish actual potability. Do not automatically actuate physical equipment (valves, pumps) based "
    "solely on this prediction."
)

# Global application state container for loaded model artifacts
app_state: Dict[str, Any] = {
    "model": None,
    "model_path": None,
    "model_name": None,
    "error": None,
}


def load_api_model(custom_path: Optional[Path] = None) -> None:
    """
    Safely load or reload the model artifact into app_state.
    Can be invoked at startup or in tests to simulate missing model conditions.
    """
    try:
        loaded = load_model(custom_path)
        app_state["model"] = loaded
        app_state["model_path"] = str(custom_path) if custom_path else "models/decision_tree.joblib"
        app_state["model_name"] = type(loaded).__name__
        app_state["error"] = None
    except Exception as exc:
        app_state["model"] = None
        app_state["model_path"] = str(custom_path) if custom_path else "models/decision_tree.joblib"
        app_state["model_name"] = None
        app_state["error"] = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to ensure model artifact is loaded on application startup."""
    if app_state["model"] is None:
        load_api_model()
    yield

# Initial model load on module import
load_api_model()


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CloudGuard-Water Prediction & Explainability API",
    description=(
        "Production-ready REST microservice for intelligent water-quality classification. "
        "Provides real-time machine learning predictions, probability estimations, and "
        "transparent Decision Tree path explainability from physical sensor measurements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Operational security: restrict CORS to local and testing domains by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Web Dashboard Integration
# ---------------------------------------------------------------------------
dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

# Simulation Console Dashboard Integration (CloudGuard-Water-Simulation-Dashboard)
sim_dashboard_dir = (
    Path(__file__).resolve().parent.parent
    / "CloudGuard-Water-Simulation-Dashboard"
    / "CloudGuard-Water-Simulation-Dashboard"
)
if not (sim_dashboard_dir / "index.html").exists():
    sim_dashboard_dir = Path(__file__).resolve().parent.parent / "CloudGuard-Water-Simulation-Dashboard"
if sim_dashboard_dir.exists() and (sim_dashboard_dir / "index.html").exists():
    app.mount("/simulation", StaticFiles(directory=str(sim_dashboard_dir), html=True), name="simulation")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class WaterQualityInput(BaseModel):
    """
    Input schema representing 4 physical water sensor measurements.
    Applies strict physical domain boundaries, finiteness validation, and rejects unknown fields.
    """
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pH": 7.35,
                "TDS_ppm": 185.0,
                "Turbidity_NTU": 2.10,
                "Temperature_C": 22.50,
            }
        },
    )

    pH: float = Field(
        ...,
        description="Potential of Hydrogen measurement on standard aqueous scale [0.0 to 14.0]",
        ge=0.0,
        le=14.0,
    )
    TDS_ppm: float = Field(
        ...,
        description="Total Dissolved Solids in parts per million (ppm) [>= 0.0]",
        ge=0.0,
    )
    Turbidity_NTU: float = Field(
        ...,
        description="Turbidity in Nephelometric Turbidity Units (NTU) [>= 0.0]",
        ge=0.0,
    )
    Temperature_C: float = Field(
        ...,
        description="Water temperature in degrees Celsius [-10.0 to 100.0]",
        ge=-10.0,
        le=100.0,
    )

    @field_validator("pH", "TDS_ppm", "Turbidity_NTU", "Temperature_C")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        """Ensure feature values are finite and reject NaN or Infinite values."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Feature value must be a finite floating-point number (NaN or Inf rejected).")
        return v


class PredictionResponse(BaseModel):
    """Response schema for model prediction endpoint."""
    model_classification: str = Field(..., description="Predicted water potability classification ('Safe' or 'Unsafe')")
    estimated_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Estimated model class probabilities from the trained Decision Tree ('Safe' vs 'Unsafe'). "
            "These are empirical leaf node sample proportions and do not guarantee real-world water safety."
        ),
    )
    model: str = Field(..., description="Model identifier and architecture details")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking per-feature training-range warnings (not an anomaly detector)",
    )
    safety_notice: str = Field(..., description="Defensive operational advisory and non-certification notice")


class DecisionPathStep(BaseModel):
    """Step in the Decision Tree node traversal path."""
    node_id: int
    feature: str
    value: float
    threshold: float
    direction: str
    rule: str


class ExplanationResponse(BaseModel):
    """Response schema for model explainability endpoint."""
    model_classification: str = Field(..., description="Predicted class ('Safe' or 'Unsafe')")
    estimated_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Estimated model class probabilities from the trained Decision Tree ('Safe' vs 'Unsafe'). "
            "These are empirical leaf node sample proportions and do not guarantee real-world water safety."
        ),
    )
    model: str = Field(..., description="Model identifier")
    leaf_node_id: int = Field(..., description="Terminal leaf node index in the Decision Tree")
    decision_path_steps: List[DecisionPathStep] = Field(..., description="Sequence of decision node threshold checks")
    primary_influencing_features: List[str] = Field(..., description="Features utilized in the decision path")
    human_explanation: str = Field(..., description="Auditable plain-English operational narrative")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking per-feature training-range warnings (not an anomaly detector)",
    )
    safety_notice: str = Field(..., description="Defensive operational advisory and non-certification notice")


class HealthResponse(BaseModel):
    """Response schema for microservice readiness and health check."""
    status: str = Field(..., description="Health status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Flag indicating whether the ML model artifact is loaded in memory")
    model_name: Optional[str] = Field(None, description="Class name of the loaded model")
    api_version: str = Field("1.0.0", description="API semantic version")
    classes: List[str] = Field(default_factory=list, description="Target classification classes")
    features: List[str] = Field(default_factory=list, description="Canonical input feature names in expected order")
    detail: Optional[str] = Field(None, description="Diagnostic error detail if unhealthy")


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------

def check_training_range_warnings(data: WaterQualityInput) -> List[str]:
    """
    Inspect whether individual sensor measurements fall outside their observed
    per-feature training ranges from the canonical training dataset.
    Generates non-blocking per-feature training-range warnings.

    Note: These are 1D marginal per-feature training-range warnings, NOT a complete
    multivariate anomaly detector or out-of-distribution detector. Values within
    the training range are not certified as safe or normal in combination, and values
    outside indicate measurements beyond the model's observed training bounds.
    """
    warnings: List[str] = []
    for feat, (low, high) in TRAINING_DISTRIBUTION_RANGES.items():
        val = getattr(data, feat)
        if val < low or val > high:
            warnings.append(
                f"Per-feature training-range warning: Feature '{feat}' value {val} is outside "
                f"the canonical training range [{low}, {high}]. "
                "Predictions on measurements outside observed training ranges may have reduced statistical reliability."
            )
    return warnings


def get_active_model() -> Any:
    """
    Retrieve the active in-memory model or raise HTTP 503 if unavailable.
    """
    model = app_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model artifact is not loaded or currently unavailable. "
                f"Diagnostic detail: {app_state.get('error', 'Artifact missing or corrupted.')}"
            ),
        )
    return model


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["Root"],
    summary="API Service Information",
    description="Returns root service metadata and documentation links.",
)
async def root():
    return {
        "service": "CloudGuard-Water Prediction & Explainability Microservice",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "dashboard": "/dashboard",
        "simulation_dashboard": "/simulation",
        "endpoints": ["/health", "/predict", "/explain"],
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Microservice Readiness and Artifact Health Check",
    description=(
        "Verifies whether the REST API is running and confirms that the required "
        "trained model artifact is loaded in memory. Returns HTTP 503 if the model is missing."
    ),
)
async def health():
    model = app_state.get("model")
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_name": None,
                "api_version": "1.0.0",
                "classes": ["Safe", "Unsafe"],
                "features": FEATURE_NAMES,
                "detail": app_state.get("error", "Model artifact not loaded or missing from disk."),
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=app_state.get("model_name", "DecisionTreeClassifier"),
        api_version="1.0.0",
        classes=["Safe", "Unsafe"],
        features=FEATURE_NAMES,
        detail=None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Water Quality Potability Classification",
    description=(
        "Accepts 4 physical sensor readings (pH, TDS_ppm, Turbidity_NTU, Temperature_C), "
        "validates input domain constraints, executes the trained Decision Tree, and returns "
        "the statistical classification, estimated model class probabilities from the trained Decision Tree, "
        "per-feature training-range warnings, and safety notice."
    ),
)
async def predict(data: WaterQualityInput):
    model = get_active_model()
    warnings = check_training_range_warnings(data)

    # Coerce to 1-row DataFrame preserving exact canonical feature order
    df = pd.DataFrame(
        [[data.pH, data.TDS_ppm, data.Turbidity_NTU, data.Temperature_C]],
        columns=FEATURE_NAMES,
    )

    # Execute prediction & probability estimation
    raw_pred = model.predict(df)[0]
    raw_probas = model.predict_proba(df)[0]

    # Map probabilities strictly using model.classes_ ordering
    classes_list = list(model.classes_)
    if 0 in classes_list and 1 in classes_list:
        idx_safe = classes_list.index(0)
        idx_unsafe = classes_list.index(1)
        p_safe = float(raw_probas[idx_safe])
        p_unsafe = float(raw_probas[idx_unsafe])
    else:
        p_safe = float(raw_probas[0])
        p_unsafe = float(raw_probas[1])

    pred_label = CLASS_LABELS.get(int(raw_pred), "Unknown")
    model_desc = f"{type(model).__name__}(max_depth={getattr(model, 'max_depth', None)})"

    return PredictionResponse(
        model_classification=pred_label,
        estimated_probabilities={
            "Safe": round(p_safe, 4),
            "Unsafe": round(p_unsafe, 4),
        },
        model=model_desc,
        warnings=warnings,
        safety_notice=SAFETY_NOTICE,
    )


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Explainable Decision Path Extraction and Narrative",
    description=(
        "Executes the trained Decision Tree on the input measurements and provides full explainability: "
        "exact node traversal, learned threshold comparisons, leaf node distribution, "
        "estimated model class probabilities from the trained Decision Tree, per-feature training-range warnings, "
        "and a structured plain-English operational explanation with advisory notices."
    ),
)
async def explain(data: WaterQualityInput):
    model = get_active_model()
    warnings = check_training_range_warnings(data)

    # Coerce to 1-row DataFrame preserving canonical feature order
    df = pd.DataFrame(
        [[data.pH, data.TDS_ppm, data.Turbidity_NTU, data.Temperature_C]],
        columns=FEATURE_NAMES,
    )

    if not isinstance(model, DecisionTreeClassifier):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The active model does not support Decision Tree path extraction.",
        )

    # Extract decision path using reusable src/explain.py logic
    path_info = extract_decision_path(model, df)

    # Generate human explanation without user-supplied ground truth (live operational inference)
    human_text = generate_human_explanation(path_info, actual_label=None)

    steps = [
        DecisionPathStep(
            node_id=s["node_id"],
            feature=s["feature"],
            value=round(s["value"], 2),
            threshold=round(s["threshold"], 2),
            direction=s["direction"],
            rule=s["rule"],
        )
        for s in path_info["steps"]
    ]

    primary_features = list(dict.fromkeys([s["feature"] for s in path_info["steps"]]))
    model_desc = f"{type(model).__name__}(max_depth={getattr(model, 'max_depth', None)})"

    return ExplanationResponse(
        model_classification=path_info["predicted_class"],
        estimated_probabilities={
            "Safe": round(path_info["probabilities"]["Safe"], 4),
            "Unsafe": round(path_info["probabilities"]["Unsafe"], 4),
        },
        model=model_desc,
        leaf_node_id=path_info["leaf_node_id"],
        decision_path_steps=steps,
        primary_influencing_features=primary_features,
        human_explanation=human_text,
        warnings=warnings,
        safety_notice=SAFETY_NOTICE,
    )
