"""
FastAPI REST backend for Cognisense.

Exposes inference, raw-event feature extraction, model insights, session
simulation, and personal-baseline calibration.

All prediction responses carry an explicit disclaimer: outputs are estimated
behavioural states, not clinical or psychological assessments.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import CognitiveLoadPredictor, score_to_band
from feature_engineering import extract_features_from_events
from preprocessing import FEATURE_COLUMNS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DISCLAIMER = (
    "Cognisense estimates cognitive-load states from observable behavioural "
    "interaction patterns. It is an experimental ML prototype trained on "
    "synthetic data and is not a clinical or psychological diagnostic tool."
)

app = FastAPI(
    title="Cognisense API",
    description=(
        "ML-powered behavioural analytics for cognitive load estimation.\n\n"
        f"**Important:** {DISCLAIMER}"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_predictor: Optional[CognitiveLoadPredictor] = None


def get_predictor() -> CognitiveLoadPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CognitiveLoadPredictor(model_dir=MODELS_DIR)
    return _predictor


# ------------------------------------------------------------------ schemas
class FeatureVectorInput(BaseModel):
    typing_speed_wpm: float = Field(45.0, ge=0, le=200)
    avg_keystroke_interval_ms: float = Field(250.0, ge=0, le=5000)
    backspace_ratio: float = Field(0.08, ge=0.0, le=1.0)
    pause_frequency_per_min: float = Field(5, ge=0, le=120)
    mouse_velocity_px_s: float = Field(280.0, ge=0, le=10000)
    mouse_distance_px: float = Field(1800.0, ge=0, le=100000)
    click_frequency_per_min: float = Field(20.0, ge=0, le=500)
    idle_time_seconds: float = Field(7.0, ge=0, le=3600)
    error_rate: float = Field(0.10, ge=0.0, le=1.0)
    response_time_seconds: float = Field(8.5, ge=0, le=600)
    retry_count: float = Field(2, ge=0, le=100)
    context_switches_per_min: float = Field(4, ge=0, le=200)


class PersonalizedPredictInput(BaseModel):
    features: FeatureVectorInput
    calibration_windows: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Feature dicts recorded at the user's comfortable baseline.")


class RawEventsInput(BaseModel):
    duration_seconds: float = Field(30.0, gt=0)
    keystroke_timestamps: List[float] = []
    backspace_count: int = Field(0, ge=0)
    total_keystrokes: int = Field(0, ge=0)
    words_typed: int = Field(0, ge=0)
    mouse_positions: List[Dict[str, Any]] = []
    click_count: int = Field(0, ge=0)
    idle_periods_seconds: List[float] = []
    error_count: int = Field(0, ge=0)
    attempt_count: int = Field(1, ge=1)
    response_time_sec: float = Field(5.0, ge=0)
    retry_count: int = Field(0, ge=0)
    context_switches: int = Field(0, ge=0)


class CalibrationInput(BaseModel):
    calibration_windows: List[Dict[str, float]]


# ------------------------------------------------------------------ routes
@app.get("/")
@app.get("/api/health")
def health_check():
    try:
        predictor = get_predictor()
        return {
            "status": "online",
            "system": "Cognisense Behavioral Analytics",
            "model_active": predictor.model_name,
            "features_expected": FEATURE_COLUMNS,
            "disclaimer": DISCLAIMER,
        }
    except Exception as exc:
        return {"status": "degraded", "error": str(exc),
                "hint": "Run `python src/train.py` to build model artifacts."}


@app.post("/api/predict")
def predict_cognitive_load(payload: FeatureVectorInput):
    try:
        return get_predictor().predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict-personalized")
def predict_personalized(payload: PersonalizedPredictInput):
    """
    Predict relative to a personal baseline (PRD 23).

    Attributions are computed against the user's own calibrated norm, so a
    naturally slow typist is not flagged merely for typing slowly.
    """
    try:
        predictor = get_predictor()
        baseline = None
        if payload.calibration_windows:
            baseline = CognitiveLoadPredictor.build_personal_baseline(
                payload.calibration_windows)
        return predictor.predict(payload.features.model_dump(),
                                 personal_baseline=baseline)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/calibrate")
def calibrate_baseline(payload: CalibrationInput):
    """Compute a personal behavioural baseline from calibration windows."""
    try:
        baseline = CognitiveLoadPredictor.build_personal_baseline(
            payload.calibration_windows)
        return {"personal_baseline": baseline,
                "n_windows": len(payload.calibration_windows)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/predict-raw-events")
def predict_from_raw_events(payload: RawEventsInput):
    try:
        features = extract_features_from_events(payload.model_dump())
        result = get_predictor().predict(features)
        result["extracted_features"] = features
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-insights")
def get_model_insights():
    metrics_file = os.path.join(MODELS_DIR, "model_comparison.json")
    feat_imp_file = os.path.join(MODELS_DIR, "feature_importance.json")

    if not os.path.exists(metrics_file) or not os.path.exists(feat_imp_file):
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run `python src/train.py` first.")

    try:
        with open(metrics_file, "r") as f:
            comparison = json.load(f)
        with open(feat_imp_file, "r") as f:
            feature_importance = json.load(f)

        return {
            "comparison": comparison,
            "feature_importance": feature_importance,
            "importance_method": (
                "Permutation importance (mean macro-F1 drop when a feature is "
                "shuffled) measured on held-out subjects. Indicates which "
                "features the model relies on -- not proof those features "
                "cause cognitive load."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/feature-baselines")
def get_feature_baselines():
    """Population reference statistics used for z-score attributions."""
    try:
        return {"baselines": get_predictor().feature_baselines}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------- session simulation
# Scripted session phases. Each entry defines the behavioural profile of a
# narrative phase so the demo timeline tells a coherent story: warm-up ->
# focused work -> difficulty spike -> break -> post-break recovery.
SESSION_PHASES = [
    {"name": "warm_up",   "until": 3,  "wpm": (62, 5),  "backspace": (0.05, 0.01),
     "pause": 2,  "error": (0.04, 0.01), "switches": 1,  "response": (4.5, 0.5)},
    {"name": "focused",   "until": 7,  "wpm": (50, 6),  "backspace": (0.10, 0.02),
     "pause": 5,  "error": (0.09, 0.02), "switches": 4,  "response": (7.8, 1.0)},
    {"name": "difficulty", "until": 11, "wpm": (30, 4),  "backspace": (0.20, 0.04),
     "pause": 12, "error": (0.20, 0.04), "switches": 9,  "response": (14.0, 2.0)},
    {"name": "break",     "until": 12, "wpm": (12, 2),  "backspace": (0.02, 0.01),
     "pause": 1,  "error": (0.01, 0.005), "switches": 0, "response": (3.0, 0.4)},
    {"name": "recovery",  "until": 99, "wpm": (56, 5),  "backspace": (0.07, 0.015),
     "pause": 3,  "error": (0.05, 0.015), "switches": 2, "response": (5.5, 0.8)},
]


def _phase_for_minute(minute: int) -> dict:
    for phase in SESSION_PHASES:
        if minute <= phase["until"]:
            return phase
    return SESSION_PHASES[-1]


@app.post("/api/simulate-session")
def simulate_session(duration_minutes: int = 15,
                     session_type: str = "coding_challenge"):
    """
    Generate a scripted behavioural session timeline for demonstration.

    This is a SIMULATED session, not recorded human activity. Each minute's
    synthetic feature window is passed through the real trained model, so the
    resulting timeline reflects genuine model behaviour on scripted input.
    """
    if not 1 <= duration_minutes <= 120:
        raise HTTPException(status_code=400,
                            detail="duration_minutes must be between 1 and 120.")

    rng = np.random.default_rng(42 if session_type == "coding_challenge" else 100)
    predictor = get_predictor()
    timeline = []

    for minute in range(1, duration_minutes + 1):
        phase = _phase_for_minute(minute)

        wpm = float(max(5.0, rng.normal(*phase["wpm"])))
        error_rate = float(max(0.0, rng.normal(*phase["error"])))
        backspace = float(max(0.0, rng.normal(*phase["backspace"])))
        pause_freq = int(max(0, rng.poisson(phase["pause"])))
        switches = int(max(0, rng.poisson(phase["switches"])))
        response_time = float(max(0.5, rng.normal(*phase["response"])))

        features = {
            "typing_speed_wpm": wpm,
            "avg_keystroke_interval_ms": 60000.0 / (wpm * 5.0),
            "backspace_ratio": backspace,
            "pause_frequency_per_min": pause_freq,
            "mouse_velocity_px_s": float(rng.normal(300, 40)),
            "mouse_distance_px": float(rng.normal(1800, 200)),
            "click_frequency_per_min": float(rng.normal(20, 3)),
            "idle_time_seconds": float(max(0.0, rng.normal(6, 1.5))),
            "error_rate": error_rate,
            "response_time_seconds": response_time,
            "retry_count": int(rng.poisson(2)),
            "context_switches_per_min": switches,
        }

        result = predictor.predict(features)
        timeline.append({
            "minute": minute,
            "phase": phase["name"],
            "cognitive_load_score": result["cognitive_load_score"],
            "predicted_state": result["predicted_state"],
            "wpm": round(wpm, 1),
            "error_rate": round(error_rate * 100.0, 1),
            "pause_freq": pause_freq,
            "context_switches": switches,
            "is_break": phase["name"] == "break",
            "top_signal": result["top_contributing_signals"][0]["display_name"],
        })

    scores = [entry["cognitive_load_score"] for entry in timeline]
    states = [entry["predicted_state"] for entry in timeline]
    n = len(states)

    # Recovery: compare the last pre-break minute against the last minute.
    break_minutes = [e["minute"] for e in timeline if e["is_break"]]
    if break_minutes and break_minutes[0] > 1 and break_minutes[-1] < duration_minutes:
        pre = timeline[break_minutes[0] - 2]["cognitive_load_score"]
        post = timeline[-1]["cognitive_load_score"]
        recovery = CognitiveLoadPredictor.calculate_recovery(pre, post)
    else:
        recovery = None

    peak_entry = max(timeline, key=lambda e: e["cognitive_load_score"])
    avg_score = int(round(float(np.mean(scores))))

    return {
        "session_type": session_type,
        "duration_minutes": duration_minutes,
        "average_score": avg_score,
        "average_band": score_to_band(avg_score),
        "peak_score": peak_entry["cognitive_load_score"],
        "peak_minute": peak_entry["minute"],
        "time_in_states": {
            "Low": round(states.count("Low") / n * 100, 1),
            "Medium": round(states.count("Medium") / n * 100, 1),
            "High": round(states.count("High") / n * 100, 1),
        },
        "recovery_analysis": recovery,
        "timeline": timeline,
        "data_note": ("Simulated session for demonstration. Synthetic feature "
                      "windows scored by the real trained model."),
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
