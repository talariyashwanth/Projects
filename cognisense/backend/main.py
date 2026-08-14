"""
FastAPI REST Backend for Cognisense
Exposes endpoints for inference, real-time event feature extraction, model insights, and session analytics.
"""

import os
import sys
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from predict import CognitiveLoadPredictor
from feature_engineering import extract_features_from_events
from dataset_generator import generate_cognitive_load_dataset

app = FastAPI(
    title="Cognisense API",
    description="ML-powered Behavioral Analytics for Cognitive Load Detection",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        predictor = CognitiveLoadPredictor(model_dir=model_dir)
    return predictor

class FeatureVectorInput(BaseModel):
    typing_speed_wpm: float = Field(..., example=45.0)
    avg_keystroke_interval_ms: float = Field(..., example=250.0)
    backspace_ratio: float = Field(..., example=0.08)
    pause_frequency_per_min: int = Field(..., example=5)
    mouse_velocity_px_s: float = Field(..., example=280.0)
    mouse_distance_px: float = Field(..., example=1800.0)
    click_frequency_per_min: float = Field(..., example=20.0)
    idle_time_seconds: float = Field(..., example=7.0)
    error_rate: float = Field(..., example=0.10)
    response_time_seconds: float = Field(..., example=8.5)
    retry_count: int = Field(..., example=2)
    context_switches_per_min: int = Field(..., example=4)

class RawEventsInput(BaseModel):
    duration_seconds: float = 30.0
    keystroke_timestamps: List[float] = []
    backspace_count: int = 0
    total_keystrokes: int = 0
    words_typed: int = 0
    mouse_positions: List[Dict[str, Any]] = []
    click_count: int = 0
    idle_periods_seconds: List[float] = []
    error_count: int = 0
    attempt_count: int = 1
    response_time_sec: float = 5.0
    retry_count: int = 0
    context_switches: int = 0

@app.get("/")
@app.get("/api/health")
def health_check():
    try:
        pred = get_predictor()
        return {
            "status": "online",
            "system": "Cognisense Behavioral Analytics",
            "model_active": pred.model_name
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/api/predict")
def predict_cognitive_load(payload: FeatureVectorInput):
    try:
        pred = get_predictor()
        result = pred.predict(payload.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-raw-events")
def predict_from_raw_events(payload: RawEventsInput):
    try:
        extracted_features = extract_features_from_events(payload.dict())
        pred = get_predictor()
        result = pred.predict(extracted_features)
        result["extracted_features"] = extracted_features
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-insights")
def get_model_insights():
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        metrics_file = os.path.join(base_dir, "models", "model_comparison.json")
        feat_imp_file = os.path.join(base_dir, "models", "feature_importance.json")

        if not os.path.exists(metrics_file) or not os.path.exists(feat_imp_file):
            raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")

        with open(metrics_file, "r") as f:
            comparison = json.load(f)

        with open(feat_imp_file, "r") as f:
            feat_imp = json.load(f)

        return {
            "comparison": comparison,
            "feature_importance": feat_imp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate-session")
def simulate_session(duration_minutes: int = 15, session_type: str = "coding_challenge"):
    """
    Generates a realistic 15-minute time-series cognitive load timeline.
    Shows warm-up -> focus -> difficulty spike (high load) -> break -> recovery.
    """
    np.random.seed(42 if session_type == "coding_challenge" else 100)
    timestamps = list(range(1, duration_minutes + 1))
    timeline = []

    pred = get_predictor()
    prev_score = 30

    for minute in timestamps:
        if minute <= 3:
            # Warm up: Low load
            wpm = float(np.random.normal(65, 5))
            backspace = float(np.random.normal(0.04, 0.01))
            pause_freq = int(np.random.poisson(2))
            error_rate = float(np.random.normal(0.03, 0.01))
            context_switches = int(np.random.poisson(1))
            response_time = float(np.random.normal(4.0, 0.5))
        elif 4 <= minute <= 7:
            # Medium focus
            wpm = float(np.random.normal(50, 6))
            backspace = float(np.random.normal(0.09, 0.02))
            pause_freq = int(np.random.poisson(5))
            error_rate = float(np.random.normal(0.08, 0.02))
            context_switches = int(np.random.poisson(3))
            response_time = float(np.random.normal(7.5, 1.0))
        elif 8 <= minute <= 11:
            # High cognitive overload (debugging hard bug)
            wpm = float(np.random.normal(24, 4))
            backspace = float(np.random.normal(0.24, 0.04))
            pause_freq = int(np.random.poisson(14))
            error_rate = float(np.random.normal(0.26, 0.04))
            context_switches = int(np.random.poisson(10))
            response_time = float(np.random.normal(16.5, 2.0))
        elif minute == 12:
            # User takes a break (Rest period)
            wpm = 10.0
            backspace = 0.02
            pause_freq = 1
            error_rate = 0.01
            context_switches = 0
            response_time = 3.0
        else:
            # Post-break recovery: Low-Med load
            wpm = float(np.random.normal(60, 5))
            backspace = float(np.random.normal(0.05, 0.01))
            pause_freq = int(np.random.poisson(3))
            error_rate = float(np.random.normal(0.04, 0.01))
            context_switches = int(np.random.poisson(2))
            response_time = float(np.random.normal(5.0, 0.8))

        feat_dict = {
            "typing_speed_wpm": max(5.0, wpm),
            "avg_keystroke_interval_ms": 60000.0 / (wpm * 5.0) if wpm > 0 else 500.0,
            "backspace_ratio": max(0.0, backspace),
            "pause_frequency_per_min": max(0, pause_freq),
            "mouse_velocity_px_s": float(np.random.normal(300, 40)),
            "mouse_distance_px": float(np.random.normal(1800, 200)),
            "click_frequency_per_min": float(np.random.normal(20, 3)),
            "idle_time_seconds": float(np.random.normal(6, 1.5)),
            "error_rate": max(0.0, error_rate),
            "response_time_seconds": max(0.5, response_time),
            "retry_count": int(np.random.poisson(2)),
            "context_switches_per_min": max(0, context_switches)
        }

        res = pred.predict(feat_dict)
        score = res["cognitive_load_score"]

        timeline.append({
            "minute": minute,
            "cognitive_load_score": score,
            "predicted_state": res["predicted_state"],
            "wpm": round(wpm, 1),
            "error_rate": round(error_rate * 100.0, 1),
            "pause_freq": pause_freq,
            "context_switches": context_switches,
            "is_break": (minute == 12)
        })

    scores = [item["cognitive_load_score"] for item in timeline]
    avg_score = int(round(np.mean(scores)))
    peak_score = int(np.max(scores))

    # Time in states
    states = [item["predicted_state"] for item in timeline]
    low_pct = round(states.count("Low") / len(states) * 100, 1)
    med_pct = round(states.count("Medium") / len(states) * 100, 1)
    high_pct = round(states.count("High") / len(states) * 100, 1)

    pre_break = timeline[10]["cognitive_load_score"] # Minute 11 (peak high)
    post_break = timeline[12]["cognitive_load_score"] # Minute 13 (post break)
    recovery_data = CognitiveLoadPredictor.calculate_recovery(pre_break, post_break)

    return {
        "session_type": session_type,
        "duration_minutes": duration_minutes,
        "average_score": avg_score,
        "peak_score": peak_score,
        "time_in_states": {
            "Low": low_pct,
            "Medium": med_pct,
            "High": high_pct
        },
        "recovery_analysis": recovery_data,
        "timeline": timeline
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
