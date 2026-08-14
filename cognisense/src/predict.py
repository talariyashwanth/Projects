"""
Inference & Explainability Engine for Cognisense
Loads trained ML models, predicts cognitive load probabilities, converts to a 0-100 Cognitive Load Score,
and generates feature attribution explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
from preprocessing import FEATURE_COLUMNS, CLASS_NAMES

class CognitiveLoadPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        
        model_path = os.path.join(model_dir, "cognitive_load_model.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or Scaler not found in {model_dir}. Please run train.py first.")

        model_artifact = joblib.load(model_path)
        self.model = model_artifact["model"]
        self.model_name = model_artifact["name"]
        self.scaler = joblib.load(scaler_path)

        # Typical population feature baselines (mean & std) for attribution calculation
        self.feature_baselines = {
            "typing_speed_wpm": {"mean": 55.0, "std": 18.0, "high_load_direction": -1},  # Lower WPM -> Higher load
            "avg_keystroke_interval_ms": {"mean": 250.0, "std": 90.0, "high_load_direction": 1},
            "backspace_ratio": {"mean": 0.08, "std": 0.06, "high_load_direction": 1},
            "pause_frequency_per_min": {"mean": 5.0, "std": 4.0, "high_load_direction": 1},
            "mouse_velocity_px_s": {"mean": 300.0, "std": 100.0, "high_load_direction": -1},
            "mouse_distance_px": {"mean": 1800.0, "std": 600.0, "high_load_direction": 1},
            "click_frequency_per_min": {"mean": 20.0, "std": 8.0, "high_load_direction": 1},
            "idle_time_seconds": {"mean": 7.0, "std": 4.5, "high_load_direction": 1},
            "error_rate": {"mean": 0.08, "std": 0.07, "high_load_direction": 1},
            "response_time_seconds": {"mean": 8.0, "std": 4.5, "high_load_direction": 1},
            "retry_count": {"mean": 2.0, "std": 2.0, "high_load_direction": 1},
            "context_switches_per_min": {"mean": 4.0, "std": 3.0, "high_load_direction": 1}
        }

    def predict(self, feature_dict: dict) -> dict:
        """
        Takes a dict of 12 behavioral features, returns class prediction, score, and feature attribution.
        """
        # Ensure all features exist with fallback defaults
        features = []
        for col in FEATURE_COLUMNS:
            val = feature_dict.get(col, self.feature_baselines[col]["mean"])
            features.append(val)

        X_raw = np.array([features])
        X_df = pd.DataFrame(X_raw, columns=FEATURE_COLUMNS)

        # Scale input
        X_scaled = self.scaler.transform(X_df)

        # Check if model requires scaled input
        if self.model_name == "Logistic Regression":
            X_input = X_scaled
        else:
            X_input = X_df

        probs = self.model.predict_proba(X_input)[0]
        prob_low, prob_med, prob_high = float(probs[0]), float(probs[1]), float(probs[2])

        pred_class_idx = int(np.argmax(probs))
        predicted_state = CLASS_NAMES[pred_class_idx]

        # Calculate Cognitive Load Score (0 - 100)
        # Weighted expectation: 0*prob_low + 50*prob_med + 100*prob_high
        load_score = int(round(prob_low * 15.0 + prob_med * 52.5 + prob_high * 92.5))
        load_score = max(0, min(100, load_score))

        # Determine top contributing signals (explainability)
        attributions = []
        for col in FEATURE_COLUMNS:
            val = feature_dict.get(col, self.feature_baselines[col]["mean"])
            base = self.feature_baselines[col]
            z_score = (val - base["mean"]) / (base["std"] if base["std"] > 0 else 1.0)
            
            # Impact direction towards high load
            impact = z_score * base["high_load_direction"]
            attributions.append({
                "feature": col,
                "display_name": col.replace("_", " ").title(),
                "value": round(float(val), 2),
                "z_score": round(float(z_score), 2),
                "impact_score": round(float(impact), 2),
                "is_elevating_load": impact > 0.3
            })

        # Sort by impact score descending
        attributions = sorted(attributions, key=lambda x: abs(x["impact_score"]), reverse=True)
        top_drivers = attributions[:4]

        return {
            "predicted_state": predicted_state,
            "cognitive_load_score": load_score,
            "probabilities": {
                "Low": round(prob_low, 4),
                "Medium": round(prob_med, 4),
                "High": round(prob_high, 4)
            },
            "model_used": self.model_name,
            "top_contributing_signals": top_drivers,
            "all_attributions": attributions
        }

    @staticmethod
    def calculate_recovery(pre_break_score: int, post_break_score: int) -> dict:
        """Computes recovery points after a rest break."""
        recovery_delta = pre_break_score - post_break_score
        is_recovered = recovery_delta > 10
        return {
            "pre_break_score": pre_break_score,
            "post_break_score": post_break_score,
            "recovery_points": recovery_delta,
            "is_recovered": is_recovered,
            "message": f"Observed behavioral load decreased by {recovery_delta} points post-break." if is_recovered else "Minor or no behavioral recovery detected."
        }

if __name__ == "__main__":
    predictor = CognitiveLoadPredictor()
    sample_input = {
        "typing_speed_wpm": 28.5,
        "avg_keystroke_interval_ms": 390.0,
        "backspace_ratio": 0.21,
        "pause_frequency_per_min": 11,
        "mouse_velocity_px_s": 170.0,
        "mouse_distance_px": 2400.0,
        "click_frequency_per_min": 30.0,
        "idle_time_seconds": 12.0,
        "error_rate": 0.22,
        "response_time_seconds": 14.5,
        "retry_count": 4,
        "context_switches_per_min": 8
    }
    res = predictor.predict(sample_input)
    print("Prediction Output:", json.dumps(res, indent=2))
