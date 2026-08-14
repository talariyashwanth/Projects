"""
Inference & Explainability Engine for Cognisense

Loads the trained model, produces a class prediction with calibrated
probabilities, converts those into a 0-100 Cognitive Load Score, and
generates feature attributions explaining the prediction.

Interpretation caveat (PRD 16 / PRD 30): attributions describe which
observed behavioural signals deviated from the reference distribution in the
direction the model associates with higher load. They are NOT causal claims
about a person's mental state.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from preprocessing import FEATURE_COLUMNS, CLASS_NAMES

# Score bands (PRD 17). A product visualisation convention, not a
# clinically validated scale.
BAND_LOW_MAX = 35
BAND_MED_MAX = 70

# Representative score anchors per class used to map probabilities to a score.
SCORE_ANCHOR_LOW = 15.0
SCORE_ANCHOR_MED = 52.5
SCORE_ANCHOR_HIGH = 92.5

# |z| above which a deviation is called out as materially elevating load.
# 0.75 sigma is used rather than a smaller value because roughly 45% of normal
# observations fall within +/-0.75 sigma of the reference mean; a lower bar
# flags ordinary fluctuation as a load signal and makes explanations noisy.
ATTRIBUTION_Z_THRESHOLD = 0.75

# Fallback population statistics, used only if models/feature_baselines.json
# is absent (i.e. an older training run). Training regenerates the real ones.
FALLBACK_BASELINES = {
    "typing_speed_wpm":          {"mean": 48.0,  "std": 16.0, "high_load_direction": -1},
    "avg_keystroke_interval_ms": {"mean": 260.0, "std": 75.0, "high_load_direction": 1},
    "backspace_ratio":           {"mean": 0.10,  "std": 0.05, "high_load_direction": 1},
    "pause_frequency_per_min":   {"mean": 5.5,   "std": 3.5,  "high_load_direction": 1},
    "mouse_velocity_px_s":       {"mean": 300.0, "std": 95.0, "high_load_direction": -1},
    "mouse_distance_px":         {"mean": 1800.0, "std": 480.0, "high_load_direction": 1},
    "click_frequency_per_min":   {"mean": 21.0,  "std": 7.0,  "high_load_direction": 1},
    "idle_time_seconds":         {"mean": 7.5,   "std": 4.0,  "high_load_direction": 1},
    "error_rate":                {"mean": 0.09,  "std": 0.06, "high_load_direction": 1},
    "response_time_seconds":     {"mean": 7.8,   "std": 4.0,  "high_load_direction": 1},
    "retry_count":               {"mean": 1.8,   "std": 1.7,  "high_load_direction": 1},
    "context_switches_per_min":  {"mean": 3.8,   "std": 2.6,  "high_load_direction": 1},
}

DISPLAY_NAMES = {
    "typing_speed_wpm": "Typing Speed",
    "avg_keystroke_interval_ms": "Keystroke Interval",
    "backspace_ratio": "Backspace Ratio",
    "pause_frequency_per_min": "Pause Frequency",
    "mouse_velocity_px_s": "Mouse Velocity",
    "mouse_distance_px": "Mouse Distance",
    "click_frequency_per_min": "Click Frequency",
    "idle_time_seconds": "Idle Time",
    "error_rate": "Error Rate",
    "response_time_seconds": "Response Time",
    "retry_count": "Retry Count",
    "context_switches_per_min": "Context Switches",
}


def score_to_band(score):
    """Maps a 0-100 cognitive load score to its product-convention band."""
    if score <= BAND_LOW_MAX:
        return "Low"
    if score <= BAND_MED_MAX:
        return "Medium"
    return "High"


class CognitiveLoadPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models"))

        model_path = os.path.join(model_dir, "cognitive_load_model.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model or scaler not found in {model_dir}. Run src/train.py first.")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.model_name = artifact["name"]
        self.scaler = joblib.load(scaler_path)

        # Models that consume standardised features.
        self.needs_scaling = self.model_name in ("Logistic Regression",)

        # Population reference statistics, preferring those exported by training.
        baselines_path = os.path.join(model_dir, "feature_baselines.json")
        if os.path.exists(baselines_path):
            with open(baselines_path, "r") as f:
                self.feature_baselines = json.load(f)
        else:
            self.feature_baselines = FALLBACK_BASELINES

    # ------------------------------------------------------------------ core
    def predict(self, feature_dict: dict, personal_baseline: dict = None) -> dict:
        """
        Predict cognitive load from a 12-feature behavioural vector.

        Args:
            feature_dict: behavioural features; missing keys fall back to the
                population mean for that channel.
            personal_baseline: optional per-user reference statistics (same
                shape as feature_baselines). When supplied, attributions are
                computed relative to THIS user's normal behaviour rather than
                the population -- see PRD 23. A 45 WPM typist is not flagged
                as overloaded simply for typing 45 WPM.

        Returns:
            Prediction, score, probabilities, and ranked feature attributions.
        """
        reference = personal_baseline or self.feature_baselines

        values = []
        for col in FEATURE_COLUMNS:
            fallback = self.feature_baselines.get(col, {}).get("mean", 0.0)
            values.append(float(feature_dict.get(col, fallback)))

        X = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        X_input = self.scaler.transform(X) if self.needs_scaling else X

        probs = self.model.predict_proba(X_input)[0]
        prob_low, prob_med, prob_high = (float(probs[0]), float(probs[1]), float(probs[2]))

        predicted_state = CLASS_NAMES[int(np.argmax(probs))]

        # Probability-weighted score keeps the gauge continuous and stable
        # instead of jumping discretely when the argmax flips.
        load_score = int(round(
            prob_low * SCORE_ANCHOR_LOW
            + prob_med * SCORE_ANCHOR_MED
            + prob_high * SCORE_ANCHOR_HIGH
        ))
        load_score = max(0, min(100, load_score))

        # ------------------------------------------------------ attributions
        attributions = []
        for col in FEATURE_COLUMNS:
            fallback = self.feature_baselines.get(col, {})
            ref = reference.get(col, fallback)

            value = float(feature_dict.get(col, ref.get("mean", 0.0)))
            mean = float(ref.get("mean", 0.0))
            std = float(ref.get("std", 1.0)) or 1.0
            direction = int(fallback.get("high_load_direction",
                                         ref.get("high_load_direction", 1)))

            z = (value - mean) / std
            impact = z * direction

            attributions.append({
                "feature": col,
                "display_name": DISPLAY_NAMES.get(col, col.replace("_", " ").title()),
                "value": round(value, 2),
                "baseline_mean": round(mean, 2),
                "z_score": round(float(z), 2),
                "impact_score": round(float(impact), 2),
                "is_elevating_load": impact > ATTRIBUTION_Z_THRESHOLD,
                "direction_label": "above typical" if z > 0 else "below typical",
            })

        attributions.sort(key=lambda d: abs(d["impact_score"]), reverse=True)

        return {
            "predicted_state": predicted_state,
            "cognitive_load_score": load_score,
            "score_band": score_to_band(load_score),
            "probabilities": {
                "Low": round(prob_low, 4),
                "Medium": round(prob_med, 4),
                "High": round(prob_high, 4),
            },
            "model_used": self.model_name,
            "personalized": personal_baseline is not None,
            "top_contributing_signals": attributions[:4],
            "all_attributions": attributions,
            "disclaimer": (
                "Estimated cognitive-load state inferred from behavioural "
                "interaction patterns. Not a clinical or psychological "
                "diagnosis."
            ),
        }

    # --------------------------------------------------------- personalisation
    @staticmethod
    def build_personal_baseline(calibration_windows: list) -> dict:
        """
        Build a personal behavioural baseline from calibration windows (PRD 23).

        Each window is a feature dict recorded while the user works at a
        comfortable, self-reported normal load. Per-channel mean and standard
        deviation become that individual's reference point, so later sessions
        are scored as deviation from their own norm.
        """
        if not calibration_windows:
            raise ValueError("At least one calibration window is required.")

        df = pd.DataFrame(calibration_windows)
        baseline = {}

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            mean = float(series.mean())
            # With few windows the sample std is unstable; floor it at 10% of
            # the mean so a single feature cannot produce absurd z-scores.
            std = float(series.std()) if len(series) > 1 else 0.0
            std = max(std, abs(mean) * 0.10, 1e-6)

            baseline[col] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "n_windows": int(len(series)),
            }

        return baseline

    @staticmethod
    def calculate_recovery(pre_break_score: int, post_break_score: int,
                           threshold: int = 10) -> dict:
        """
        Compare load scores before and after a rest break.

        Caveat (PRD 22): a lower score after a break is an OBSERVED
        BEHAVIOURAL CHANGE. It is not proof that psychological cognitive load
        objectively recovered.
        """
        delta = int(pre_break_score) - int(post_break_score)
        recovered = delta > threshold

        return {
            "pre_break_score": int(pre_break_score),
            "post_break_score": int(post_break_score),
            "recovery_points": delta,
            "is_recovered": recovered,
            "message": (
                f"Observed behavioural load score decreased by {delta} points "
                f"after the break."
                if recovered else
                "No substantial change in behavioural load score after the break."
            ),
        }


if __name__ == "__main__":
    predictor = CognitiveLoadPredictor()

    sample = {
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
        "context_switches_per_min": 8,
    }

    result = predictor.predict(sample)
    print(f"Model            : {result['model_used']}")
    print(f"Predicted state  : {result['predicted_state']}")
    print(f"Load score       : {result['cognitive_load_score']}/100 "
          f"({result['score_band']})")
    print("\nTop contributing signals:")
    for signal in result["top_contributing_signals"]:
        arrow = "^" if signal["z_score"] > 0 else "v"
        print(f"  {arrow} {signal['display_name']:<22} {signal['value']:>8} "
              f"(z={signal['z_score']:+.2f})")

    # Demonstrate personalisation: a naturally slow typist working normally.
    print("\n--- Personalised baseline demo (naturally slow typist) ---")
    calibration = [dict(sample, typing_speed_wpm=30.0 + i, error_rate=0.05,
                        response_time_seconds=6.0, pause_frequency_per_min=3,
                        backspace_ratio=0.06) for i in range(5)]
    personal = CognitiveLoadPredictor.build_personal_baseline(calibration)
    personalised = predictor.predict(sample, personal_baseline=personal)
    print(f"Population-relative top signal : "
          f"{result['top_contributing_signals'][0]['display_name']}")
    print(f"Personal-relative top signal   : "
          f"{personalised['top_contributing_signals'][0]['display_name']}")
