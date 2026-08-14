"""
Unit & integration tests for the Cognisense ML pipeline.

Coverage:
  - synthetic data generation, including the realism properties the
    generator is specifically designed to produce
  - preprocessing, cleaning, and subject-aware splitting (no leakage)
  - feature engineering arithmetic
  - inference, scoring, explainability, personalisation, recovery
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_generator import generate_cognitive_load_dataset
from preprocessing import (
    clean_and_validate_data, prepare_train_test_data,
    FEATURE_COLUMNS, GROUP_COLUMN, CLASS_NAMES,
)
from feature_engineering import extract_features_from_events
from predict import CognitiveLoadPredictor, score_to_band

MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_EXISTS = os.path.exists(os.path.join(MODELS_DIR, "cognitive_load_model.pkl"))
requires_model = pytest.mark.skipif(
    not MODEL_EXISTS, reason="Trained model not found; run src/train.py first.")


# --------------------------------------------------------------- dataset
def test_dataset_shape_and_schema():
    df = generate_cognitive_load_dataset(n_samples_per_class=100)
    assert len(df) == 300
    for col in FEATURE_COLUMNS + ["cognitive_load", GROUP_COLUMN]:
        assert col in df.columns
    assert set(df["cognitive_load"].unique()) == {0, 1, 2}


def test_dataset_is_class_balanced():
    df = generate_cognitive_load_dataset(n_samples_per_class=150)
    counts = df["cognitive_load"].value_counts()
    assert counts.nunique() == 1, "Classes should be balanced"


def test_dataset_has_multiple_subjects():
    """Individual differences require more than one simulated person."""
    df = generate_cognitive_load_dataset(n_samples_per_class=200)
    assert df[GROUP_COLUMN].nunique() > 10


def test_dataset_features_stay_in_physical_bounds():
    df = generate_cognitive_load_dataset(n_samples_per_class=200)
    assert df["typing_speed_wpm"].between(0, 200).all()
    assert df["error_rate"].between(0.0, 1.0).all()
    assert df["backspace_ratio"].between(0.0, 1.0).all()
    assert (df["pause_frequency_per_min"] >= 0).all()
    assert (df["retry_count"] >= 0).all()


def test_classes_genuinely_overlap():
    """
    Guards the core data-integrity property: classes must NOT be trivially
    separable. If within-class spread collapses relative to the gap between
    class means, the dataset has become unrealistically clean and reported
    accuracy stops being meaningful.
    """
    df = generate_cognitive_load_dataset(n_samples_per_class=400)
    grouped = df.groupby("cognitive_load")["typing_speed_wpm"]
    means, stds = grouped.mean(), grouped.std()

    gap = abs(means[2] - means[0]) / 2.0
    pooled_std = float(stds.mean())

    # A separation of >3 pooled std devs would mean near-perfect separability.
    assert gap / pooled_std < 3.0, (
        f"Classes are too separable (gap/std={gap / pooled_std:.2f}); "
        "the generator has lost its realistic overlap.")


def test_dataset_is_reproducible():
    a = generate_cognitive_load_dataset(n_samples_per_class=50, random_state=7)
    b = generate_cognitive_load_dataset(n_samples_per_class=50, random_state=7)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------- preprocessing
def test_clean_and_validate_drops_nulls():
    df = generate_cognitive_load_dataset(n_samples_per_class=50)
    df.loc[0, "typing_speed_wpm"] = np.nan
    cleaned = clean_and_validate_data(df)
    assert len(cleaned) == len(df) - 1


def test_clean_and_validate_rejects_missing_columns():
    df = generate_cognitive_load_dataset(n_samples_per_class=20)
    with pytest.raises(ValueError, match="missing required columns"):
        clean_and_validate_data(df.drop(columns=["error_rate"]))


def test_clean_and_validate_clips_out_of_range():
    df = generate_cognitive_load_dataset(n_samples_per_class=30)
    df.loc[0, "error_rate"] = 9.9
    cleaned = clean_and_validate_data(df)
    assert cleaned["error_rate"].max() <= 1.0


def test_split_shapes_and_scaling():
    df = generate_cognitive_load_dataset(n_samples_per_class=100)
    result = prepare_train_test_data(df, test_size=0.2)

    n_total = len(result["X_train"]) + len(result["X_test"])
    assert n_total == len(clean_and_validate_data(df))
    assert result["X_train_scaled"].shape == (len(result["X_train"]), len(FEATURE_COLUMNS))
    assert result["X_test_scaled"].shape == (len(result["X_test"]), len(FEATURE_COLUMNS))

    # Scaler must be fit on train only -> train is ~zero-mean/unit-variance.
    assert np.allclose(result["X_train_scaled"].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(result["X_train_scaled"].std(axis=0), 1, atol=1e-6)


def test_split_is_subject_aware_no_leakage():
    """The same subject must never appear in both train and test."""
    df = generate_cognitive_load_dataset(n_samples_per_class=200)
    result = prepare_train_test_data(df, test_size=0.25)

    assert "subject-aware" in result["split_strategy"]

    cleaned = clean_and_validate_data(df)
    train_subjects = set(cleaned.loc[result["X_train"].index, GROUP_COLUMN])
    test_subjects = set(cleaned.loc[result["X_test"].index, GROUP_COLUMN])

    assert train_subjects & test_subjects == set(), "Subject leaked across the split"


def test_split_falls_back_without_subject_ids():
    df = generate_cognitive_load_dataset(n_samples_per_class=80).drop(columns=[GROUP_COLUMN])
    result = prepare_train_test_data(df, test_size=0.2)
    assert "stratified" in result["split_strategy"]


# ---------------------------------------------------- feature engineering
def test_feature_engineering_arithmetic():
    window = {
        "duration_seconds": 30.0,
        "keystroke_timestamps": [0.1, 0.4, 0.7, 1.0, 1.3],
        "backspace_count": 2,
        "total_keystrokes": 20,
        "words_typed": 4,
        "mouse_positions": [{"x": 10, "y": 10, "t": 0.0}, {"x": 100, "y": 100, "t": 1.0}],
        "click_count": 2,
        "idle_periods_seconds": [3.5],
        "error_count": 1,
        "attempt_count": 2,
        "response_time_sec": 5.0,
        "retry_count": 1,
        "context_switches": 2,
    }
    feats = extract_features_from_events(window)

    assert feats["typing_speed_wpm"] == 8.0      # 4 words / 0.5 min
    assert feats["backspace_ratio"] == 0.1       # 2 / 20
    assert feats["pause_frequency_per_min"] == 2  # 1 pause / 0.5 min
    assert feats["error_rate"] == 0.5            # 1 / 2
    assert feats["click_frequency_per_min"] == 4.0  # 2 clicks / 0.5 min

    # Mouse travelled (90,90) over 1s -> ~127px at ~127px/s.
    assert feats["mouse_distance_px"] == pytest.approx(127.28, abs=0.1)
    assert feats["mouse_velocity_px_s"] == pytest.approx(127.28, abs=0.1)


def test_feature_engineering_returns_all_features():
    feats = extract_features_from_events({"duration_seconds": 10.0})
    for col in FEATURE_COLUMNS:
        assert col in feats


def test_feature_engineering_handles_empty_window():
    """An empty window must not divide by zero or crash."""
    feats = extract_features_from_events({})
    assert feats["typing_speed_wpm"] == 0.0
    assert feats["backspace_ratio"] == 0.0
    assert all(isinstance(v, (int, float)) for v in feats.values())


# ------------------------------------------------------- score banding
@pytest.mark.parametrize("score,expected", [
    (0, "Low"), (35, "Low"), (36, "Medium"),
    (70, "Medium"), (71, "High"), (100, "High"),
])
def test_score_to_band(score, expected):
    assert score_to_band(score) == expected


# ------------------------------------------------------------- inference
@requires_model
def test_predictor_output_contract():
    predictor = CognitiveLoadPredictor()
    features = {col: predictor.feature_baselines[col]["mean"] for col in FEATURE_COLUMNS}
    result = predictor.predict(features)

    assert result["predicted_state"] in CLASS_NAMES
    assert 0 <= result["cognitive_load_score"] <= 100
    assert result["score_band"] in CLASS_NAMES
    assert len(result["all_attributions"]) == len(FEATURE_COLUMNS)
    assert len(result["top_contributing_signals"]) == 4
    assert result["probabilities"]["Low"] + result["probabilities"]["Medium"] \
        + result["probabilities"]["High"] == pytest.approx(1.0, abs=1e-3)
    assert "not a clinical" in result["disclaimer"].lower()


@requires_model
def test_high_load_profile_scores_above_low_load_profile():
    """Directional sanity: strained behaviour must score higher than fluent."""
    predictor = CognitiveLoadPredictor()

    low = {
        "typing_speed_wpm": 72.0, "avg_keystroke_interval_ms": 165.0,
        "backspace_ratio": 0.02, "pause_frequency_per_min": 1,
        "mouse_velocity_px_s": 430.0, "mouse_distance_px": 1050.0,
        "click_frequency_per_min": 13.0, "idle_time_seconds": 2.5,
        "error_rate": 0.02, "response_time_seconds": 3.5,
        "retry_count": 0, "context_switches_per_min": 1,
    }
    high = {
        "typing_speed_wpm": 22.0, "avg_keystroke_interval_ms": 460.0,
        "backspace_ratio": 0.27, "pause_frequency_per_min": 15,
        "mouse_velocity_px_s": 150.0, "mouse_distance_px": 2800.0,
        "click_frequency_per_min": 34.0, "idle_time_seconds": 16.0,
        "error_rate": 0.28, "response_time_seconds": 18.0,
        "retry_count": 6, "context_switches_per_min": 11,
    }

    assert (predictor.predict(high)["cognitive_load_score"]
            > predictor.predict(low)["cognitive_load_score"])


@requires_model
def test_attributions_are_sorted_by_absolute_impact():
    predictor = CognitiveLoadPredictor()
    features = {col: predictor.feature_baselines[col]["mean"] for col in FEATURE_COLUMNS}
    features["error_rate"] = 0.45

    attributions = predictor.predict(features)["all_attributions"]
    impacts = [abs(a["impact_score"]) for a in attributions]
    assert impacts == sorted(impacts, reverse=True)
    assert attributions[0]["feature"] == "error_rate"


@requires_model
def test_missing_features_fall_back_to_baseline():
    predictor = CognitiveLoadPredictor()
    result = predictor.predict({"error_rate": 0.3})
    assert 0 <= result["cognitive_load_score"] <= 100


# --------------------------------------------------------- personalisation
def test_build_personal_baseline():
    windows = [
        {"typing_speed_wpm": 40.0 + i, "error_rate": 0.05, "response_time_seconds": 6.0}
        for i in range(5)
    ]
    baseline = CognitiveLoadPredictor.build_personal_baseline(windows)

    assert baseline["typing_speed_wpm"]["mean"] == pytest.approx(42.0)
    assert baseline["typing_speed_wpm"]["n_windows"] == 5
    # std is floored so a low-variance channel can't blow up z-scores.
    assert baseline["error_rate"]["std"] > 0


def test_build_personal_baseline_rejects_empty():
    with pytest.raises(ValueError):
        CognitiveLoadPredictor.build_personal_baseline([])


@requires_model
def test_personal_baseline_changes_attribution_for_slow_typist():
    """
    A naturally slow typist working at their own normal pace should not have
    typing speed flagged as elevating load once personalised (PRD 23).
    """
    predictor = CognitiveLoadPredictor()

    calibration = [
        {**{col: predictor.feature_baselines[col]["mean"] for col in FEATURE_COLUMNS},
         "typing_speed_wpm": 29.0 + i}
        for i in range(6)
    ]
    personal = CognitiveLoadPredictor.build_personal_baseline(calibration)

    current = {**{col: predictor.feature_baselines[col]["mean"] for col in FEATURE_COLUMNS},
               "typing_speed_wpm": 30.0}

    population_attr = {a["feature"]: a for a in predictor.predict(current)["all_attributions"]}
    personal_attr = {a["feature"]: a for a in
                     predictor.predict(current, personal_baseline=personal)["all_attributions"]}

    # Against the population this looks slow; against their own norm it doesn't.
    assert population_attr["typing_speed_wpm"]["is_elevating_load"]
    assert not personal_attr["typing_speed_wpm"]["is_elevating_load"]


# -------------------------------------------------------------- recovery
def test_recovery_detected():
    result = CognitiveLoadPredictor.calculate_recovery(86, 52)
    assert result["recovery_points"] == 34
    assert result["is_recovered"]


def test_recovery_not_detected_for_small_change():
    result = CognitiveLoadPredictor.calculate_recovery(60, 56)
    assert not result["is_recovered"]


def test_recovery_wording_avoids_clinical_claims():
    message = CognitiveLoadPredictor.calculate_recovery(90, 40)["message"]
    assert "behavioural" in message.lower()


# -------------------------------------------------------- API contract
# These guard the frontend/backend boundary. A real bug lived here: the browser
# sent attempt_count=0 for an untouched session, the API required >= 1, and every
# live prediction failed with a silent 422 that the UI swallowed.
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    from backend.main import app
    return TestClient(app)


def test_raw_events_accepts_a_realistic_typing_window(client):
    payload = {
        "duration_seconds": 20.0,
        "keystroke_timestamps": [1.0, 1.2, 1.5, 2.0, 2.4, 3.1],
        "backspace_count": 2,
        "total_keystrokes": 84,
        "words_typed": 12,
        "mouse_positions": [],
        "click_count": 0,
        "idle_periods_seconds": [],
        "error_count": 2,
        "attempt_count": 84,
        "response_time_sec": 1.7,
        "retry_count": 0,
        "context_switches": 0,
    }
    res = client.post("/api/predict-raw-events", json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["predicted_state"] in CLASS_NAMES
    assert 0 <= body["cognitive_load_score"] <= 100
    assert "extracted_features" in body


def test_raw_events_rejects_zero_attempt_count(client):
    """
    attempt_count is the denominator of error_rate, so 0 is genuinely invalid.
    The contract is asserted here so the frontend keeps clamping it to >= 1
    instead of silently sending 0 and losing every live prediction.
    """
    payload = {
        "duration_seconds": 5.0,
        "keystroke_timestamps": [],
        "backspace_count": 0,
        "total_keystrokes": 0,
        "words_typed": 0,
        "mouse_positions": [],
        "click_count": 0,
        "idle_periods_seconds": [],
        "error_count": 0,
        "attempt_count": 0,
        "response_time_sec": 4.0,
        "retry_count": 0,
        "context_switches": 0,
    }
    assert client.post("/api/predict-raw-events", json=payload).status_code == 422


def test_predict_returns_probabilities_that_sum_to_one(client):
    res = client.post("/api/predict", json={"typing_speed_wpm": 24,
                                            "error_rate": 0.25})
    assert res.status_code == 200, res.text
    probs = res.json()["probabilities"]
    assert set(probs) == set(CLASS_NAMES)
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_model_insights_exposes_honesty_metadata(client):
    """The UI renders these fields; missing ones would silently blank the page."""
    body = client.get("/api/model-insights").json()
    comparison = body["comparison"]

    for field in ("best_model", "models", "split_strategy",
                  "cv_strategy", "data_provenance"):
        assert field in comparison, f"missing {field}"

    assert "synthetic" in comparison["data_provenance"].lower()
    assert body["importance_method"]

    for name, model in comparison["models"].items():
        assert "cv_f1_mean" in model, f"{name} missing cv_f1_mean"
        assert len(model["confusion_matrix"]) == 3


def test_simulate_session_fields_used_by_the_ui(client):
    body = client.post("/api/simulate-session"
                       "?duration_minutes=15&session_type=coding_challenge").json()
    for field in ("average_score", "average_band", "peak_score", "peak_minute",
                  "duration_minutes", "time_in_states", "timeline"):
        assert field in body, f"missing {field}"

    assert body["duration_minutes"] == 15
    assert len(body["timeline"]) == 15
    assert body["average_band"] in CLASS_NAMES
    # peak_minute must point at the actual maximum in the timeline.
    assert body["peak_score"] == max(p["cognitive_load_score"] for p in body["timeline"])

