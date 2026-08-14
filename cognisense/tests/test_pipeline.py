"""
Unit & Integration Tests for Cognisense ML Pipeline
"""

import os
import sys
import pytest
import pandas as pd

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset_generator import generate_cognitive_load_dataset
from preprocessing import clean_and_validate_data, prepare_train_test_data, FEATURE_COLUMNS
from feature_engineering import extract_features_from_events

def test_dataset_generator():
    df = generate_cognitive_load_dataset(n_samples_per_class=100)
    assert len(df) == 300
    for col in FEATURE_COLUMNS + ["cognitive_load"]:
        assert col in df.columns
    assert set(df["cognitive_load"].unique()) == {0, 1, 2}

def test_preprocessing():
    df = generate_cognitive_load_dataset(n_samples_per_class=50)
    df_clean = clean_and_validate_data(df)
    assert len(df_clean) > 0
    
    split_res = prepare_train_test_data(df_clean, test_size=0.2)
    assert len(split_res["X_train"]) == 120
    assert len(split_res["X_test"]) == 30
    assert split_res["X_train_scaled"].shape == (120, len(FEATURE_COLUMNS))

def test_feature_engineering_extraction():
    raw_window = {
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
        "context_switches": 2
    }
    feats = extract_features_from_events(raw_window)
    assert "typing_speed_wpm" in feats
    assert feats["typing_speed_wpm"] == 8.0  # 4 words in 0.5 mins = 8 WPM
    assert feats["backspace_ratio"] == 0.1  # 2 backspaces / 20 total = 0.1
    assert feats["pause_frequency_per_min"] == 2  # 1 pause in 0.5 mins = 2/min
