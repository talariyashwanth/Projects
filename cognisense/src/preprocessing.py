"""
Preprocessing Module for Cognisense
Handles data cleaning, feature validation, splitting, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "typing_speed_wpm",
    "avg_keystroke_interval_ms",
    "backspace_ratio",
    "pause_frequency_per_min",
    "mouse_velocity_px_s",
    "mouse_distance_px",
    "click_frequency_per_min",
    "idle_time_seconds",
    "error_rate",
    "response_time_seconds",
    "retry_count",
    "context_switches_per_min"
]

TARGET_COLUMN = "cognitive_load"
CLASS_NAMES = ["Low", "Medium", "High"]

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates schema, drops nulls, clips invalid numeric bounds."""
    df_clean = df.copy()
    
    # Fill or drop missing values
    df_clean = df_clean.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    
    # Clip numeric bounds for sanity
    df_clean["typing_speed_wpm"] = df_clean["typing_speed_wpm"].clip(0, 200)
    df_clean["avg_keystroke_interval_ms"] = df_clean["avg_keystroke_interval_ms"].clip(10, 2000)
    df_clean["backspace_ratio"] = df_clean["backspace_ratio"].clip(0.0, 1.0)
    df_clean["pause_frequency_per_min"] = df_clean["pause_frequency_per_min"].clip(0, 60)
    df_clean["mouse_velocity_px_s"] = df_clean["mouse_velocity_px_s"].clip(0, 5000)
    df_clean["mouse_distance_px"] = df_clean["mouse_distance_px"].clip(0, 50000)
    df_clean["click_frequency_per_min"] = df_clean["click_frequency_per_min"].clip(0, 200)
    df_clean["idle_time_seconds"] = df_clean["idle_time_seconds"].clip(0, 300)
    df_clean["error_rate"] = df_clean["error_rate"].clip(0.0, 1.0)
    df_clean["response_time_seconds"] = df_clean["response_time_seconds"].clip(0, 120)
    df_clean["retry_count"] = df_clean["retry_count"].clip(0, 50)
    df_clean["context_switches_per_min"] = df_clean["context_switches_per_min"].clip(0, 60)
    
    return df_clean

def prepare_train_test_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Splits features X and target y with stratification, fits StandardScaler."""
    df_clean = clean_and_validate_data(df)
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }
