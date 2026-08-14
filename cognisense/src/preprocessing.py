"""
Preprocessing Module for Cognisense
Handles data cleaning, feature validation, splitting, and scaling.

Note on splitting strategy: when the dataset carries a `subject_id`, the
train/test split is SUBJECT-AWARE (GroupShuffleSplit). Behavioural windows
from one person are highly correlated, so allowing the same subject into
both train and test leaks individual baselines and inflates scores. Held-out
subjects measure what we actually care about: generalisation to a NEW person.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit

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
GROUP_COLUMN = "subject_id"
CLASS_NAMES = ["Low", "Medium", "High"]

# Physically realistic bounds used for sanity clipping.
CLIP_BOUNDS = {
    "typing_speed_wpm": (0, 200),
    "avg_keystroke_interval_ms": (10, 2000),
    "backspace_ratio": (0.0, 1.0),
    "pause_frequency_per_min": (0, 60),
    "mouse_velocity_px_s": (0, 5000),
    "mouse_distance_px": (0, 50000),
    "click_frequency_per_min": (0, 200),
    "idle_time_seconds": (0, 300),
    "error_rate": (0.0, 1.0),
    "response_time_seconds": (0, 120),
    "retry_count": (0, 50),
    "context_switches_per_min": (0, 60),
}


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validates schema, drops nulls, clips invalid numeric bounds."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    for column, (low, high) in CLIP_BOUNDS.items():
        df_clean[column] = df_clean[column].clip(low, high)

    return df_clean


def prepare_train_test_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Splits features X and target y, then fits a StandardScaler on train only.

    Uses a subject-aware split when `subject_id` is present so that no
    individual appears in both train and test; falls back to a stratified
    random split otherwise.
    """
    df_clean = clean_and_validate_data(df)

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    if GROUP_COLUMN in df_clean.columns and df_clean[GROUP_COLUMN].nunique() > 1:
        groups = df_clean[GROUP_COLUMN]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        split_strategy = "subject-aware (GroupShuffleSplit, held-out subjects)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        groups_train = None
        split_strategy = "stratified random split"

    # Scaler is fit on training data only -- never on the test set.
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
        "groups_train": groups_train,
        "scaler": scaler,
        "split_strategy": split_strategy,
    }
