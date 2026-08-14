"""
Feature Engineering Module for Cognisense
Extracts structured behavioral features from raw interaction events (keystrokes, mouse, pauses, task events).
"""

import numpy as np

def extract_features_from_events(events_window: dict) -> dict:
    """
    Computes a standard 12-feature vector from raw session activity logs.
    
    events_window expected format:
    {
        "duration_seconds": 30.0,
        "keystroke_timestamps": [0.1, 0.35, 0.52, ...],
        "backspace_count": 4,
        "total_keystrokes": 45,
        "words_typed": 9,
        "mouse_positions": [{"x": 100, "y": 200, "t": 0.1}, ...],
        "click_count": 3,
        "idle_periods_seconds": [2.5, 4.1],
        "error_count": 1,
        "attempt_count": 5,
        "response_time_sec": 6.5,
        "retry_count": 1,
        "context_switches": 2
    }
    """
    duration = max(events_window.get("duration_seconds", 30.0), 1.0)
    minutes = duration / 60.0

    # Typing WPM
    words = events_window.get("words_typed", 0)
    wpm = (words / minutes) if minutes > 0 else 0.0

    # Keystroke intervals
    ks_times = events_window.get("keystroke_timestamps", [])
    if len(ks_times) > 1:
        intervals = [ (ks_times[i] - ks_times[i-1]) * 1000.0 for i in range(1, len(ks_times)) ]
        avg_interval = float(np.mean(intervals))
    else:
        avg_interval = 250.0  # Default baseline

    # Backspace ratio
    total_ks = max(events_window.get("total_keystrokes", 0), 1)
    backspaces = events_window.get("backspace_count", 0)
    backspace_ratio = float(backspaces / total_ks)

    # Pause frequency (> 2 seconds)
    pauses = [t for t in events_window.get("idle_periods_seconds", []) if t >= 2.0]
    pause_freq = int(len(pauses) / minutes) if minutes > 0 else len(pauses)

    # Mouse dynamics
    mouse_pos = events_window.get("mouse_positions", [])
    total_dist = 0.0
    velocities = []
    if len(mouse_pos) > 1:
        for i in range(1, len(mouse_pos)):
            dx = mouse_pos[i]["x"] - mouse_pos[i-1]["x"]
            dy = mouse_pos[i]["y"] - mouse_pos[i-1]["y"]
            dt = mouse_pos[i]["t"] - mouse_pos[i-1]["t"]
            dist = np.sqrt(dx*dx + dy*dy)
            total_dist += dist
            if dt > 0:
                velocities.append(dist / dt)

    avg_mouse_vel = float(np.mean(velocities)) if velocities else 250.0

    # Click frequency
    clicks = events_window.get("click_count", 0)
    click_freq = float(clicks / minutes) if minutes > 0 else float(clicks)

    # Idle time total
    idle_total = float(sum(events_window.get("idle_periods_seconds", [0.0])))

    # Task metrics
    attempts = max(events_window.get("attempt_count", 1), 1)
    errors = events_window.get("error_count", 0)
    error_rate = float(errors / attempts)

    response_time = float(events_window.get("response_time_sec", 5.0))
    retries = int(events_window.get("retry_count", 0))
    ctx_switches = int(events_window.get("context_switches", 0))
    ctx_switches_per_min = int(ctx_switches / minutes) if minutes > 0 else ctx_switches

    return {
        "typing_speed_wpm": float(np.round(wpm, 2)),
        "avg_keystroke_interval_ms": float(np.round(avg_interval, 2)),
        "backspace_ratio": float(np.round(backspace_ratio, 4)),
        "pause_frequency_per_min": int(pause_freq),
        "mouse_velocity_px_s": float(np.round(avg_mouse_vel, 2)),
        "mouse_distance_px": float(np.round(total_dist, 2)),
        "click_frequency_per_min": float(np.round(click_freq, 2)),
        "idle_time_seconds": float(np.round(idle_total, 2)),
        "error_rate": float(np.round(error_rate, 4)),
        "response_time_seconds": float(np.round(response_time, 2)),
        "retry_count": int(retries),
        "context_switches_per_min": int(ctx_switches_per_min)
    }
