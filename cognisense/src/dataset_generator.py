"""
Dataset Generator for Cognisense — Cognitive Load Detector
Generates a realistic synthetic dataset grounded in HCI & human cognitive performance research.
Explicitly labeled as Simulated Behavioral Training Data.
"""

import os
import numpy as np
import pandas as pd

def generate_cognitive_load_dataset(n_samples_per_class=1000, random_state=42):
    """
    Generates realistic behavioral feature data across 3 cognitive load states:
    - 0: Low (comfortable, smooth interaction, high WPM, low error/pause)
    - 1: Medium (moderate effort, slightly slower, occasional hesitation)
    - 2: High (cognitive overload, long pauses, frequent corrections, erratic mouse/context switching)
    """
    np.random.seed(random_state)
    data = []

    classes = [
        {"label": 0, "name": "Low"},
        {"label": 1, "name": "Medium"},
        {"label": 2, "name": "High"}
    ]

    for c in classes:
        label = c["label"]
        n = n_samples_per_class

        if label == 0:  # Low Load
            wpm = np.random.normal(64.0, 10.0, n)
            keystroke_interval = np.random.normal(185.0, 35.0, n)
            backspace_ratio = np.random.normal(0.045, 0.025, n)
            pause_freq = np.random.poisson(2.5, n)
            mouse_velocity = np.random.normal(420.0, 70.0, n)
            mouse_distance = np.random.normal(1300.0, 250.0, n)
            click_freq = np.random.normal(15.0, 4.0, n)
            idle_time = np.random.normal(4.0, 1.8, n)
            error_rate = np.random.normal(0.04, 0.02, n)
            response_time = np.random.normal(4.5, 1.2, n)
            retry_count = np.random.poisson(0.5, n)
            context_switches = np.random.poisson(1.5, n)

        elif label == 1:  # Medium Load
            wpm = np.random.normal(48.0, 9.0, n)
            keystroke_interval = np.random.normal(255.0, 45.0, n)
            backspace_ratio = np.random.normal(0.10, 0.04, n)
            pause_freq = np.random.poisson(5.5, n)
            mouse_velocity = np.random.normal(300.0, 60.0, n)
            mouse_distance = np.random.normal(1800.0, 300.0, n)
            click_freq = np.random.normal(21.0, 5.0, n)
            idle_time = np.random.normal(7.5, 2.8, n)
            error_rate = np.random.normal(0.09, 0.035, n)
            response_time = np.random.normal(7.8, 2.0, n)
            retry_count = np.random.poisson(1.6, n)
            context_switches = np.random.poisson(3.8, n)

        else:  # High Load
            wpm = np.random.normal(32.0, 8.0, n)
            keystroke_interval = np.random.normal(360.0, 60.0, n)
            backspace_ratio = np.random.normal(0.18, 0.06, n)
            pause_freq = np.random.poisson(11.0, n)
            mouse_velocity = np.random.normal(190.0, 50.0, n)
            mouse_distance = np.random.normal(2400.0, 400.0, n)
            click_freq = np.random.normal(29.0, 6.5, n)
            idle_time = np.random.normal(13.0, 4.0, n)
            error_rate = np.random.normal(0.19, 0.06, n)
            response_time = np.random.normal(13.5, 3.5, n)
            retry_count = np.random.poisson(4.2, n)
            context_switches = np.random.poisson(8.0, n)

        # Clip values to physically realistic bounds
        wpm = np.clip(wpm, 5.0, 140.0)
        keystroke_interval = np.clip(keystroke_interval, 80.0, 1200.0)
        backspace_ratio = np.clip(backspace_ratio, 0.0, 0.6)
        pause_freq = np.clip(pause_freq, 0, 40)
        mouse_velocity = np.clip(mouse_velocity, 10.0, 1000.0)
        mouse_distance = np.clip(mouse_distance, 100.0, 8000.0)
        click_freq = np.clip(click_freq, 0.0, 80.0)
        idle_time = np.clip(idle_time, 0.0, 60.0)
        error_rate = np.clip(error_rate, 0.0, 0.8)
        response_time = np.clip(response_time, 0.5, 60.0)
        retry_count = np.clip(retry_count, 0, 20)
        context_switches = np.clip(context_switches, 0, 30)

        for i in range(n):
            data.append({
                "typing_speed_wpm": float(np.round(wpm[i], 2)),
                "avg_keystroke_interval_ms": float(np.round(keystroke_interval[i], 2)),
                "backspace_ratio": float(np.round(backspace_ratio[i], 4)),
                "pause_frequency_per_min": int(pause_freq[i]),
                "mouse_velocity_px_s": float(np.round(mouse_velocity[i], 2)),
                "mouse_distance_px": float(np.round(mouse_distance[i], 2)),
                "click_frequency_per_min": float(np.round(click_freq[i], 2)),
                "idle_time_seconds": float(np.round(idle_time[i], 2)),
                "error_rate": float(np.round(error_rate[i], 4)),
                "response_time_seconds": float(np.round(response_time[i], 2)),
                "retry_count": int(retry_count[i]),
                "context_switches_per_min": int(context_switches[i]),
                "cognitive_load": int(label)
            })

    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cognitive_load_behavior_dataset.csv")
    
    df = generate_cognitive_load_dataset(n_samples_per_class=1000)
    df.to_csv(output_file, index=False)
    print(f"Generated dataset with {len(df)} samples saved to: {output_file}")
