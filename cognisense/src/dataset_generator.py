"""
Dataset Generator for Cognisense - Cognitive Load Detector

============================================================================
SYNTHETIC / SIMULATED BEHAVIORAL TRAINING DATA -- READ THIS FIRST
============================================================================
This module generates SIMULATED behavioral data. It is NOT collected from
human participants and carries NO real-world validity. It exists to
demonstrate a complete, honest ML pipeline end-to-end.

Per PRD 12 (Data Integrity Requirement), labels here are NOT assigned by an
arbitrary deterministic rule and then presented as genuine measurements.
Instead we model a *generative process* with three properties that real
keyboard/mouse behavioural data always has, and that a naive generator
omits:

  1. INDIVIDUAL DIFFERENCES (PRD 23)
     Each simulated subject gets their own behavioural baseline. A fast
     typist at HIGH load may still out-type a slow typist at LOW load.
     Cognitive load is modelled as a *deviation from personal baseline*,
     not as an absolute WPM threshold.

  2. A SHARED LATENT LOAD FACTOR + CORRELATED NOISE
     Features are not conditionally independent given the label. A single
     latent "effort" variable drives all channels, so features co-vary the
     way they do in real interaction data. Independent per-feature noise
     is what made the earlier version trivially separable.

  3. LABEL AMBIGUITY AT THE BOUNDARIES (PRD 15, PRD 30)
     Cognitive load is continuous; the 3-class discretisation is a product
     convention. Samples near a class boundary get a stochastic label, and
     a small fraction of labels are flipped to represent subjective /
     mismeasured self-report ground truth.

Consequence: models score in a realistic ~75-90% band rather than a
suspicious 99%+. That band is the honest result and is what should be
reported. A near-perfect score on synthetic data measures the generator,
not the model.
============================================================================
"""

import os
import numpy as np
import pandas as pd

# Feature channel definitions.
#   base       : population mean for this channel at neutral load
#   spread     : between-subject variation of the personal baseline
#   load_effect: shift per +1 unit of latent load (in channel units).
#                Sign encodes direction: negative = decreases under load.
#   noise      : within-subject residual noise
#   lo, hi     : physically realistic clipping bounds
#   integer    : whether the channel is a count
CHANNELS = {
    "typing_speed_wpm":          dict(base=48.0,  spread=12.0, load_effect=-9.0,  noise=7.0,   lo=5.0,   hi=140.0, integer=False),
    "avg_keystroke_interval_ms": dict(base=260.0, spread=55.0, load_effect=52.0,  noise=45.0,  lo=80.0,  hi=1200.0, integer=False),
    "backspace_ratio":           dict(base=0.10,  spread=0.03, load_effect=0.045, noise=0.035, lo=0.0,   hi=0.6,   integer=False),
    "pause_frequency_per_min":   dict(base=5.5,   spread=1.8,  load_effect=3.1,   noise=2.0,   lo=0,     hi=40,    integer=True),
    "mouse_velocity_px_s":       dict(base=300.0, spread=70.0, load_effect=-70.0, noise=60.0,  lo=10.0,  hi=1000.0, integer=False),
    "mouse_distance_px":         dict(base=1800.0, spread=350.0, load_effect=380.0, noise=320.0, lo=100.0, hi=8000.0, integer=False),
    "click_frequency_per_min":   dict(base=21.0,  spread=5.0,  load_effect=5.5,   noise=5.0,   lo=0.0,   hi=80.0,  integer=False),
    "idle_time_seconds":         dict(base=7.5,   spread=2.5,  load_effect=3.6,   noise=2.8,   lo=0.0,   hi=60.0,  integer=False),
    "error_rate":                dict(base=0.09,  spread=0.03, load_effect=0.055, noise=0.04,  lo=0.0,   hi=0.8,   integer=False),
    "response_time_seconds":     dict(base=7.8,   spread=2.2,  load_effect=3.6,   noise=2.3,   lo=0.5,   hi=60.0,  integer=False),
    "retry_count":               dict(base=1.8,   spread=0.9,  load_effect=1.5,   noise=1.3,   lo=0,     hi=20,    integer=True),
    "context_switches_per_min":  dict(base=3.8,   spread=1.5,  load_effect=2.6,   noise=2.2,   lo=0,     hi=30,    integer=True),
}

FEATURE_ORDER = list(CHANNELS.keys())
CLASS_NAMES = ["Low", "Medium", "High"]

# Latent-load cut points mapping continuous load -> 3 classes.
# Load is modelled on a roughly standard-normal scale.
LOW_MED_CUT = -0.45
MED_HIGH_CUT = 0.45

# Fraction of labels randomised to represent subjective/mismeasured ground truth.
LABEL_NOISE_RATE = 0.05

# Width of the ambiguous band around each cut point in which the label is
# assigned stochastically rather than deterministically.
BOUNDARY_BAND = 0.18


def _assign_label(load_value, rng):
    """
    Discretise a continuous latent load into Low/Medium/High.

    Near a cut point the label is drawn probabilistically: a sample sitting
    right on a boundary is genuinely ambiguous, and pretending otherwise is
    what produces artificially separable data.
    """
    for cut, lower, upper in ((LOW_MED_CUT, 0, 1), (MED_HIGH_CUT, 1, 2)):
        if abs(load_value - cut) < BOUNDARY_BAND:
            # Linear probability of landing in the upper class across the band.
            p_upper = 0.5 + (load_value - cut) / (2.0 * BOUNDARY_BAND)
            return upper if rng.random() < p_upper else lower

    if load_value < LOW_MED_CUT:
        return 0
    if load_value < MED_HIGH_CUT:
        return 1
    return 2


def generate_cognitive_load_dataset(n_samples_per_class=1000,
                                    random_state=42,
                                    n_subjects=60,
                                    windows_per_subject=None):
    """
    Generate a simulated behavioural dataset with realistic class overlap.

    Generative process, per observation window:
        subject baseline b_c  ~ N(base_c, spread_c)      (once per subject)
        latent load       L   ~ N(mu_subject, 1)          (per window)
        feature           x_c  = b_c + load_effect_c * L + eps_c
        label                 = discretise(L) with boundary + label noise

    Because every channel is driven by the same latent L, the features
    co-vary; because b_c is per-subject, absolute values are not
    diagnostic on their own. Both effects create the honest overlap that
    a conditionally-independent generator lacks.

    Args:
        n_samples_per_class: target samples per class. The generator
            oversamples then balances down to this count per class.
        random_state: reproducibility seed.
        n_subjects: number of distinct simulated individuals.
        windows_per_subject: observation windows per subject. Defaults to
            a value sized to comfortably fill the per-class target.

    Returns:
        Shuffled DataFrame of 12 feature columns + `cognitive_load` label,
        plus a `subject_id` column so subject-aware splitting is possible.
    """
    rng = np.random.default_rng(random_state)

    target_total = n_samples_per_class * len(CLASS_NAMES)
    if windows_per_subject is None:
        # Oversample ~2.2x so class balancing has material to draw from.
        windows_per_subject = int(np.ceil(target_total * 2.2 / n_subjects))

    rows = []
    for subject_id in range(n_subjects):
        # --- Individual differences: this person's own behavioural baseline.
        baseline = {
            name: rng.normal(cfg["base"], cfg["spread"])
            for name, cfg in CHANNELS.items()
        }
        # Some people simply run at a higher typical effort level than others.
        subject_load_mu = rng.normal(0.0, 0.35)
        # Some people react more strongly to load than others.
        subject_sensitivity = rng.normal(1.0, 0.18)

        for _ in range(windows_per_subject):
            latent_load = rng.normal(subject_load_mu, 1.0)
            effective_load = latent_load * subject_sensitivity

            record = {}
            for name, cfg in CHANNELS.items():
                value = (baseline[name]
                         + cfg["load_effect"] * effective_load
                         + rng.normal(0.0, cfg["noise"]))
                value = float(np.clip(value, cfg["lo"], cfg["hi"]))
                record[name] = int(round(value)) if cfg["integer"] else float(np.round(value, 4))

            label = _assign_label(latent_load, rng)

            # Subjective / mismeasured ground truth on a small fraction.
            if rng.random() < LABEL_NOISE_RATE:
                label = int(rng.integers(0, len(CLASS_NAMES)))

            record["cognitive_load"] = int(label)
            record["subject_id"] = subject_id
            rows.append(record)

    df = pd.DataFrame(rows)

    # --- Balance to n_samples_per_class per class for clean evaluation.
    balanced = []
    for label in range(len(CLASS_NAMES)):
        subset = df[df["cognitive_load"] == label]
        if len(subset) >= n_samples_per_class:
            subset = subset.sample(n=n_samples_per_class, random_state=random_state)
        balanced.append(subset)

    df = pd.concat(balanced, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    ordered_columns = FEATURE_ORDER + ["cognitive_load", "subject_id"]
    return df[ordered_columns]


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cognitive_load_behavior_dataset.csv")

    df = generate_cognitive_load_dataset(n_samples_per_class=1000)
    df.to_csv(output_file, index=False)

    print(f"Generated SYNTHETIC dataset: {len(df)} samples -> {output_file}")
    print(f"Distinct simulated subjects: {df['subject_id'].nunique()}")
    print("\nClass balance:")
    print(df["cognitive_load"].value_counts().sort_index().to_string())
    print("\nClass means (note the deliberate overlap):")
    print(df.groupby("cognitive_load")[
        ["typing_speed_wpm", "error_rate", "response_time_seconds"]
    ].agg(["mean", "std"]).round(2).to_string())
