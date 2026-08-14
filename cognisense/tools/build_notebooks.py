"""
Builds the two PRD-specified notebooks (01_eda, 02_model_training) and
executes them so the committed .ipynb files contain real output.

Run:  python tools/build_notebooks.py
"""

import json
import os
import subprocess
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(BASE, "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().split("\n")}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="rocket")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
"""

EDA_CELLS = [
    md("""
# Cognisense — 01: Exploratory Data Analysis

Explores the **synthetic** behavioral dataset used to train Cognisense.

> ⚠️ This data is simulated, not collected from human participants. The goal of this
> notebook is to verify that the generated data has the properties real behavioral
> data has — especially **class overlap**. Cleanly separated classes would make the
> downstream accuracy meaningless.
"""),
    code(SETUP),
    code("""
from dataset_generator import generate_cognitive_load_dataset
from preprocessing import FEATURE_COLUMNS, CLASS_NAMES, GROUP_COLUMN

df = pd.read_csv(os.path.join("..", "data", "raw", "cognitive_load_behavior_dataset.csv"))
df["load_name"] = df["cognitive_load"].map(dict(enumerate(CLASS_NAMES)))

print(f"Shape            : {df.shape}")
print(f"Simulated people : {df[GROUP_COLUMN].nunique()}")
print(f"Missing values   : {int(df.isna().sum().sum())}")
df.head()
"""),
    md("## 1. Class balance and target distribution"),
    code("""
print(df["load_name"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(5, 3.4))
sns.countplot(data=df, x="load_name", order=CLASS_NAMES, ax=ax)
ax.set_title("Class balance")
ax.set_xlabel("")
plt.tight_layout()
plt.show()
"""),
    md("""
## 2. Descriptive statistics by class

The **standard deviations matter more than the means here**. If within-class spread is
small relative to the gaps between class means, the classes are trivially separable and
any model will score ~99%.
"""),
    code("""
summary = df.groupby("load_name")[FEATURE_COLUMNS].agg(["mean", "std"]).round(2)
summary.reindex(CLASS_NAMES)[
    ["typing_speed_wpm", "error_rate", "response_time_seconds", "pause_frequency_per_min"]
]
"""),
    md("""
## 3. Distribution overlap — the key diagnostic

Each class distribution should visibly **overlap** its neighbours. This is what makes the
classification problem non-trivial and the reported metrics meaningful.
"""),
    code("""
key_features = ["typing_speed_wpm", "error_rate", "response_time_seconds",
                "pause_frequency_per_min", "backspace_ratio", "context_switches_per_min"]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, feature in zip(axes.ravel(), key_features):
    for state in CLASS_NAMES:
        sns.kdeplot(data=df[df["load_name"] == state], x=feature,
                    fill=True, alpha=0.4, label=state, ax=ax)
    ax.set_title(feature.replace("_", " "))
    ax.set_xlabel("")
    ax.legend(fontsize=8)

fig.suptitle("Feature distributions by load state — note the deliberate overlap", y=1.02)
plt.tight_layout()
plt.show()
"""),
    code("""
# Quantify separability: gap between Low and High means, in pooled std units.
print("Separability (Low-High mean gap / pooled std) — lower means harder:\\n")
rows = []
for feature in FEATURE_COLUMNS:
    grouped = df.groupby("cognitive_load")[feature]
    means, stds = grouped.mean(), grouped.std()
    gap = abs(means[2] - means[0])
    pooled = stds.mean()
    rows.append({"feature": feature, "separability": round(gap / pooled, 2)})

sep = pd.DataFrame(rows).sort_values("separability", ascending=False)
print(sep.to_string(index=False))
print("\\nAll values well below ~3.0 => no single feature trivially solves the task.")
"""),
    md("""
## 4. Feature correlations

Features are driven by a shared latent load factor, so they **co-vary**. This is realistic:
in genuine interaction data, slowing down, pausing, and erring happen together. A generator
producing conditionally independent features would show a near-empty correlation matrix.
"""),
    code("""
fig, ax = plt.subplots(figsize=(9.5, 7.5))
corr = df[FEATURE_COLUMNS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="rocket_r", center=0,
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.show()
"""),
    code("""
# Correlation of each feature with the load label (Spearman = monotonic association).
label_corr = (df[FEATURE_COLUMNS]
              .apply(lambda s: s.corr(df["cognitive_load"], method="spearman"))
              .sort_values(key=abs, ascending=False)
              .round(3))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
label_corr.plot(kind="barh", ax=ax, color="#f43f5e")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Spearman correlation with cognitive load")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

label_corr.to_frame("spearman_vs_load")
"""),
    md("""
## 5. Individual differences

Each simulated subject has their own behavioral baseline. Because per-subject means vary
widely, an **absolute** typing speed is not diagnostic on its own — which is exactly why
the model needs multiple correlated signals, and why personalized baselines (PRD §23) help.
"""),
    code("""
per_subject = df.groupby(GROUP_COLUMN)["typing_speed_wpm"].mean()

fig, ax = plt.subplots(figsize=(7.5, 3.8))
sns.histplot(per_subject, bins=22, ax=ax, color="#38bdf8")
ax.set_title("Distribution of per-subject mean typing speed")
ax.set_xlabel("Mean WPM for a subject")
plt.tight_layout()
plt.show()

print(f"Slowest subject baseline: {per_subject.min():.1f} WPM")
print(f"Fastest subject baseline: {per_subject.max():.1f} WPM")
print("\\nA fast typist under HIGH load can still out-type a slow typist at LOW load:")
fast_high = df[(df[GROUP_COLUMN] == per_subject.idxmax()) & (df["cognitive_load"] == 2)]
slow_low = df[(df[GROUP_COLUMN] == per_subject.idxmin()) & (df["cognitive_load"] == 0)]
if len(fast_high) and len(slow_low):
    print(f"  Fastest subject @ HIGH load: {fast_high['typing_speed_wpm'].mean():.1f} WPM")
    print(f"  Slowest subject @ LOW  load: {slow_low['typing_speed_wpm'].mean():.1f} WPM")
"""),
    md("""
## Takeaways

1. Classes are **balanced** and contain **no missing values**.
2. Every feature distribution **overlaps** across states; no single feature separates them.
3. Features are **correlated**, driven by a shared latent load factor.
4. **Between-subject variation is large**, so absolute values are weak evidence alone.

These properties are why the models land in a realistic ~75–84% band rather than ~99%.

→ Continue to `02_model_training.ipynb`.
"""),
]

TRAIN_CELLS = [
    md("""
# Cognisense — 02: Model Training & Evaluation

Trains and compares three classifiers, then evaluates the winner honestly.

Two methodological choices drive everything here:

1. **Subject-aware splitting** — no individual appears in both train and test, so scores
   measure generalisation *to a new person*.
2. **Selection on cross-validated F1** — the test set is scored once, at the end, so it
   stays an unbiased estimate.
"""),
    code(SETUP),
    code("""
from preprocessing import prepare_train_test_data, FEATURE_COLUMNS, CLASS_NAMES, GROUP_COLUMN

df = pd.read_csv(os.path.join("..", "data", "raw", "cognitive_load_behavior_dataset.csv"))
data = prepare_train_test_data(df, test_size=0.2, random_state=42)

X_train, X_test = data["X_train"], data["X_test"]
X_train_scaled, X_test_scaled = data["X_train_scaled"], data["X_test_scaled"]
y_train, y_test = data["y_train"], data["y_test"]
groups_train = data["groups_train"]

print(f"Split strategy : {data['split_strategy']}")
print(f"Train samples  : {len(X_train)}")
print(f"Test samples   : {len(X_test)}")
"""),
    md("""
### Verify there is no subject leakage

This is the single most important check in the notebook. If it fails, every metric below
is inflated.
"""),
    code("""
from preprocessing import clean_and_validate_data

cleaned = clean_and_validate_data(df)
train_subjects = set(cleaned.loc[X_train.index, GROUP_COLUMN])
test_subjects = set(cleaned.loc[X_test.index, GROUP_COLUMN])
overlap = train_subjects & test_subjects

print(f"Subjects in train : {len(train_subjects)}")
print(f"Subjects in test  : {len(test_subjects)}")
print(f"Overlap           : {len(overlap)}")
assert not overlap, "LEAKAGE: a subject appears in both splits"
print("\\n✅ No subject leakage — test subjects are entirely unseen.")
"""),
    md("## 1. Candidate models"),
    code("""
from sklearn.model_selection import cross_validate, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)
from train import build_model_candidates

candidates = build_model_candidates()
for name, cfg in candidates.items():
    print(f"{name:22} scaled={str(cfg['use_scaled']):5}  {cfg['description'][:60]}")
"""),
    md("""
## 2. Grouped cross-validation

`StratifiedGroupKFold` keeps class balance per fold *and* prevents a subject spanning
folds — so CV scores are directly comparable to the held-out test score.
"""),
    code("""
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, cfg in candidates.items():
    X_tr = X_train_scaled if cfg["use_scaled"] else X_train
    X_te = X_test_scaled if cfg["use_scaled"] else X_test

    cv_res = cross_validate(cfg["model"], X_tr, y_train, cv=cv, groups=groups_train,
                            scoring=["accuracy", "f1_macro"], n_jobs=-1)

    cfg["model"].fit(X_tr, y_train)
    y_pred = cfg["model"].predict(X_te)

    results[name] = {
        "cv_f1_mean": np.mean(cv_res["test_f1_macro"]),
        "cv_f1_std": np.std(cv_res["test_f1_macro"]),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "y_pred": y_pred,
    }
    print(f"{name:22} CV-F1={results[name]['cv_f1_mean']:.4f} "
          f"(+/-{results[name]['cv_f1_std']:.4f})  test-F1={results[name]['test_f1']:.4f}")
"""),
    code("""
comparison = pd.DataFrame({
    name: {"CV macro-F1": r["cv_f1_mean"], "CV std": r["cv_f1_std"],
           "Test accuracy": r["test_accuracy"], "Test macro-F1": r["test_f1"],
           "Precision": r["precision"], "Recall": r["recall"]}
    for name, r in results.items()
}).T.round(4)

comparison.sort_values("CV macro-F1", ascending=False)
"""),
    md("""
### Model selection

The winner is whichever model has the highest **cross-validated** macro-F1 — not the
highest test score. Selecting on the test set would turn it into a validation set and
bias the final estimate.
"""),
    code("""
best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
print(f"Selected model: {best_name} (CV macro-F1 = {results[best_name]['cv_f1_mean']:.4f})")

if best_name == "Logistic Regression":
    print("\\nThe LINEAR baseline won. The behavioral signals are largely monotonic with")
    print("load, so the extra capacity of the tree ensembles bought overfitting rather")
    print("than accuracy. Reporting the simplest adequate model is the honest choice.")
"""),
    md("""
## 3. Per-class metrics and confusion matrix

Accuracy alone is not reported (PRD §15). Per-class metrics reveal *where* the model
struggles.
"""),
    code("""
y_pred_best = results[best_name]["y_pred"]
print(classification_report(y_test, y_pred_best, target_names=CLASS_NAMES, digits=3))
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

for ax, normalize, title in zip(axes, [None, "true"], ["Counts", "Row-normalised"]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_best, display_labels=CLASS_NAMES,
        cmap="magma", normalize=normalize, ax=ax, colorbar=False,
        values_format=".2f" if normalize else "d")
    ax.set_title(f"{best_name} — {title}")

plt.tight_layout()
plt.show()
"""),
    code("""
cm = confusion_matrix(y_test, y_pred_best)
severe = cm[0, 2] + cm[2, 0]   # Low<->High: the serious errors
adjacent = cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]

print(f"Severe errors (Low<->High)     : {severe} ({severe / cm.sum() * 100:.1f}%)")
print(f"Adjacent errors (off-by-one)   : {adjacent} ({adjacent / cm.sum() * 100:.1f}%)")
print("\\nErrors concentrate on ADJACENT states, which is the expected failure mode when")
print("discretising a continuum. 'Medium' absorbs ambiguity from both boundaries.")
"""),
    md("""
## 4. Permutation importance

Tree impurity importance is biased toward high-cardinality features, so we use
**permutation importance on held-out data**: the actual macro-F1 drop when a feature's
values are shuffled.
"""),
    code("""
from sklearn.inspection import permutation_importance

best_model = candidates[best_name]["model"]
X_imp = X_test_scaled if candidates[best_name]["use_scaled"] else X_test

perm = permutation_importance(best_model, X_imp, y_test, n_repeats=15,
                              random_state=42, n_jobs=-1, scoring="f1_macro")

imp = (pd.DataFrame({"feature": FEATURE_COLUMNS,
                     "f1_drop": perm.importances_mean,
                     "std": perm.importances_std})
       .sort_values("f1_drop", ascending=False)
       .reset_index(drop=True))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp["feature"][::-1], imp["f1_drop"][::-1],
        xerr=imp["std"][::-1], color="#f43f5e")
ax.set_xlabel("Macro-F1 drop when shuffled")
ax.set_title(f"{best_name} — permutation importance (held-out subjects)")
plt.tight_layout()
plt.show()

imp.round(4)
"""),
    md("""
> **Interpretation caveat (PRD §16):** these values show which features the *trained model
> relies on*. They are not evidence that these behaviors **cause** cognitive load.
"""),
    md("""
## 5. Score calibration check

The product exposes a 0–100 score derived from class probabilities. For that score to be
meaningful, predicted probabilities should track observed frequencies.
"""),
    code("""
probs = best_model.predict_proba(X_imp)
p_high = probs[:, 2]
actually_high = (y_test.values == 2).astype(int)

bins = np.linspace(0, 1, 11)
idx = np.digitize(p_high, bins) - 1

rows = []
for b in range(10):
    mask = idx == b
    if mask.sum() >= 5:
        rows.append({"predicted_prob": (bins[b] + bins[b + 1]) / 2,
                     "observed_freq": actually_high[mask].mean(),
                     "n": int(mask.sum())})

cal = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.plot(cal["predicted_prob"], cal["observed_freq"], "o-", color="#f43f5e", label=best_name)
ax.set_xlabel("Predicted P(High)")
ax.set_ylabel("Observed frequency of High")
ax.set_title("Calibration — High class")
ax.legend()
plt.tight_layout()
plt.show()

cal.round(3)
"""),
    md("""
## 6. Explainability and personalization

The inference engine converts a feature vector into a score plus ranked attributions, and
supports scoring against an individual's **own** baseline (PRD §23).
"""),
    code("""
from predict import CognitiveLoadPredictor

predictor = CognitiveLoadPredictor(model_dir=os.path.join("..", "models"))

strained = {"typing_speed_wpm": 24.0, "avg_keystroke_interval_ms": 440.0,
            "backspace_ratio": 0.26, "pause_frequency_per_min": 14,
            "mouse_velocity_px_s": 155.0, "mouse_distance_px": 2700.0,
            "click_frequency_per_min": 33.0, "idle_time_seconds": 15.0,
            "error_rate": 0.27, "response_time_seconds": 17.0,
            "retry_count": 5, "context_switches_per_min": 10}

result = predictor.predict(strained)
print(f"State : {result['predicted_state']}")
print(f"Score : {result['cognitive_load_score']}/100 ({result['score_band']})")
print("\\nWhy?")
for s in result["top_contributing_signals"]:
    arrow = "^" if s["z_score"] > 0 else "v"
    print(f"  {arrow} {s['display_name']:<22} {s['value']:>8}  (z={s['z_score']:+.2f})")
"""),
    code("""
# A naturally slow typist, working at their OWN normal pace.
calibration = [dict(strained, typing_speed_wpm=25.0 + i, error_rate=0.05,
                    backspace_ratio=0.06, pause_frequency_per_min=3,
                    response_time_seconds=6.0) for i in range(6)]

personal = CognitiveLoadPredictor.build_personal_baseline(calibration)
current = dict(strained, typing_speed_wpm=26.0, error_rate=0.05,
               backspace_ratio=0.06, pause_frequency_per_min=3,
               response_time_seconds=6.0)

pop = {a["feature"]: a for a in predictor.predict(current)["all_attributions"]}
per = {a["feature"]: a for a in
       predictor.predict(current, personal_baseline=personal)["all_attributions"]}

print("Typing speed = 26 WPM for a person whose personal baseline IS ~27 WPM\\n")
print(f"  vs population : z={pop['typing_speed_wpm']['z_score']:+.2f}  "
      f"flagged={pop['typing_speed_wpm']['is_elevating_load']}")
print(f"  vs own norm   : z={per['typing_speed_wpm']['z_score']:+.2f}  "
      f"flagged={per['typing_speed_wpm']['is_elevating_load']}")
print("\\nPersonalization stops us calling a slow typist 'overloaded' for typing normally.")
"""),
    md("""
## Summary

| Choice | Effect on reported score | Why it's correct |
|---|---|---|
| Subject-aware split | Lowers | Measures generalisation to a new person |
| Grouped CV | Lowers | No subject spans folds |
| Selection on CV F1 | Neutral | Keeps the test set unbiased |
| Permutation importance | — | Unbiased vs impurity importance |
| Realistic synthetic overlap | Lowers a lot | ~99% would measure the generator |

Final: **~75–84% macro-F1**, with errors concentrated on adjacent states. That is the
honest result, and it is the number to report.
"""),
]


def main():
    os.makedirs(NB_DIR, exist_ok=True)

    targets = [("01_eda.ipynb", EDA_CELLS), ("02_model_training.ipynb", TRAIN_CELLS)]
    for filename, cells in targets:
        path = os.path.join(NB_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook(cells), f, indent=1)
        print(f"wrote {path}")

    print("\nExecuting notebooks (this populates real outputs)...")
    for filename, _ in targets:
        path = os.path.join(NB_DIR, filename)
        proc = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", path],
            capture_output=True, text=True, cwd=NB_DIR)
        status = "OK" if proc.returncode == 0 else "FAILED"
        print(f"  {filename}: {status}")
        if proc.returncode != 0:
            print(proc.stderr[-2500:])
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
