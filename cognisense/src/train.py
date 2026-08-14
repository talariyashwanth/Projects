"""
Training Pipeline for Cognisense

Trains and compares Logistic Regression (baseline), Random Forest (main),
and XGBoost (optional experiment) per PRD 13.

Evaluation follows PRD 15: accuracy is never reported alone. We record
macro precision/recall/F1, per-class metrics, confusion matrices, and
cross-validated scores. Cross-validation is GROUPED by subject where
subject IDs exist, so reported scores reflect generalisation to unseen
individuals rather than memorised personal baselines.

Model selection uses cross-validated macro-F1 rather than test-set F1, so
the test set stays an untouched estimate of generalisation and we avoid
selecting a model by peeking at it.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

from dataset_generator import generate_cognitive_load_dataset
from preprocessing import prepare_train_test_data, FEATURE_COLUMNS, CLASS_NAMES

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


def build_model_candidates():
    """Defines the candidate models compared in the experiment."""
    candidates = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=42),
            "use_scaled": True,
            "description": "Linear baseline. Highly interpretable coefficients; "
                           "assumes roughly linear feature-to-load relationships.",
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            "use_scaled": False,
            "description": "Bagged tree ensemble. Captures nonlinear feature "
                           "interactions and yields native feature importances.",
        },
    }

    if HAS_XGBOOST:
        candidates["XGBoost"] = {
            "model": XGBClassifier(
                n_estimators=250, learning_rate=0.06, max_depth=4,
                subsample=0.85, colsample_bytree=0.85,
                reg_lambda=1.5, objective="multi:softprob",
                num_class=3, random_state=42, n_jobs=-1,
                eval_metric="mlogloss", tree_method="hist",
            ),
            "use_scaled": False,
            "description": "Gradient-boosted trees with regularisation and "
                           "subsampling. Strongest nonlinear candidate.",
        }
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        candidates["Gradient Boosting"] = {
            "model": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42
            ),
            "use_scaled": False,
            "description": "Sequential boosting fallback (XGBoost unavailable).",
        }

    return candidates


def plot_confusion_matrix(cm, model_name, out_path):
    """Renders a labelled confusion matrix heatmap."""
    cm = np.asarray(cm, dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm_pct, cmap="magma", vmin=0, vmax=100)

    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("Actual state")
    ax.set_title(f"{model_name} — Confusion Matrix\n(held-out subjects)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{cm_pct[i, j]:.0f}%",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] < 55 else "black",
                    fontsize=10)

    fig.colorbar(im, ax=ax, label="% of actual class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_importance(feat_imp, out_path):
    """Renders a horizontal bar chart of feature importances."""
    names = [f["feature"].replace("_", " ") for f in feat_imp][::-1]
    values = [f["importance"] for f in feat_imp][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(names, values, color="#f43f5e")
    ax.set_xlabel("Permutation importance (mean macro-F1 drop when shuffled)")
    ax.set_title("Behavioural Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def train_and_evaluate_models(regenerate_data=False):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.join(base_dir, "data", "raw", "cognitive_load_behavior_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    # ---------------------------------------------------------------- data
    if regenerate_data or not os.path.exists(raw_data_path):
        print("Generating synthetic dataset...")
        df_raw = generate_cognitive_load_dataset(n_samples_per_class=1000)
        df_raw.to_csv(raw_data_path, index=False)
    else:
        print(f"Loading dataset from {raw_data_path}")
        df_raw = pd.read_csv(raw_data_path)

    data = prepare_train_test_data(df_raw, test_size=0.2, random_state=42)
    X_train, X_test = data["X_train"], data["X_test"]
    X_train_scaled, X_test_scaled = data["X_train_scaled"], data["X_test_scaled"]
    y_train, y_test = data["y_train"], data["y_test"]
    groups_train = data["groups_train"]
    scaler = data["scaler"]

    print(f"Split strategy : {data['split_strategy']}")
    print(f"Train / Test    : {len(X_train)} / {len(X_test)} samples")

    # ------------------------------------------------------ cross-validator
    if groups_train is not None and groups_train.nunique() > 5:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_kwargs = {"groups": groups_train}
        cv_desc = "5-fold StratifiedGroupKFold (subjects never span folds)"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_kwargs = {}
        cv_desc = "5-fold StratifiedKFold"

    print(f"Cross-validation: {cv_desc}\n")

    candidates = build_model_candidates()
    comparison_results = {}
    best_name, best_cv_f1, best_model = None, -1.0, None

    for name, config in candidates.items():
        clf = config["model"]
        X_tr = X_train_scaled if config["use_scaled"] else X_train
        X_te = X_test_scaled if config["use_scaled"] else X_test

        cv_res = cross_validate(
            clf, X_tr, y_train, cv=cv,
            scoring=["accuracy", "f1_macro"], n_jobs=-1, **cv_kwargs
        )
        cv_acc = float(np.mean(cv_res["test_accuracy"]))
        cv_f1 = float(np.mean(cv_res["test_f1_macro"]))
        cv_f1_std = float(np.std(cv_res["test_f1_macro"]))

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        )

        comparison_results[name] = {
            "description": config["description"],
            "test_accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "cv_accuracy_mean": round(cv_acc, 4),
            "cv_f1_mean": round(cv_f1, 4),
            "cv_f1_std": round(cv_f1_std, 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print(f"=== {name} ===")
        print(f"  CV  macro-F1 : {cv_f1:.4f} (+/- {cv_f1_std:.4f})")
        print(f"  Test accuracy: {acc:.4f} | macro-F1: {f1:.4f}")

        plot_confusion_matrix(cm, name, os.path.join(
            reports_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"))

        # Selection on CV F1 keeps the test set an honest held-out estimate.
        if cv_f1 > best_cv_f1:
            best_cv_f1, best_name, best_model = cv_f1, name, clf

    print(f"\n[BEST MODEL] {best_name} (CV macro-F1: {best_cv_f1:.4f})")

    # -------------------------------------------------- feature importance
    # Permutation importance is model-agnostic and measures actual predictive
    # contribution on held-out data, unlike tree impurity importance which is
    # biased toward high-cardinality features.
    best_uses_scaled = candidates[best_name]["use_scaled"]
    X_imp = X_test_scaled if best_uses_scaled else X_test

    print("Computing permutation importance on held-out data...")
    perm = permutation_importance(
        best_model, X_imp, y_test,
        n_repeats=15, random_state=42, n_jobs=-1, scoring="f1_macro"
    )

    total = float(np.sum(np.clip(perm.importances_mean, 0, None))) or 1.0
    feat_imp = [
        {
            "feature": feat,
            "importance": round(float(max(mean_val, 0.0)) / total, 4),
            "raw_f1_drop": round(float(mean_val), 5),
            "std": round(float(std_val), 5),
        }
        for feat, mean_val, std_val in zip(
            FEATURE_COLUMNS, perm.importances_mean, perm.importances_std)
    ]
    feat_imp = sorted(feat_imp, key=lambda d: d["importance"], reverse=True)

    plot_feature_importance(feat_imp, os.path.join(reports_dir, "feature_importance.png"))

    # ------------------------------------------------------------ artifacts
    joblib.dump({"model": best_model, "name": best_name}, os.path.join(
        models_dir, "cognitive_load_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "feature_scaler.pkl"))

    with open(os.path.join(models_dir, "model_comparison.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "selection_criterion": "highest cross-validated macro-F1",
            "class_names": CLASS_NAMES,
            "feature_columns": FEATURE_COLUMNS,
            "split_strategy": data["split_strategy"],
            "cv_strategy": cv_desc,
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "data_provenance": "SYNTHETIC / SIMULATED behavioural data. Not "
                               "collected from human participants. Scores "
                               "demonstrate pipeline correctness and carry no "
                               "real-world validity.",
            "models": comparison_results,
        }, f, indent=2)

    with open(os.path.join(models_dir, "feature_importance.json"), "w") as f:
        json.dump(feat_imp, f, indent=2)

    # ------------------------------------------------- explainability baselines
    # The inference-time attribution engine expresses each incoming feature as a
    # z-score against the population. Those statistics are DERIVED FROM THE
    # TRAINING DATA here rather than hardcoded, so explanations stay consistent
    # with whatever data the model was actually fit on.
    #
    # `high_load_direction` is the sign of the feature's Spearman correlation
    # with the load label: +1 means larger values accompany higher load. It is
    # measured, not assumed.
    baselines = {}
    for col in FEATURE_COLUMNS:
        corr = pd.Series(X_train[col].values).corr(
            pd.Series(y_train.values), method="spearman")
        if pd.isna(corr):
            corr = 0.0
        baselines[col] = {
            "mean": round(float(X_train[col].mean()), 4),
            "std": round(float(X_train[col].std()), 4),
            "high_load_direction": 1 if corr >= 0 else -1,
            "spearman_with_load": round(float(corr), 4),
        }

    with open(os.path.join(models_dir, "feature_baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"Saved model artifacts -> {models_dir}")
    print(f"Saved evaluation plots -> {reports_dir}")
    return comparison_results


if __name__ == "__main__":
    train_and_evaluate_models(regenerate_data=True)
