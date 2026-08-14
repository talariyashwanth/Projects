"""
Training Pipeline for Cognisense
Trains Logistic Regression, Random Forest, and XGBoost/GradientBoosting models.
Evaluates metrics, cross-validation, confusion matrices, and feature importances.
Saves model & scaler artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from dataset_generator import generate_cognitive_load_dataset
from preprocessing import clean_and_validate_data, prepare_train_test_data, FEATURE_COLUMNS, CLASS_NAMES

def train_and_evaluate_models():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.join(base_dir, "data", "raw", "cognitive_load_behavior_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    if not os.path.exists(raw_data_path):
        print("Generating raw dataset...")
        df_raw = generate_cognitive_load_dataset(n_samples_per_class=1000)
        df_raw.to_csv(raw_data_path, index=False)
    else:
        print(f"Loading raw dataset from {raw_data_path}...")
        df_raw = pd.read_csv(raw_data_path)

    data_dict = prepare_train_test_data(df_raw, test_size=0.2, random_state=42)
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    X_train_scaled = data_dict["X_train_scaled"]
    X_test_scaled = data_dict["X_test_scaled"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    scaler = data_dict["scaler"]

    # Define Candidate Models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True,
            "description": "Linear baseline model with high interpretability"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "description": "Ensemble decision tree model capturing nonlinear interaction rules"
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, random_state=42),
            "use_scaled": False,
            "description": "Sequential boosting model optimized for complex feature interactions"
        }
    }

    comparison_results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in models.items():
        clf = config["model"]
        use_scaled = config["use_scaled"]
        
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # 5-Fold Cross Validation
        cv_res = cross_validate(clf, X_tr, y_train, cv=skf, scoring=['accuracy', 'f1_macro'])
        cv_acc_mean = float(np.mean(cv_res['test_accuracy']))
        cv_f1_mean = float(np.mean(cv_res['test_f1_macro']))

        # Fit on full training set
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)

        # Performance Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='macro'))
        rec = float(recall_score(y_test, y_pred, average='macro'))
        f1 = float(f1_score(y_test, y_pred, average='macro'))
        cm = confusion_matrix(y_test, y_pred).tolist()

        clf_report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

        comparison_results[name] = {
            "description": config["description"],
            "test_accuracy": round(acc, 4),
            "precision_macro": round(prec, 4),
            "recall_macro": round(rec, 4),
            "f1_macro": round(f1, 4),
            "cv_accuracy_mean": round(cv_acc_mean, 4),
            "cv_f1_mean": round(cv_f1_mean, 4),
            "confusion_matrix": cm,
            "classification_report": clf_report
        }

        print(f"=== {name} ===")
        print(f"Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | CV Accuracy: {cv_acc_mean:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf

    print(f"\n[BEST MODEL] Best Model Selected: {best_model_name} (F1 Macro: {best_f1:.4f})")

    # Compute Feature Importances (from Random Forest)
    rf_model = models["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    feat_imp = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in zip(FEATURE_COLUMNS, importances)
    ]
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)

    # Save artifacts
    model_save_path = os.path.join(models_dir, "cognitive_load_model.pkl")
    scaler_save_path = os.path.join(models_dir, "feature_scaler.pkl")
    metrics_save_path = os.path.join(models_dir, "model_comparison.json")
    feat_imp_save_path = os.path.join(models_dir, "feature_importance.json")

    joblib.dump({"model": best_model_obj, "name": best_model_name}, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    with open(metrics_save_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "class_names": CLASS_NAMES,
            "feature_columns": FEATURE_COLUMNS,
            "models": comparison_results
        }, f, indent=2)

    with open(feat_imp_save_path, "w") as f:
        json.dump(feat_imp, f, indent=2)

    print(f"Saved artifacts to {models_dir}")
    return comparison_results

if __name__ == "__main__":
    train_and_evaluate_models()
