"""
Модел сургах, hyperparameter tuning хийх модуль
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any


# ============================================================
# Загварууд тодорхойлох
# ============================================================

def get_models() -> Dict[str, Any]:
    """Харьцуулах загваруудын жагсаалт."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,  # imbalanced data-д зориулсан
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }


# ============================================================
# Загвар үнэлэх
# ============================================================

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """
    Загварыг олон metric-ээр үнэлнэ.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    return metrics


# ============================================================
# Бүх загваруудыг сургаж харьцуулах
# ============================================================

def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Бүх загваруудыг сургаж, үр дүнг харьцуулна.

    Returns:
        results: Загвар бүрийн metrics
        best_model: Хамгийн сайн загвар
        best_name: Хамгийн сайн загварын нэр
    """
    models = get_models()
    results = {}
    best_f1 = 0
    best_model = None
    best_name = ""

    print("=" * 60)
    print("ЗАГВАРУУД СУРГАЖ БАЙНА")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"{'─' * 40}")

        # Сургах
        model.fit(X_train, y_train)

        # Үнэлэх
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        # Үр дүн хэвлэх
        for metric_name, value in metrics.items():
            print(f"  {metric_name:>12}: {value:.4f}")

        # Хамгийн сайн загварыг хадгалах
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = model
            best_name = name

    print(f"\n{'=' * 60}")
    print(f"🏆 ХАМГИЙН САЙН ЗАГВАР: {best_name} (F1: {best_f1:.4f})")
    print(f"{'=' * 60}")

    return {
        "results": results,
        "best_model": best_model,
        "best_name": best_name,
    }


# ============================================================
# Загвар хадгалах
# ============================================================

def save_model(model, model_path: str, metrics: dict, metrics_path: str):
    """Загвар болон metrics-ийг хадгална."""
    joblib.dump(model, model_path)
    print(f"Загвар хадгалагдлаа: {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics хадгалагдлаа: {metrics_path}")


def load_model(model_path: str):
    """Хадгалсан загварыг уншина."""
    return joblib.load(model_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    from src.data.preprocess import run_preprocessing_pipeline

    # Preprocessing
    data = run_preprocessing_pipeline("data/raw/telco_churn.csv")

    # Сургалт & Харьцуулалт
    output = train_and_compare(
        data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    )

    # Хамгийн сайн загварыг хадгалах
    save_model(
        model=output["best_model"],
        model_path="models/best_model.pkl",
        metrics=output["results"],
        metrics_path="models/model_metrics.json",
    )
