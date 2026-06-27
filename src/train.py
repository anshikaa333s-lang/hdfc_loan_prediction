"""
train.py
--------
Trains Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
Compares accuracy, saves best model + all artifacts to models/.

Matches the MODEL TRAINING & EVALUATION cell in the notebook exactly.

Run from project root:
    python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    N_ESTIMATORS, RANDOM_SEED, CLASS_NAMES,
    MODEL_DIR, MODEL_PATH, FIGURES_DIR, METRICS_DIR, RESULTS_CSV,
)
from data_loader import load_data
from preprocess  import preprocess

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def get_models() -> dict:
    """Returns all models matching the notebook exactly."""
    return {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest"       : RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED),
        "XGBoost"             : XGBClassifier(eval_metric="logloss", random_state=RANDOM_SEED),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED),
    }


def plot_accuracy_comparison(results: dict):
    """Bar chart — matches notebook's Model Accuracy Comparison plot."""
    best_val = max(results.values())
    colors   = ["#1D9E75" if v == best_val else "#B5D4F4" for v in results.values()]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor="none")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0.75, 1.0)
    plt.xticks(rotation=15, ha="right")
    for bar, val in zip(bars, results.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val*100:.2f}%",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Accuracy comparison  → {path}")


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    """Confusion matrix for best model — matches notebook."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix     → {path}")


def train():
    print("\n💳 Loan Approval Prediction — Training\n" + "=" * 45)

    # 1. Load + preprocess
    df = load_data()
    X_train, X_test, y_train, y_test, artifacts = preprocess(df)

    # 2. Train all models + collect results
    models  = get_models()
    results = {}
    rows    = []

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"  ✅ {name} Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds, target_names=CLASS_NAMES))
        rows.append({"model": name, "accuracy": round(acc * 100, 2)})

    # 3. Save results CSV
    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results CSV          → {RESULTS_CSV}")

    # 4. Identify best model
    best_model_name = max(results, key=results.get)
    best_model      = models[best_model_name]
    print(f"\n  Best model : {best_model_name} ({results[best_model_name]*100:.2f}%)")

    # 5. Save all artifacts to models/
    os.makedirs("models/", exist_ok=True)
    joblib.dump(best_model,               MODEL_PATH)
    joblib.dump(artifacts["scaler"],      os.path.join("models", "scaler.pkl"))
    joblib.dump(artifacts["le_home"],     os.path.join("models", "le_home.pkl"))
    joblib.dump(artifacts["skewed_cols"], os.path.join("models", "skewed_cols.pkl"))
    print(f"  ✅ {best_model_name} saved → models/")

    # 6. Plots
    print("\n  Generating plots ...")
    plot_accuracy_comparison(results)
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name)

    return best_model, best_model_name, models, X_test, y_test, artifacts


if __name__ == "__main__":
    train()