"""
evaluate.py
-----------
Loads the saved best model and evaluates it on the test set.

Generates:
  - Classification report (printed)
  - ROC curve              → results/figures/roc_curve.png
  - Feature importances    → results/figures/feature_importances.png
  - Sample predictions     → results/figures/sample_predictions.png

Run from project root:
    python src/evaluate.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    FEATURES, TARGET, CLASS_NAMES, PRED_THRESHOLD, LABEL_MAP,
    MODEL_PATH, SCALER_PATH, SKEWED_COLS_PATH,
    FIGURES_DIR, RANDOM_SEED, TEST_SIZE,
)
from data_loader import load_data
from preprocess  import preprocess

os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_roc_curve(model, X_test, y_test, model_name: str):
    """ROC curve — roc_curve and auc are now actually used here."""
    y_prob       = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve            → {path}  (AUC = {roc_auc:.4f})")
    return roc_auc


def plot_feature_importances(model, model_name: str):
    """Feature importances — shows which inputs drive loan decisions most."""
    if not hasattr(model, "feature_importances_"):
        print(f"  [{model_name}] does not support feature_importances_ — skipping.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=True)

    plt.figure(figsize=(7, 4))
    plt.barh(feat_df["feature"], feat_df["importance"], color="#5DCAA5", edgecolor="none")
    plt.title(f"Feature Importances — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_importances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importances  → {path}")


def plot_sample_predictions(model, X_test, y_test, n: int = 6):
    """
    Shows n random test applicants with true label, predicted label,
    and confidence — matches the notebook's sample prediction grid.
    """
    plt.figure(figsize=(14, 4))

    for i in range(n):
        idx        = random.randint(0, len(X_test) - 1)
        row        = X_test[idx].reshape(1, -1)
        true_label = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]

        prob       = model.predict_proba(row)[0][1]
        pred       = 1 if prob > PRED_THRESHOLD else 0
        confidence = prob if pred == 1 else 1 - prob
        color      = "green" if pred == true_label else "red"

        ax = plt.subplot(1, n, i + 1)
        ax.barh(
            ["Rejected", "Approved"],
            [1 - prob, prob],
            color=["#F5C4B3", "#9FE1CB"],
        )
        ax.set_xlim(0, 1)
        ax.set_title(
            f"True: {LABEL_MAP[true_label]}\nPred: {LABEL_MAP[pred]}\n{confidence*100:.1f}%",
            color=color, fontsize=9,
        )
        ax.axvline(0.5, color="gray", linestyle="--", lw=0.8)

    plt.suptitle("Sample Predictions (green = correct, red = wrong)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Sample predictions   → {path}")


def evaluate():
    print("\n📊 Loan Approval Prediction — Evaluation\n" + "=" * 45)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at '{MODEL_PATH}'. Run train.py first."
        )

    model      = joblib.load(MODEL_PATH)
    model_name = type(model).__name__
    print(f"  Model loaded: {model_name}\n")

    # Re-load + re-split with same seed → identical test set as training
    df = load_data()
    _, X_test, _, y_test, _ = preprocess(df)

    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("\nGenerating figures ...")
    roc_auc = plot_roc_curve(model, X_test, y_test, model_name)
    plot_feature_importances(model, model_name)
    plot_sample_predictions(model, X_test, y_test)

    print(f"\n🎯 AUC-ROC : {roc_auc:.4f}")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    evaluate()