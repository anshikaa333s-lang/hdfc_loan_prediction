"""
visualise.py
------------
Exploratory Data Analysis — matches the EDA - ENHANCED cell in the notebook.

Generates:
  - Loan status count plot          → results/figures/class_distribution.png
  - Histograms + KDE (skew-fixed)   → results/figures/feature_distributions.png
  - Boxplots for numeric features   → results/figures/feature_boxplots.png
  - Correlation heatmap             → results/figures/correlation_heatmap.png

Run from project root:
    python src/visualise.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from config import NUMERIC_COLS, TARGET, FIGURES_DIR, SKEW_THRESHOLD
from data_loader import load_data

os.makedirs(FIGURES_DIR, exist_ok=True)


def apply_skew_correction(df):
    """Apply log1p to skewed columns — same logic as preprocess.py."""
    le = LabelEncoder()
    df = df.copy()
    df["person_home_ownership"] = le.fit_transform(df["person_home_ownership"])
    df[TARGET] = le.fit_transform(df[TARGET])

    for col in NUMERIC_COLS:
        if abs(df[col].skew()) > SKEW_THRESHOLD:
            df[col] = np.log1p(df[col])
    return df


def plot_class_distribution(df):
    """Loan status distribution — matches notebook countplot."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="loan_status", data=df, palette="pastel", edgecolor="none")
    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status  (0 = Rejected, 1 = Approved)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Class distribution   → {path}")


def plot_feature_distributions(df):
    """Histograms + KDE — matches notebook 'Skew Corrected' histplots."""
    plt.figure(figsize=(18, 5))
    for i, col in enumerate(NUMERIC_COLS, 1):
        plt.subplot(1, 4, i)
        sns.histplot(df[col], bins=30, kde=True, color="skyblue", edgecolor="none")
        plt.title(f"{col} Distribution\n(Skew Corrected)")
        plt.xlabel("")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature histograms   → {path}")


def plot_boxplots(df):
    """Boxplots for all numeric features — matches notebook."""
    plt.figure(figsize=(18, 5))
    for i, col in enumerate(NUMERIC_COLS, 1):
        plt.subplot(1, 4, i)
        sns.boxplot(y=df[col], color="lightgreen")
        plt.title(f"{col} Boxplot")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "feature_boxplots.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Boxplots             → {path}")


def plot_correlation_heatmap(df):
    """Correlation heatmap — matches notebook coolwarm heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[NUMERIC_COLS + [TARGET]].corr(),
        annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Correlation heatmap  → {path}")


def run_eda():
    print("\n🔍 Loan Approval — Exploratory Data Analysis\n" + "=" * 45)

    df_raw = load_data()
    df     = apply_skew_correction(df_raw)

    print("\nGenerating EDA plots ...")
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)

    print("\n✅ EDA complete — all figures in results/figures/")


if __name__ == "__main__":
    run_eda()