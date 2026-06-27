"""
data_loader.py
--------------
Downloads the Loan Approval dataset from Kaggle via kagglehub,
selects the required features, fills missing values, and clips outliers.

Matches exactly what the notebook does in the DATA PREPROCESSING cell.

Usage:
    from src.data_loader import load_data
    df = load_data()
"""

import os
import sys
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    KAGGLE_DATASET, KAGGLE_FILE,
    FEATURES, TARGET, NUMERIC_COLS, CLIP_BOUNDS,
)


def download_dataset() -> pd.DataFrame:
    """
    Downloads the credit risk dataset from Kaggle using kagglehub.
    Requires KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.

    Returns:
        Raw pandas DataFrame.
    """
    print(f"⬇️  Downloading: {KAGGLE_DATASET} ...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        KAGGLE_FILE,
    )
    print("✅ Dataset loaded! Shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())
    return df


def load_data() -> pd.DataFrame:
    """
    Downloads and performs initial cleaning on the loan dataset.

    Steps (matching notebook exactly):
        1. Download from Kaggle
        2. Select features + target
        3. Fill missing values with median / mode
        4. Clip extreme outliers to sensible bounds

    Returns:
        Cleaned DataFrame — shape (N, 6).
    """
    df = download_dataset()

    # Select only the columns we need — .copy() prevents SettingWithCopyWarning
    df = df[FEATURES + [TARGET]].copy()

    # ── Fill missing values ────────────────────────────────────────────────────
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    df["person_home_ownership"] = df["person_home_ownership"].fillna(
        df["person_home_ownership"].mode()[0]
    )
    df[TARGET] = df[TARGET].fillna(df[TARGET].mode()[0])

    # ── Clip extreme outliers ──────────────────────────────────────────────────
    for col, (lo, hi) in CLIP_BOUNDS.items():
        df[col] = df[col].clip(lo, hi)

    print(f"\n✅ Data cleaned — shape: {df.shape}")
    return df