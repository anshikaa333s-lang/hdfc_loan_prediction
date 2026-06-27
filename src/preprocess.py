"""
preprocess.py
-------------
Handles encoding, skew correction, train/test split, and feature scaling.

CRITICAL: skewed_cols list is built BEFORE applying log1p to df, then
saved to disk. The predict script loads this list to apply the same
transformation to new inputs — without checking skewness on the already-
transformed DataFrame (which would always return a low skew score).

Usage:
    from src.preprocess import preprocess
    X_train, X_test, y_train, y_test, artifacts = preprocess(df)
"""

import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    FEATURES, TARGET, NUMERIC_COLS,
    SKEW_THRESHOLD, RANDOM_SEED, TEST_SIZE,
    MODEL_DIR, SCALER_PATH, LE_HOME_PATH, SKEWED_COLS_PATH,
)

os.makedirs(MODEL_DIR, exist_ok=True)


def preprocess(df):
    """
    Full preprocessing pipeline — matches the notebook cell order exactly.

    Steps:
        1. Encode person_home_ownership with LabelEncoder
        2. Encode loan_status (target) with LabelEncoder
        3. Detect skewed numeric columns and apply log1p
           — skewed_cols list is recorded BEFORE transforming df
        4. Train / test split (80/20, stratified, seed=42)
        5. StandardScaler fit on train, transform both
        6. Save scaler, le_home, skewed_cols to models/

    Args:
        df : Cleaned DataFrame from data_loader.load_data()

    Returns:
        X_train_scaled, X_test_scaled : np.ndarray
        y_train, y_test               : np.ndarray
        artifacts : dict — scaler, le_home, le_target, skewed_cols
    """
    df = df.copy()

    # ── Step 1: Encode categorical feature ────────────────────────────────────
    le_home = LabelEncoder()
    df["person_home_ownership"] = le_home.fit_transform(df["person_home_ownership"])

    # ── Step 2: Encode target ─────────────────────────────────────────────────
    le_target = LabelEncoder()
    df[TARGET] = le_target.fit_transform(df[TARGET])

    # ── Step 3: Skew correction ────────────────────────────────────────────────
    # Record skewed columns BEFORE transforming df so we can
    # use this list in predict.py on new inputs
    skewed_cols = []
    for col in NUMERIC_COLS:
        skewness = df[col].skew()
        if abs(skewness) > SKEW_THRESHOLD:
            print(f"  log1p applied to '{col}' (skew = {skewness:.2f})")
            skewed_cols.append(col)
            df[col] = np.log1p(df[col])

    print(f"\n  Columns with log1p applied: {skewed_cols}")

    # ── Step 4: Train / test split ─────────────────────────────────────────────
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_SEED,
        stratify     = y,
    )
    print(f"\n  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")

    # ── Step 5: Scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Step 6: Save artifacts ─────────────────────────────────────────────────
    joblib.dump(scaler,      SCALER_PATH)
    joblib.dump(le_home,     LE_HOME_PATH)
    joblib.dump(skewed_cols, SKEWED_COLS_PATH)
    print(f"\n  Artifacts saved → {MODEL_DIR}/")

    artifacts = {
        "scaler"      : scaler,
        "le_home"     : le_home,
        "le_target"   : le_target,
        "skewed_cols" : skewed_cols,
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, artifacts