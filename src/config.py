"""
config.py
---------
Central configuration for Loan Approval Prediction.
All hyperparameters, feature lists, and file paths live here.
Change a value here and it propagates to every other module.
"""

import os

# ── Dataset ───────────────────────────────────────────────────────────────────
KAGGLE_DATASET   = "chilledwanker/loan-approval-prediction"
KAGGLE_FILE      = "credit_risk_dataset.csv"
RANDOM_SEED      = 42
TEST_SIZE        = 0.2

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "loan_amnt",
    "loan_int_rate",
]
TARGET           = "loan_status"
NUMERIC_COLS     = ["person_age", "person_income", "loan_amnt", "loan_int_rate"]
CATEGORICAL_COLS = ["person_home_ownership"]

# ── Preprocessing ─────────────────────────────────────────────────────────────
SKEW_THRESHOLD = 1.0        # abs(skewness) above this → apply log1p

CLIP_BOUNDS = {
    "person_age"     : (18,  100),
    "person_income"  : (0,   5_000_000),
    "loan_amnt"      : (0,   500_000),
    "loan_int_rate"  : (0,   40),
}

# ── Model ─────────────────────────────────────────────────────────────────────
N_ESTIMATORS   = 200
PRED_THRESHOLD = 0.5        # probability > this → Approved (label 1)
CLASS_NAMES    = ["Rejected", "Approved"]   # index 0, index 1
LABEL_MAP      = {0: "Rejected", 1: "Approved"}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR        = os.path.join(BASE_DIR, "models")
FIGURES_DIR      = os.path.join(BASE_DIR, "results", "figures")
METRICS_DIR      = os.path.join(BASE_DIR, "results", "metrics")

MODEL_PATH       = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")
LE_HOME_PATH     = os.path.join(MODEL_DIR, "le_home.pkl")
SKEWED_COLS_PATH = os.path.join(MODEL_DIR, "skewed_cols.pkl")  # saved after training
RESULTS_CSV      = os.path.join(METRICS_DIR, "model_results.csv")