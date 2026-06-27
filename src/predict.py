"""
predict.py
----------
Single applicant loan approval prediction — local version of the
DEMO PREDICTION cell in the notebook.

Critical fix applied:
    Uses saved skewed_cols list (from training) instead of re-checking
    df[col].skew() on the already-transformed DataFrame. Without this,
    log1p would NOT be applied to the demo input, giving a wrong result.

Usage (from project root):
    python src/predict.py                               ← runs demo applicant
    python src/predict.py --age 25 --income 35000 \\
        --home RENT --loan 10000 --rate 14.0
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    FEATURES, LABEL_MAP,
    MODEL_PATH, SCALER_PATH, LE_HOME_PATH, SKEWED_COLS_PATH,
)


def load_artifacts() -> dict:
    """Loads model + all preprocessing artifacts saved by train.py."""
    for path, name in [
        (MODEL_PATH,       "best_model.pkl"),
        (SCALER_PATH,      "scaler.pkl"),
        (LE_HOME_PATH,     "le_home.pkl"),
        (SKEWED_COLS_PATH, "skewed_cols.pkl"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{name}' not found at '{path}'. Run train.py first."
            )

    return {
        "model"       : joblib.load(MODEL_PATH),
        "scaler"      : joblib.load(SCALER_PATH),
        "le_home"     : joblib.load(LE_HOME_PATH),
        "skewed_cols" : joblib.load(SKEWED_COLS_PATH),  # list saved during training
    }


def build_applicant(
    age:    float,
    income: float,
    home:   str,
    loan:   float,
    rate:   float,
) -> pd.DataFrame:
    """Creates a single-row DataFrame for one loan applicant."""
    return pd.DataFrame([{
        "person_age"            : age,
        "person_income"         : income,
        "person_home_ownership" : home,
        "loan_amnt"             : loan,
        "loan_int_rate"         : rate,
    }])


def predict_applicant(applicant_df: pd.DataFrame, artifacts: dict) -> dict:
    """
    Applies the same preprocessing used in training, then predicts.

    Pipeline:
        1. Encode person_home_ownership with saved LabelEncoder
        2. Apply log1p to saved skewed_cols only (not all numeric)
        3. Scale with saved StandardScaler
        4. Predict with saved model

    Args:
        applicant_df : Single-row DataFrame from build_applicant()
        artifacts    : Dict returned by load_artifacts()

    Returns:
        dict — prediction (0/1), label, confidence, raw_score
    """
    df = applicant_df.copy()

    # Step 1: Encode home ownership
    try:
        df["person_home_ownership"] = artifacts["le_home"].transform(
            df["person_home_ownership"]
        )
    except ValueError:
        valid = list(artifacts["le_home"].classes_)
        raise ValueError(
            f"Unknown home ownership value: '{applicant_df['person_home_ownership'].values[0]}'.\n"
            f"Valid options: {valid}"
        )

    # Step 2: Apply log1p — uses saved list, NOT df[col].skew()
    for col in artifacts["skewed_cols"]:
        df[col] = np.log1p(df[col])

    # Step 3: Scale
    df_scaled = artifacts["scaler"].transform(df[FEATURES])

    # Step 4: Predict
    raw_score  = float(artifacts["model"].predict_proba(df_scaled)[0][1])
    prediction = 1 if raw_score > 0.5 else 0
    confidence = raw_score if prediction == 1 else 1.0 - raw_score

    return {
        "prediction" : prediction,
        "label"      : LABEL_MAP[prediction],
        "confidence" : confidence,
        "raw_score"  : raw_score,
    }


def demo_prediction():
    """
    Runs the default demo applicant from the notebook's DEMO PREDICTION cell.

    Applicant:
        age=30, income=50000, home=MORTGAGE, loan=15000, rate=12.5
    """
    print("\n💳 Loan Approval — Demo Prediction\n" + "=" * 40)

    artifacts = load_artifacts()

    demo_applicant = build_applicant(
        age    = 30,
        income = 50000,
        home   = "MORTGAGE",
        loan   = 15000,
        rate   = 12.5,
    )

    result = predict_applicant(demo_applicant, artifacts)

    # Matches notebook: print(f"\n🎯 Demo Prediction: Loan {prediction_label}")
    print(f"\n🎯 Demo Prediction: Loan {result['label']}")
    print(f"   Confidence : {result['confidence']:.2%}")
    print(f"   Raw score  : {result['raw_score']:.4f}")
    print("=" * 40)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Loan Approval Prediction — single applicant"
    )
    parser.add_argument("--age",    type=float, default=30,         help="Applicant age")
    parser.add_argument("--income", type=float, default=50000,      help="Annual income ($)")
    parser.add_argument("--home",   type=str,   default="MORTGAGE", help="Home ownership (MORTGAGE/RENT/OWN/OTHER)")
    parser.add_argument("--loan",   type=float, default=15000,      help="Loan amount ($)")
    parser.add_argument("--rate",   type=float, default=12.5,       help="Interest rate (%)")
    args = parser.parse_args()

    artifacts     = load_artifacts()
    applicant     = build_applicant(args.age, args.income, args.home, args.loan, args.rate)
    result        = predict_applicant(applicant, artifacts)

    print("\n" + "=" * 52)
    print(f"  Age        : {int(args.age)}")
    print(f"  Income     : ${args.income:,.0f}")
    print(f"  Ownership  : {args.home}")
    print(f"  Loan amt   : ${args.loan:,.0f}")
    print(f"  Int rate   : {args.rate}%")
    print(f"  Prediction : Loan {result['label']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        demo_prediction()