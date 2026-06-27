"""
test_predict.py
---------------
Unit tests for the Loan Approval prediction pipeline.

Run from project root:
    python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── LABEL_MAP tests ───────────────────────────────────────────────────────────

def test_label_map_has_both_keys():
    from predict import LABEL_MAP
    assert 0 in LABEL_MAP and 1 in LABEL_MAP

def test_label_map_rejected_is_0():
    from predict import LABEL_MAP
    assert "Rejected" in LABEL_MAP[0]

def test_label_map_approved_is_1():
    from predict import LABEL_MAP
    assert "Approved" in LABEL_MAP[1]


# ── build_applicant tests ─────────────────────────────────────────────────────

def test_build_applicant_returns_dataframe():
    from predict import build_applicant
    result = build_applicant(30, 50000, "MORTGAGE", 15000, 12.5)
    assert isinstance(result, pd.DataFrame)

def test_build_applicant_shape():
    from predict import build_applicant
    df = build_applicant(30, 50000, "MORTGAGE", 15000, 12.5)
    assert df.shape == (1, 5)

def test_build_applicant_columns():
    from predict import build_applicant
    from config import FEATURES
    df = build_applicant(30, 50000, "MORTGAGE", 15000, 12.5)
    assert list(df.columns) == FEATURES

def test_build_applicant_values():
    from predict import build_applicant
    df = build_applicant(30, 50000, "MORTGAGE", 15000, 12.5)
    assert df["person_age"].values[0] == 30
    assert df["person_income"].values[0] == 50000
    assert df["loan_int_rate"].values[0] == 12.5


# ── load_artifacts tests ──────────────────────────────────────────────────────

def test_load_artifacts_raises_if_missing(tmp_path, monkeypatch):
    """Should raise FileNotFoundError if model files don't exist."""
    monkeypatch.setattr("config.MODEL_PATH", str(tmp_path / "missing.pkl"))
    from predict import load_artifacts
    with pytest.raises(FileNotFoundError):
        load_artifacts()


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_features_count():
    from config import FEATURES
    assert len(FEATURES) == 5

def test_config_required_features_present():
    from config import FEATURES
    for col in ["person_age", "person_income", "loan_amnt", "loan_int_rate", "person_home_ownership"]:
        assert col in FEATURES

def test_config_skew_threshold():
    from config import SKEW_THRESHOLD
    assert 0 < SKEW_THRESHOLD <= 2.0

def test_config_class_names():
    from config import CLASS_NAMES
    assert CLASS_NAMES == ["Rejected", "Approved"]


# ── Integration test (with mock artifacts) ────────────────────────────────────

def test_predict_full_pipeline(tmp_path, monkeypatch):
    """End-to-end predict with dummy model and scaler."""
    import joblib
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    # Dummy artifacts
    le = LabelEncoder()
    le.fit(["MORTGAGE", "RENT", "OWN", "OTHER"])
    scaler = StandardScaler()
    scaler.fit(np.random.rand(50, 5))
    clf = RandomForestClassifier(n_estimators=3, random_state=0)
    clf.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
    skewed = ["person_income", "loan_amnt"]

    joblib.dump(clf,    str(tmp_path / "best_model.pkl"))
    joblib.dump(scaler, str(tmp_path / "scaler.pkl"))
    joblib.dump(le,     str(tmp_path / "le_home.pkl"))
    joblib.dump(skewed, str(tmp_path / "skewed_cols.pkl"))

    monkeypatch.setattr("config.MODEL_PATH",       str(tmp_path / "best_model.pkl"))
    monkeypatch.setattr("config.SCALER_PATH",      str(tmp_path / "scaler.pkl"))
    monkeypatch.setattr("config.LE_HOME_PATH",     str(tmp_path / "le_home.pkl"))
    monkeypatch.setattr("config.SKEWED_COLS_PATH", str(tmp_path / "skewed_cols.pkl"))

    import importlib
    import predict as pm
    importlib.reload(pm)

    arts   = pm.load_artifacts()
    app    = pm.build_applicant(30, 50000, "MORTGAGE", 15000, 12.5)
    result = pm.predict_applicant(app, arts)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["label"] in ("Rejected", "Approved")