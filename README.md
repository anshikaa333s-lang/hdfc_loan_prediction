# 💳 Loan Approval Prediction

A machine learning pipeline that predicts whether a loan application should be
**Approved** or **Rejected**, based on applicant financial profile data.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-0081A5?style=flat-square)](https://xgboost.readthedocs.io)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

> *Predicting credit risk with interpretable machine learning.*

---

## 📌 Project Overview

Banks and financial institutions process thousands of loan applications daily.
Manual review is slow and inconsistent. This project trains and compares four
classifiers — Logistic Regression, Random Forest, XGBoost, and Gradient Boosting
— on real-world credit risk data to predict loan approval status.

| Label | Class | Meaning |
|-------|-------|---------|
| `1` | **Approved** | Applicant is likely to repay |
| `0` | **Rejected** | High credit risk detected |

> **Note:** The model optimises for overall accuracy. It does not apply a
> risk-adjusted threshold (approving a bad loan is riskier than rejecting a
> good one). This is a known limitation — see Future Work.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle — Loan Approval Prediction](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction) |
| File | `credit_risk_dataset.csv` |
| Target | `loan_status` (0 = Rejected, 1 = Approved) |

### Features used

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | Numeric | Applicant age (clipped 18–100) |
| `person_income` | Numeric | Annual income (clipped 0–5M) |
| `person_home_ownership` | Categorical | MORTGAGE / RENT / OWN / OTHER |
| `loan_amnt` | Numeric | Requested loan amount (clipped 0–500K) |
| `loan_int_rate` | Numeric | Interest rate % (clipped 0–40) |

> ⚠️ Raw data is excluded via `.gitignore`. Download from Kaggle and place at
> `data/raw/credit_risk_dataset.csv`.

---

## 🧠 Models Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline — linear decision boundary |
| Random Forest | Ensemble of 200 trees |
| XGBoost | Gradient boosted trees — typically best performer |
| Gradient Boosting | sklearn's GBM implementation |

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | TBD |
| Random Forest | TBD |
| XGBoost | TBD |
| Gradient Boosting | TBD |

> Fill in after running `python src/train.py`.

---

## 📁 Repository Structure

```
Loan_Approval_Prediction/
│
├── src/
│   ├── config.py           ← all hyperparameters & paths (edit here)
│   ├── data_loader.py      ← Kaggle download + missing value handling
│   ├── preprocess.py       ← encoding, skew correction, split, scaling
│   ├── train.py            ← trains all 4 models, saves best to disk
│   ├── evaluate.py         ← ROC curve, feature importances, report
│   ├── visualise.py        ← EDA plots (histograms, boxplots, heatmap)
│   └── predict.py          ← single applicant CLI inference
│
├── tests/
│   └── test_predict.py     ← unit tests for predict pipeline
│
├── notebooks/
│   └── loan_approval_prediction.ipynb   ← full Colab notebook
│
├── docs/
│   └── report.md           ← methodology & results write-up
│
├── data/
│   └── raw/                ← put credit_risk_dataset.csv here
│
├── models/                 ← saved .pkl artifacts after training
├── results/figures/        ← all plots output here
├── results/metrics/        ← model_results.csv saved here
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Loan_Approval_Prediction.git
cd Loan_Approval_Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle credentials
```bash
# Place kaggle.json at:
~/.kaggle/kaggle.json

# Or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 4. Run the full pipeline
```bash
# Step 1 — EDA visualisations
python src/visualise.py

# Step 2 — Train all models, save best
python src/train.py

# Step 3 — Evaluate, generate ROC curve + feature importances
python src/evaluate.py

# Step 4 — Predict for a single applicant (demo)
python src/predict.py

# Step 4b — Custom applicant via CLI
python src/predict.py --age 25 --income 35000 --home RENT --loan 10000 --rate 14.0

# Step 5 — Run tests
python -m pytest tests/ -v
```

---

## 🔬 Key Design Decisions

### Skew Correction
Highly skewed features (`person_income`, `loan_amnt`) receive `log1p`
transformation to normalise their distributions before training.
The list of skewed columns is **saved to disk** alongside the model so
that new inputs receive the exact same transformation at inference time.

### Feature Importance
All tree-based models expose `feature_importances_`. The evaluate script
generates a horizontal bar chart showing which applicant attributes drive
loan approval decisions most — critical for model interpretability.

### Model Persistence
All artifacts are saved after training:
- `models/best_model.pkl` — best classifier
- `models/scaler.pkl` — fitted StandardScaler
- `models/le_home.pkl` — fitted LabelEncoder for home ownership
- `models/skewed_cols.pkl` — list of log1p-transformed columns

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Models, preprocessing, metrics |
| XGBoost | Gradient boosted classifier |
| matplotlib, seaborn | Visualisation |
| joblib | Model serialisation |
| kagglehub | Dataset download |
| pytest | Unit testing |

---

## 🔮 Future Work

- Lower the prediction threshold (e.g. 0.3) to reduce false approvals
- Add `class_weight='balanced'` to handle class imbalance
- SHAP values for per-applicant explanation
- Hyperparameter tuning with GridSearchCV or Optuna
- Streamlit web app for non-technical users
- API deployment with FastAPI

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**[Your Name]** — Data Science | Machine Learning | Finance

[LinkedIn](#) · [GitHub](#)

---

> *"Better models mean fewer wrong loan decisions — for both the bank and the borrower."*