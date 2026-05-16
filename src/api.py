from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
import os

app = FastAPI(
    title="Credit Risk API",
    description="API for predicting Probability of Default and Expected Loss",
    version="1.0.0"
)

# Load Models
MODEL_PATH = os.path.join("models", "lr_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessing_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"Error loading models: {e}")

LGD = 0.45

# Setup Database for Audit Logging
DB_PATH = 'data/audit_logs.db'
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            loan_amt REAL,
            total_debt REAL,
            income REAL,
            fico INTEGER,
            years_emp INTEGER,
            credit_lines INTEGER,
            raw_pd REAL,
            blended_pd REAL,
            expected_loss REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class LoanApplication(BaseModel):
    loan_amt_outstanding: float
    total_debt_outstanding: float
    income: float
    fico_score: int
    years_employed: int
    credit_lines_outstanding: int

class PredictionResult(BaseModel):
    probability_of_default: float
    expected_loss: float
    risk_category: str

def calculate_risk_category(pd_val: float) -> str:
    if pd_val < 0.05: return "Low Risk"
    if pd_val < 0.15: return "Moderate Risk"
    if pd_val < 0.30: return "Moderate to High Risk"
    return "High Risk"

@app.post("/predict", response_model=PredictionResult)
def predict_default_risk(app_data: LoanApplication):
    if app_data.income <= 0:
        raise HTTPException(status_code=400, detail="Income must be strictly positive.")
    if app_data.loan_amt_outstanding < 0:
        raise HTTPException(status_code=400, detail="Loan amount must be positive.")

    # Clamp features
    clamped_credit_lines = min(app_data.credit_lines_outstanding, 5)
    
    # Derived features
    dti = app_data.total_debt_outstanding / app_data.income
    lti = app_data.loan_amt_outstanding / app_data.income
    cu = clamped_credit_lines / 5.0
    
    input_df = pd.DataFrame([{
        'loan_amt_outstanding': app_data.loan_amt_outstanding,
        'total_debt_outstanding': app_data.total_debt_outstanding,
        'income': app_data.income,
        'fico_score': app_data.fico_score,
        'years_employed': app_data.years_employed,
        'credit_lines_outstanding': clamped_credit_lines,
        'dti_ratio': dti,
        'lti_ratio': lti,
        'credit_utilisation': cu
    }])
    
    feature_order = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 
                   'fico_score', 'years_employed', 'credit_lines_outstanding',
                   'dti_ratio', 'lti_ratio', 'credit_utilisation']
                   
    X_scaled = preprocessor.transform(input_df[feature_order])
    raw_pd = model.predict_proba(X_scaled)[0, 1]
    
    # Real-World Smoothing Heuristic
    heuristic_pd = 0.05
    fico = app_data.fico_score
    if fico >= 800: heuristic_pd -= 0.04
    elif fico >= 750: heuristic_pd -= 0.02
    elif fico < 600: heuristic_pd += 0.30
    elif fico < 650: heuristic_pd += 0.15
    elif fico < 700: heuristic_pd += 0.05
    
    if dti > 2.0: heuristic_pd += 0.25
    elif dti > 0.5: heuristic_pd += 0.05
    elif dti < 0.2: heuristic_pd -= 0.01
    
    if app_data.years_employed < 2: heuristic_pd += 0.08
    
    blended_pd = (raw_pd * 0.1) + (heuristic_pd * 0.9)
    pd_val = max(blended_pd, 0.02)
    pd_val = min(pd_val, 0.99)
    
    el = pd_val * app_data.loan_amt_outstanding * LGD
    
    # Log to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO prediction_logs (timestamp, loan_amt, total_debt, income, fico, years_emp, credit_lines, raw_pd, blended_pd, expected_loss)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), app_data.loan_amt_outstanding, app_data.total_debt_outstanding, 
          app_data.income, app_data.fico_score, app_data.years_employed, app_data.credit_lines_outstanding, 
          raw_pd, pd_val, el))
    conn.commit()
    conn.close()
    
    return PredictionResult(
        probability_of_default=round(pd_val, 4),
        expected_loss=round(el, 2),
        risk_category=calculate_risk_category(pd_val)
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}
