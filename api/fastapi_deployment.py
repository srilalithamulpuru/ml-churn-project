from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

# Load scaler & model
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/logistic_model.pkl")

class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    AvgCharges: float

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: ChurnInput):

    X = np.array([[  
        data.gender,
        data.SeniorCitizen,
        data.tenure,
        data.PhoneService,
        data.MultipleLines,
        data.OnlineSecurity,
        data.OnlineBackup,
        data.DeviceProtection,
        data.TechSupport,
        data.MonthlyCharges,
        data.TotalCharges,
        data.InternetService_Fiber_optic,
        data.InternetService_No,
        data.Contract_One_year,
        data.Contract_Two_year,
        data.AvgCharges
    ]])

    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    return {
        "churn": int(pred),
        "probability": round(float(prob), 4),
        "label": "Churn" if pred == 1 else "No Churn"
    }
