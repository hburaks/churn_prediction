from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_final_model_lite.json")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "feature_list.json")
DATA_PATH = os.path.join(BASE_DIR, "model_ready_lite_sample.csv")

model = xgb.XGBClassifier()
df_db = pd.DataFrame()
feature_names = []
feature_importance_map = {}

app = FastAPI(
    title="KKBox Churn Prediction API",
    description="Müzik platformu kullanıcıları için ayrılma (churn) riski tahmin sistemi.",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_artifacts():
    global model, df_db, feature_names, feature_importance_map
    
    if os.path.exists(FEATURE_LIST_PATH):
        with open(FEATURE_LIST_PATH, 'r') as f:
            feature_names = json.load(f)
    
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        try:
            importances = model.feature_importances_
            feature_importance_map = dict(zip(feature_names, importances))
        except:
            pass

    if os.path.exists(DATA_PATH):
        df_db = pd.read_csv(DATA_PATH)

def get_user_data(user_id: int):
    if user_id < 0 or user_id >= len(df_db):
        return None
    
    user_row = df_db.iloc[user_id]
    features = user_row[feature_names].values.reshape(1, -1)
    actual_churn = int(user_row.get('is_churn', -1))
    
    return features, actual_churn, user_row

def generate_explanation(user_row: pd.Series) -> List[Dict]:
    reasons = []
    
    if 'is_cancel_sum' in user_row and user_row['is_cancel_sum'] > 0:
        reasons.append({
            "feature": "Cancellation History",
            "value": int(user_row['is_cancel_sum']),
            "impact": "High",
            "message": "User has previously cancelled their subscription."
        })
        
    if 'is_auto_renew_max' in user_row and user_row['is_auto_renew_max'] == 0:
        reasons.append({
            "feature": "Auto-Renewal",
            "value": 0,
            "impact": "Critical",
            "message": "Auto-renewal is turned OFF."
        })

    if 'days_to_expire' in user_row and user_row['days_to_expire'] < 5:
        days = int(user_row['days_to_expire'])
        if days < 0:
            msg = f"Subscription expired {abs(days)} days ago."
        else:
            msg = f"Subscription ends in {days} days."
            
        reasons.append({
            "feature": "Expiration Date",
            "value": days,
            "impact": "High",
            "message": msg
        })
        
    if 'num_100_trend' in user_row and user_row['num_100_trend'] < 0.5:
         reasons.append({
            "feature": "Usage Trend",
            "value": round(user_row['num_100_trend'], 2),
            "impact": "Medium",
            "message": "Listening activity dropped by >50% in last 14 days."
        })

    if not reasons:
        reasons.append({
            "feature": "General Profile",
            "value": "-",
            "impact": "Uncertain",
            "message": "User profile matches high-risk patterns."
        })
        
    return reasons

class PredictionResponse(BaseModel):
    user_id: int
    churn_probability: float
    is_churn_prediction: bool
    risk_level: str
    actual_status: Optional[int] = None

class ExplanationResponse(BaseModel):
    user_id: int
    risk_score: float
    reasons: List[Dict]

class UserStatsResponse(BaseModel):
    user_id: int
    membership_days: int
    total_transactions: int
    days_to_expire: int
    last_active_date: str

@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction API. Visit /docs for documentation."}

@app.get("/users/random")
def get_random_user():
    risk_users = df_db[df_db['is_churn'] == 1].index.tolist()
    if risk_users:
        import random
        return {"user_id": random.choice(risk_users)}
    return {"user_id": 0}

@app.get("/user-stats/{user_id}", response_model=UserStatsResponse)
def get_user_stats(user_id: int):
    data = get_user_data(user_id)
    if not data:
        raise HTTPException(status_code=404, detail="User not found")
    
    _, _, user_row = data
    mem_days = int(user_row.get('membership_days', 365))
    if mem_days < 0: mem_days = 0
    total_trans = int(user_row.get('total_transactions', 12))
    days_expire = int(user_row.get('days_to_expire', 30))
    last_active = "2017-03-31"

    return {
        "user_id": user_id,
        "membership_days": mem_days,
        "total_transactions": total_trans,
        "days_to_expire": days_expire,
        "last_active_date": last_active
    }

@app.get("/predict/{user_id}", response_model=PredictionResponse)
def predict_churn(user_id: int):
    data = get_user_data(user_id)
    if not data:
        raise HTTPException(status_code=404, detail="User not found")
    
    features, actual_churn, _ = data
    prob = float(model.predict_proba(features)[:, 1][0])
    threshold = 0.9
    prediction = bool(prob >= threshold)
    
    if prob > 0.9: risk_level = "Critical"
    elif prob > 0.7: risk_level = "High"
    elif prob > 0.4: risk_level = "Moderate"
    else: risk_level = "Low"
    
    return {
        "user_id": user_id,
        "churn_probability": round(prob, 4),
        "is_churn_prediction": prediction,
        "risk_level": risk_level,
        "actual_status": actual_churn
    }

@app.get("/explain/{user_id}", response_model=ExplanationResponse)
def explain_churn(user_id: int):
    data = get_user_data(user_id)
    if not data:
        raise HTTPException(status_code=404, detail="User not found")
    
    features, _, user_row = data
    prob = float(model.predict_proba(features)[:, 1][0])
    reasons = generate_explanation(user_row)
    
    if prob < 0.5:
        reasons = [{
            "feature": "-", 
            "value": "-", 
            "impact": "None", 
            "message": "User appears safe based on current activity."
        }]

    return {
        "user_id": user_id,
        "risk_score": round(prob, 4),
        "reasons": reasons
    }
