import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import xgboost as xgb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model XGBoost Native dengan Path Aman untuk Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# 2. Angka Mean & Scale manual kamu (5 fitur)
MEAN = np.array([300.00545, 310.0060625, 1539.356875, 40.0033625, 107.685])
SCALE = np.array([1.996719258558899, 1.4793392092734339, 180.9716311061885, 10.018919350089297, 63.608026419627265])

class InputData(BaseModel):
    air_temp: float
    proc_temp: float
    rot_speed: float
    torque: float
    tool_wear: float

@app.get("/")
def read_root():
    return {"status": "Backend FastAPI Active!"}

@app.post("/predict")
def predict(data: InputData):
    raw_features = np.array([[
        data.air_temp,
        data.proc_temp,
        data.rot_speed,
        data.torque,
        data.tool_wear
    ]])
    
    # Scaling manual tanpa scikit-learn!
    scaled_features = (raw_features - MEAN) / SCALE
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    probs = model.predict_proba(scaled_features)[0]
    
    fail_prob = round(float(probs[1]) * 100, 2)
    norm_prob = round(float(probs[0]) * 100, 2)
    is_fail = bool(prediction == 1)
    
    return {
        "is_failure": is_fail,
        "status": "FAILURE DETECTED" if is_fail else "NORMAL",
        "failure_probability": fail_prob,
        "normal_probability": norm_prob
    }