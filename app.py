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

# Load Model menggunakan Native Booster (Tanpa butuh scikit-learn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")

model = xgb.Booster()
model.load_model(MODEL_PATH)

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
    
    scaled_features = (raw_features - MEAN) / SCALE
    
    # XGBoost Native menggunakan DMatrix
    dmatrix = xgb.DMatrix(scaled_features)
    preds = model.predict(dmatrix)
    
    prob = float(preds[0])
    fail_prob = round(prob * 100, 2)
    norm_prob = round((1.0 - prob) * 100, 2)
    is_fail = fail_prob > 50.0
    
    return {
        "is_failure": is_fail,
        "status": "FAILURE DETECTED" if is_fail else "NORMAL",
        "failure_probability": fail_prob,
        "normal_probability": norm_prob
    }
