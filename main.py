import os
import json
import logging
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    Age: float = Field(..., description="Patient age")
    Sex: int = Field(..., description="1 = Male, 2 = Female")
    Estado_Civil: int = Field(..., description="Marital status code")
    Ciudad: int = Field(..., description="City code")
    Steroid: int = Field(..., description="1 = No, 2 = Yes")
    Antivirals: int = Field(..., description="1 = No, 2 = Yes")
    Fatigue: int = Field(..., description="1 = No, 2 = Yes")
    Malaise: int = Field(..., description="1 = No, 2 = Yes")
    Anorexia: int = Field(..., description="1 = No, 2 = Yes")
    Liver_Big: int = Field(..., description="1 = No, 2 = Yes")
    Liver_Firm: int = Field(..., description="1 = No, 2 = Yes")
    Spleen_Palpable: int = Field(..., description="1 = No, 2 = Yes")
    Spiders: int = Field(..., description="1 = No, 2 = Yes")
    Ascites: int = Field(..., description="1 = No, 2 = Yes")
    Varices: int = Field(..., description="1 = No, 2 = Yes")
    Bilirubin: float = Field(..., description="Bilirubin level")
    Alk_Phosphate: float = Field(..., description="Alkaline Phosphate level")
    Sgot: float = Field(..., description="SGOT level")
    Albumin: float = Field(..., description="Albumin level")
    Protime: float = Field(..., description="Protime level")
    Histology: int = Field(..., description="1 = No, 2 = Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 30.0, "Sex": 1, "Estado_Civil": 1, "Ciudad": 1,
                "Steroid": 1, "Antivirals": 2, "Fatigue": 1, "Malaise": 1,
                "Anorexia": 1, "Liver_Big": 2, "Liver_Firm": 1, "Spleen_Palpable": 1,
                "Spiders": 1, "Ascites": 1, "Varices": 1, "Bilirubin": 1.0,
                "Alk_Phosphate": 85.0, "Sgot": 18.0, "Albumin": 4.0,
                "Protime": 100.0, "Histology": 1
            }
        }

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo_regresion_logistica.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
INFO_PATH = os.path.join(BASE_DIR, "modelo_regresion_logistica_info.json")

# State
ml_models = {}

def safe_load(path):
    """Attempt to load using joblib, fallback to pickle."""
    if joblib_load:
        try:
            return joblib_load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and scaler
    try:
        ml_models["model"] = safe_load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        ml_models["model"] = None

    try:
        ml_models["scaler"] = safe_load(SCALER_PATH)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load scaler: {e}")
        ml_models["scaler"] = None

    # Load metadata
    try:
        with open(INFO_PATH, "r") as f:
            ml_models["info"] = json.load(f)
        logger.info(f"Metadata loaded from {INFO_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load metadata: {e}")
        ml_models["info"] = {}

    yield
    ml_models.clear()

app = FastAPI(
    title="Hepatitis Prediction API",
    description="Professional API for medical outcomes prediction based on logistic regression.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Hepatitis Prediction API is ready.",
        "docs": "/docs"
    }

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Predict medical outcome based on 21 patient features.
    """
    model = ml_models.get("model")
    scaler = ml_models.get("scaler")
    info = ml_models.get("info")

    if not model:
        raise HTTPException(status_code=500, detail="Prediction model not loaded.")

    try:
        # Extract features in the correct order as per model metadata
        feature_names = info.get("features", [])
        input_dict = data.model_dump()
        
        if feature_names:
            ordered_features = [input_dict[name] for name in feature_names]
        else:
            # Fallback if metadata is missing (not ideal)
            ordered_features = list(input_dict.values())
            
        features_array = np.array(ordered_features, dtype=float).reshape(1, -1)

        # Apply scaling if available
        if scaler:
            features_array = scaler.transform(features_array)

        # Execute prediction
        prediction_raw = model.predict(features_array)[0]
        prediction_raw = int(prediction_raw)

        # Map labels from metadata if available, otherwise use hardcoded defaults
        # According to main.py original code: 0 -> Vive, 2 -> Muere
        label_map = {0: "Vive", 1: "Muere", 2: "Muere"}
        prediction_label = label_map.get(prediction_raw, f"Class {prediction_raw}")

        response = {
            "prediction": prediction_label,
            "prediction_raw": prediction_raw,
            "status": "success"
        }

        # Include probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_array)[0].tolist()
            response["probabilities"] = {
                "Vive": round(probs[0], 4),
                "Muere": round(probs[1] if len(probs) > 1 else 0.0, 4)
            }

        return response

    except Exception as e:
        logger.error(f"Prediction process failed: {e}")
        raise HTTPException(status_code=400, detail="Internal processing error during prediction.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
