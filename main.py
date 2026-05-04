import os
import json
import logging
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

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
async def predict(data: dict):
    model = ml_models.get("model")
    scaler = ml_models.get("scaler")
    info = ml_models.get("info")

    if not model:
        raise HTTPException(status_code=500, detail="Prediction model not loaded.")

    try:
        # Initial implementation (to be refined in next commit with Pydantic)
        if "features" in data:
            features = np.array(data["features"], dtype=float).reshape(1, -1)
        else:
            # Sort features based on metadata if available
            feature_names = info.get("features", [])
            if feature_names and all(k in data for k in feature_names):
                features = np.array([float(data[k]) for k in feature_names]).reshape(1, -1)
            else:
                # Fallback to sorted keys
                sorted_keys = sorted(data.keys())
                features = np.array([float(data[k]) for k in sorted_keys]).reshape(1, -1)

        if scaler:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]
        prediction = int(prediction)

        # Mapping labels
        label_map = {0: "Vive", 1: "Muere", 2: "Muere"}
        result = label_map.get(prediction, f"Class {prediction}")

        response = {
            "prediction": result,
            "prediction_raw": prediction
        }

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0].tolist()
            response["probabilities"] = {
                "Vive": probs[0],
                "Muere": probs[1] if len(probs) > 1 else 0.0
            }

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
