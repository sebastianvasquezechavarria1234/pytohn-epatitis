# main.py
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Intentamos usar joblib (más seguro para modelos sklearn), si falla usamos pickle
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

import pickle

APP_PORT = int(os.environ.get("PORT", 5000))

app = Flask(__name__)
CORS(app)


def safe_load(path):
    """Intenta cargar con joblib, si no con pickle."""
    if joblib_load:
        try:
            return joblib_load(path)
        except Exception:
            pass
    # fallback a pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# Rutas de los archivos (mismo directorio)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_regresion_logistica.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Cargamos artefactos al iniciar
try:
    model = safe_load(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo desde {MODEL_PATH}: {e}")
    model = None

try:
    scaler = safe_load(SCALER_PATH)
except Exception as e:
    print(f"[WARN] No se pudo cargar el scaler desde {SCALER_PATH}: {e}")
    scaler = None


@app.route("/")
def index():
    return jsonify({
        "status": "ok",
        "message": "API de predicción lista. POST /predict con JSON {\"features\": [...]} o un objeto de características."
    })


def prepare_features_from_json(json_data):
    """
    Acepta:
    - {"features": [v1, v2, ...]} -> usa esa lista
    - {"feature1": v1, "feature2": v2, ...} -> intenta ordenar según model.feature_names_in_
    Devuelve: numpy array shape (1, n_features)
    """
    if "features" in json_data:
        features = json_data["features"]
        arr = np.array(features, dtype=float).reshape(1, -1)
        return arr

    # Si viene como dict
    if isinstance(json_data, dict):
        data = json_data.copy()

        # si el modelo tiene feature_names_in_ usamos ese orden
        if hasattr(model, "feature_names_in_"):
            keys_order = list(model.feature_names_in_)
            try:
                values = [float(data[k]) for k in keys_order]
                return np.array(values, dtype=float).reshape(1, -1)
            except Exception:
                pass

        # fallback: orden alfabético
        keys = sorted(k for k in data.keys())
        values = [float(data[k]) for k in keys]
        return np.array(values, dtype=float).reshape(1, -1)

    raise ValueError("Formato de entrada no soportado")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado en el servidor."}), 500

    try:
        json_data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido o cabecera Content-Type faltante."}), 400

    try:
        X = prepare_features_from_json(json_data)

        # Aplica scaler si existe
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                print("[WARN] scaler.transform falló:", e)

        # Predicción de clase
        preds = model.predict(X)
        raw_pred = preds[0].item() if hasattr(preds[0], "item") else int(preds[0])

        # Mapeo de clases: tu modelo usa 0 y 2
        label_map = {
            0: "Vive",
            1: "Muere",  # por si algún día aparece
            2: "Muere"
        }

        prediction_label = label_map.get(raw_pred, f"Clase {raw_pred}")

        # Probabilidades si el modelo tiene predict_proba
        probabilities = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0].tolist()

            # Suponiendo orden: [0, 2]
            probabilities = {
                "Vive": proba[0],
                "Muere": proba[1]
            }

        return jsonify({
            "prediction": prediction_label,
            "prediction_raw": raw_pred,
            "probabilities": probabilities
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Error interno al predecir", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
