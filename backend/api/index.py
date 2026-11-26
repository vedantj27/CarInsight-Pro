"""Flask API for CarInsight Pro. Render entrypoint: api.index:app"""
from pathlib import Path
import os
import threading
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "model_bundle.joblib"
DATA_PATH = ROOT / "data" / "cardekho_dataset.csv"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

allowed_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
CORS(app, resources={r"/*": {"origins": allowed_origins}})

_bundle = None
_data = None
_explainer = None
_lock = threading.Lock()

def get_bundle():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not BUNDLE_PATH.exists():
                    raise FileNotFoundError("Model artifact missing. Add artifacts/model_bundle.joblib.")
                _bundle = joblib.load(BUNDLE_PATH)
    return _bundle

def get_data():
    global _data
    if _data is None and DATA_PATH.exists():
        _data = pd.read_csv(DATA_PATH).drop(columns=["Unnamed: 0", "car_name"], errors="ignore")
    return _data

def get_explainer():
    global _explainer
    if _explainer is None:
        with _lock:
            if _explainer is None:
                _explainer = shap.TreeExplainer(get_bundle()["model"])
    return _explainer

def prepare_records(records):
    bundle = get_bundle()
    if not isinstance(records, list) or not records:
        raise ValueError("Provide a non-empty list of vehicle records.")
    frame = pd.DataFrame(records)
    missing = [c for c in bundle["features"] if c not in frame.columns]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    frame = frame[bundle["features"]].copy()
    for col in bundle["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[bundle["numeric"]].isnull().any().any():
        raise ValueError("Numeric vehicle fields must contain valid numbers.")
    for col in bundle["categorical"]:
        frame[col] = frame[col].astype(str)
    return frame

@app.get("/")
def root():
    return jsonify({"name":"CarInsight Pro API", "status":"ok", "docs":"Use /health, /metadata, /predict, /batch-predict, /explain"})

@app.get("/health")
def health():
    try:
        bundle = get_bundle()
        return jsonify({"status":"ok", "model":bundle["model_name"], "version":"1.0.0"})
    except Exception as exc:
        return jsonify({"status":"not_ready", "error":str(exc)}), 503

@app.get("/metadata")
def metadata():
    try:
        bundle = get_bundle()
        df = get_data()
        if df is None:
            return jsonify({"error":"Dataset metadata is unavailable."}), 503
        def values(col):
            return sorted(df[col].dropna().astype(str).unique().tolist())
        return jsonify({
            "features":bundle["features"],
            "model_name":bundle["model_name"],
            "metrics":bundle["metrics"],
            "brands":values("brand"),
            "models_by_brand": {str(k): sorted(v.dropna().astype(str).unique().tolist())
                                for k,v in df.groupby("brand")["model"]},
            "seller_types":values("seller_type"),
            "fuel_types":values("fuel_type"),
            "transmission_types":values("transmission_type"),
            "seats":sorted([int(x) for x in df["seats"].dropna().unique()]),
            "dataset_rows":int(len(df))
        })
    except Exception as exc:
        return jsonify({"error":str(exc)}), 500

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        record = payload.get("record", payload)
        frame = prepare_records([record])
        bundle = get_bundle()
        transformed = bundle["preprocessor"].transform(frame)
        price = float(bundle["model"].predict(transformed)[0])
        return jsonify({"prediction":round(price,2), "currency":"INR", "model":bundle["model_name"]})
    except ValueError as exc:
        return jsonify({"error":str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error":"Prediction failed. Check server logs."}), 500

@app.post("/batch-predict")
def batch_predict():
    try:
        payload = request.get_json(silent=True) or {}
        frame = prepare_records(payload.get("records"))
        bundle = get_bundle()
        transformed = bundle["preprocessor"].transform(frame)
        predictions = bundle["model"].predict(transformed)
        results = []
        for record, price in zip(frame.to_dict(orient="records"), predictions):
            results.append({**record, "predicted_price":round(float(price),2)})
        return jsonify({"count":len(results), "model":bundle["model_name"], "results":results})
    except ValueError as exc:
        return jsonify({"error":str(exc)}), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error":"Batch prediction failed. Check server logs."}), 500

@app.post("/explain")
def explain():
    try:
        payload = request.get_json(silent=True) or {}
        frame = prepare_records([payload.get("record", payload)])
        bundle = get_bundle()
        transformed = bundle["preprocessor"].transform(frame)
        explainer = get_explainer()
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[0, :, 0]
        else:
            values = values[0]
        feature_names = bundle["preprocessor"].get_feature_names_out().tolist()
        base_values = np.asarray(explanation.base_values).reshape(-1)
        base_value = float(base_values[0])
        top = sorted(
            [{"feature":str(name).replace("numeric__","").replace("categorical__",""),
              "impact":float(value)} for name,value in zip(feature_names, values)],
            key=lambda x:abs(x["impact"]), reverse=True
        )[:12]
        return jsonify({
            "base_value":base_value,
            "prediction":float(bundle["model"].predict(transformed)[0]),
            "features":top,
            "interpretation":"Positive SHAP values increase the estimate; negative values decrease it relative to the model baseline."
        })
    except ValueError as exc:
        return jsonify({"error":str(exc)}), 400
    except Exception:
        app.logger.exception("SHAP explanation failed")
        return jsonify({"error":"SHAP explanation failed. Check server logs."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
