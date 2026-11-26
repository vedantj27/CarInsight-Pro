"""Train and compare four regression models for CarInsight Pro.

Run from the repository root:
    python train_model.py
The script creates artifacts/model_bundle.joblib and reports model metrics.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cardekho_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET = "selling_price"
CATEGORICAL = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]
NUMERIC = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
FEATURES = CATEGORICAL + NUMERIC

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put cardekho_dataset.csv in the data/ folder."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0", "car_name"], errors="ignore")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[(df[TARGET] > 10000) & (df["seats"] > 0)]
    # Avoid target leakage: no feature is calculated from selling_price.
    return df[FEATURES + [TARGET]].copy()

def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

def main():
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, min_samples_leaf=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=24, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3,
            loss="huber", random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.035, max_depth=7,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.05, reg_lambda=1.5, objective="reg:squarederror",
            eval_metric="rmse", n_jobs=-1, random_state=42
        )
    }
    results = []
    fitted = {}
    preprocessor = build_preprocessor()
    # Fit preprocessing once on training data; all candidates use identical features.
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    for name, estimator in models.items():
        print(f"Training {name}...")
        estimator.fit(X_train_t, y_train)
        pred = estimator.predict(X_test_t)
        results.append({
            "model": name,
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred)))
        })
        fitted[name] = estimator

    results = sorted(results, key=lambda x: x["r2"], reverse=True)
    # Choose the highest R² model on the same held-out test split.
    winner = results[0]["model"]
    bundle = {
        "preprocessor": preprocessor,
        "model": fitted[winner],
        "model_name": winner,
        "metrics": results,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model_bundle.joblib", compress=3)
    pd.DataFrame(results).to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    # Compact metadata for dashboards and API.
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(bundle | {"preprocessor": None, "model": None}, indent=2),
        encoding="utf-8"
    )
    print("\nHeld-out test results (sorted by R²):")
    print(pd.DataFrame(results).to_string(index=False, formatters={
        "r2": "{:.4f}".format, "mae": "₹{:,.0f}".format, "rmse": "₹{:,.0f}".format
    }))
    print(f"\nSelected model: {winner}")
    if winner != "XGBoost":
        print("NOTE: XGBoost was evaluated but did not achieve the highest R² on this split.")
        print("Update resume wording to reflect the measured winner, or tune/re-evaluate transparently.")
    else:
        print("XGBoost achieved the highest R² on this held-out test split.")
    print(f"Saved: {ARTIFACT_DIR / 'model_bundle.joblib'}")

if __name__ == "__main__":
    main()
