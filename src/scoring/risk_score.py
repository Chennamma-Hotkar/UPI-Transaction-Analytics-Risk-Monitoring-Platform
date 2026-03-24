import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MODELS_PATH = BASE / "models"

FEATURE_COLS = [
    "amount", "hour", "day_of_week", "is_weekend", "is_night",
    "amount_deviation", "daily_tx_count", "time_gap_minutes",
    "device_variety", "location_variety", "device_encoded", "location_encoded",
    "user_avg_amount", "user_tx_count"
]


def load_models():
    with open(MODELS_PATH / "random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODELS_PATH / "logistic_regression.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(MODELS_PATH / "lr_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_PATH / "isolation_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    return rf, lr, scaler, iso


rf, lr, scaler, iso = load_models()


def compute_risk_score(features: dict) -> dict:
    """
    Given a dict of transaction features, return risk score and label.
    Score: 0.0 - 1.0  |  Label: Low / Medium / High
    """
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0)

    rf_prob = rf.predict_proba(X)[0][1]
    lr_prob = lr.predict_proba(scaler.transform(X))[0][1]
    iso_flag = 1 if iso.predict(X)[0] == -1 else 0

    # Weighted ensemble
    score = round(0.6 * rf_prob + 0.3 * lr_prob + 0.1 * iso_flag, 4)

    if score >= 0.7:
        label = "High"
    elif score >= 0.35:
        label = "Medium"
    else:
        label = "Low"

    return {
        "risk_score": score,
        "risk_label": label,
        "rf_probability": round(rf_prob, 4),
        "lr_probability": round(lr_prob, 4),
        "anomaly_flag": iso_flag
    }


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        result = compute_risk_score(row.to_dict())
        results.append(result)
    scored = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), scored], axis=1)


if __name__ == "__main__":
    sample = {
        "amount": 150000, "hour": 2, "day_of_week": 6, "is_weekend": 1,
        "is_night": 1, "amount_deviation": 8.5, "daily_tx_count": 15,
        "time_gap_minutes": 0.5, "device_variety": 4, "location_variety": 8,
        "device_encoded": 3, "location_encoded": 5,
        "user_avg_amount": 1200, "user_tx_count": 200
    }
    result = compute_risk_score(sample)
    print("Risk Assessment:", result)
