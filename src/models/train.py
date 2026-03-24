import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

BASE = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE / "data" / "processed" / "features.csv"
MODELS_PATH = BASE / "models"

FEATURE_COLS = [
    "amount", "hour", "day_of_week", "is_weekend", "is_night",
    "amount_deviation", "daily_tx_count", "time_gap_minutes",
    "device_variety", "location_variety", "device_encoded", "location_encoded",
    "user_avg_amount", "user_tx_count"
]


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]
    return X, y, df


def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_scaled, y_train)
    return lr, scaler


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def train_isolation_forest(X_train):
    iso = IsolationForest(contamination=0.08, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    return iso


def evaluate(name, model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        auc = roc_auc_score(y_test, y_prob)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
        print(f"AUC-ROC: {auc:.4f}")
    else:
        # Isolation Forest: -1 = anomaly, 1 = normal
        y_pred = model.predict(X)
        y_pred_bin = (y_pred == -1).astype(int)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred_bin, target_names=["Legit", "Fraud"]))


def save_model(obj, name):
    path = MODELS_PATH / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    X, y, df = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Logistic Regression...")
    lr, scaler = train_logistic_regression(X_train, y_train)
    evaluate("Logistic Regression", lr, X_test, y_test, scaler)
    save_model(lr, "logistic_regression")
    save_model(scaler, "lr_scaler")

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)
    save_model(rf, "random_forest")

    print("\nTraining Isolation Forest...")
    iso = train_isolation_forest(X_train)
    evaluate("Isolation Forest", iso, X_test, y_test)
    save_model(iso, "isolation_forest")

    print("\nAll models trained and saved.")
