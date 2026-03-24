from flask import Flask, request, jsonify
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.scoring.risk_score import compute_risk_score, score_dataframe
from src.preprocessing.preprocess import load_data, engineer_features

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "UPI Risk Monitor API", "version": "1.0"})


@app.route("/score", methods=["POST"])
def score_transaction():
    """
    POST /score
    Body: JSON with transaction features
    Returns: risk_score, risk_label, model probabilities
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input provided"}), 400
    try:
        result = compute_risk_score(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transactions", methods=["GET"])
def get_transactions():
    """GET /transactions — returns all transactions with risk scores"""
    try:
        df = load_data()
        df = engineer_features(df)
        df_scored = score_dataframe(df)
        cols = ["transaction_id", "user_id", "merchant_id", "amount",
                "timestamp", "location", "device_info",
                "risk_score", "risk_label", "is_fraud"]
        return jsonify(df_scored[cols].head(100).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summary", methods=["GET"])
def get_summary():
    """GET /summary — returns overall risk distribution stats"""
    try:
        df = load_data()
        df = engineer_features(df)
        df_scored = score_dataframe(df)

        total = len(df_scored)
        summary = {
            "total_transactions": total,
            "fraud_detected": int(df_scored["is_fraud"].sum()),
            "risk_distribution": df_scored["risk_label"].value_counts().to_dict(),
            "avg_risk_score": round(float(df_scored["risk_score"].mean()), 4),
            "high_risk_count": int((df_scored["risk_label"] == "High").sum()),
            "total_amount": round(float(df_scored["amount"].sum()), 2),
            "flagged_amount": round(
                float(df_scored[df_scored["risk_label"] == "High"]["amount"].sum()), 2
            )
        }
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """GET /alerts — returns High risk transactions"""
    try:
        df = load_data()
        df = engineer_features(df)
        df_scored = score_dataframe(df)
        alerts = df_scored[df_scored["risk_label"] == "High"]
        cols = ["transaction_id", "user_id", "amount", "timestamp",
                "location", "device_info", "risk_score"]
        return jsonify(alerts[cols].to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
