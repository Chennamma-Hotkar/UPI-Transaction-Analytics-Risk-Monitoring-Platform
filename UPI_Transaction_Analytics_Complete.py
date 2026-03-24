"""
UPI Transaction Analytics & Risk Monitoring Platform
Full ML Pipeline using PaySim1 Dataset
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent
DATA_RAW    = BASE / "data" / "raw" / "paysim.csv"
DATA_PROC   = BASE / "data" / "processed"
MODELS_DIR  = BASE / "models"
REPORTS_DIR = BASE / "reports"

for d in [DATA_PROC, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
PALETTE = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

print("=" * 65)
print("  UPI Transaction Analytics & Risk Monitoring Platform")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Load Data
# ══════════════════════════════════════════════════════════════════
print("\n[1/11] Loading PaySim dataset...")
df = pd.read_csv(DATA_RAW)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Fraud records: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.3f}%)")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Data Cleaning
# ══════════════════════════════════════════════════════════════════
print("\n[2/11] Cleaning data...")
df = df.drop(columns=["isFlaggedFraud"], errors="ignore")
df = df.dropna()
df = df.drop_duplicates()

# Encode transaction type
le = LabelEncoder()
df["type_encoded"] = le.fit_transform(df["type"])

# Sender/Receiver type flag (C = customer, M = merchant)
df["orig_is_merchant"] = df["nameOrig"].str.startswith("M").astype(int)
df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(int)

print(f"  Clean rows: {len(df):,}")
print(f"  Transaction types: {df['type'].unique().tolist()}")

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Engineering
# ══════════════════════════════════════════════════════════════════
print("\n[3/11] Engineering features...")

# Balance-based features
df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["balance_mismatch"]  = (df["balance_diff_orig"] - df["amount"]).abs()
df["dest_received_ratio"] = np.where(
    df["amount"] > 0, df["balance_diff_dest"] / df["amount"], 0
)

# Zero balance flags
df["orig_zero_after"] = (df["newbalanceOrig"] == 0).astype(int)
df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)

# Hour-of-day proxy from step (step = hour in simulation)
df["hour"] = df["step"] % 24
df["is_night"] = df["hour"].between(22, 6).astype(int)

# Amount log-transform
df["log_amount"] = np.log1p(df["amount"])

# User-level aggregate features
print("  Computing user-level aggregates...")
user_stats = df.groupby("nameOrig").agg(
    user_tx_count=("amount", "count"),
    user_avg_amount=("amount", "mean"),
    user_total_amount=("amount", "sum")
).reset_index().rename(columns={"nameOrig": "nameOrig_key"})

df["nameOrig_key"] = df["nameOrig"]
df = df.merge(user_stats, on="nameOrig_key", how="left")
df["amount_deviation"] = (
    (df["amount"] - df["user_avg_amount"]) /
    (df["user_avg_amount"].replace(0, 1))
)
df = df.drop(columns=["nameOrig_key"])

FEATURES = [
    "amount", "log_amount", "type_encoded", "step", "hour", "is_night",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest", "balance_mismatch",
    "dest_received_ratio", "orig_zero_after", "dest_zero_before",
    "orig_is_merchant", "dest_is_merchant",
    "user_tx_count", "user_avg_amount", "amount_deviation"
]

X = df[FEATURES].fillna(0)
y = df["isFraud"]

# Save processed data (sample for speed)
df.to_csv(DATA_PROC / "paysim_features.csv", index=False)
print(f"  Features engineered: {len(FEATURES)}")
print(f"  Saved: data/processed/paysim_features.csv")

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Sample & Split (use 200k rows for speed)
# ══════════════════════════════════════════════════════════════════
print("\n[4/11] Sampling & splitting data...")
SAMPLE_SIZE = 200_000
if len(df) > SAMPLE_SIZE:
    fraud_rows = df[df["isFraud"] == 1]
    legit_rows = df[df["isFraud"] == 0].sample(
        SAMPLE_SIZE - len(fraud_rows), random_state=42
    )
    df_sample = pd.concat([fraud_rows, legit_rows]).sample(frac=1, random_state=42)
else:
    df_sample = df

X_s = df_sample[FEATURES].fillna(0)
y_s = df_sample["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.2, stratify=y_s, random_state=42
)

# SMOTE to handle class imbalance
print("  Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print(f"  Train: {len(X_train_bal):,}  |  Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — EDA Charts
# ══════════════════════════════════════════════════════════════════
print("\n[5/11] Generating EDA charts...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("PaySim EDA — UPI Risk Platform", fontsize=15, fontweight="bold")

# Transaction type distribution
type_counts = df["type"].value_counts()
axes[0,0].bar(type_counts.index, type_counts.values, color="#3498db", edgecolor="white")
axes[0,0].set_title("Transaction Type Distribution")
axes[0,0].set_ylabel("Count")

# Fraud by type
fraud_by_type = df.groupby("type")["isFraud"].sum().sort_values(ascending=False)
axes[0,1].bar(fraud_by_type.index, fraud_by_type.values, color="#e74c3c", edgecolor="white")
axes[0,1].set_title("Fraud Count by Transaction Type")
axes[0,1].set_ylabel("Fraud Count")

# Amount distribution (legit vs fraud)
axes[0,2].hist(np.log1p(df[df["isFraud"]==0]["amount"]), bins=50,
               alpha=0.6, color="#2ecc71", label="Legit")
axes[0,2].hist(np.log1p(df[df["isFraud"]==1]["amount"]), bins=50,
               alpha=0.6, color="#e74c3c", label="Fraud")
axes[0,2].set_title("Log Amount: Legit vs Fraud")
axes[0,2].set_xlabel("log(amount)")
axes[0,2].legend()

# Hourly fraud heatmap
hour_fraud = df.groupby("hour")["isFraud"].mean() * 100
axes[1,0].plot(hour_fraud.index, hour_fraud.values, color="#9b59b6", linewidth=2)
axes[1,0].fill_between(hour_fraud.index, hour_fraud.values, alpha=0.3, color="#9b59b6")
axes[1,0].set_title("Fraud Rate by Hour (%)")
axes[1,0].set_xlabel("Hour")
axes[1,0].set_ylabel("Fraud %")

# Balance mismatch
axes[1,1].hist(np.log1p(df[df["isFraud"]==1]["balance_mismatch"].clip(0)), bins=50,
               color="#e74c3c", alpha=0.8)
axes[1,1].set_title("Balance Mismatch (Fraud txns)")
axes[1,1].set_xlabel("log(balance_mismatch)")

# Correlation heatmap
corr_cols = ["amount", "balance_diff_orig", "balance_mismatch",
             "orig_zero_after", "dest_zero_before", "isFraud"]
corr = df[corr_cols].corr()
sns.heatmap(corr, ax=axes[1,2], annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 8})
axes[1,2].set_title("Feature Correlation")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "05_EDA_complete.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: reports/05_EDA_complete.png")

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Train ML Models
# ══════════════════════════════════════════════════════════════════
print("\n[6/11] Training ML models...")
results = {}

# Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
lr.fit(X_train_sc, y_train_bal)
lr_pred  = lr.predict(X_test_sc)
lr_prob  = lr.predict_proba(X_test_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_prob)
results["Logistic Regression"] = {"auc": lr_auc, "prob": lr_prob, "pred": lr_pred}
print(f"    AUC: {lr_auc:.4f}")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
rf_pred  = rf.predict(X_test)
rf_prob  = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_prob)
results["Random Forest"] = {"auc": rf_auc, "prob": rf_prob, "pred": rf_pred}
print(f"    AUC: {rf_auc:.4f}")

# XGBoost
print("  Training XGBoost...")
scale_pos = int((y_train_bal == 0).sum() / (y_train_bal == 1).sum())
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos, use_label_encoder=False,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
xgb_model.fit(X_train_bal, y_train_bal)
xgb_pred  = xgb_model.predict(X_test)
xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_prob)
results["XGBoost"] = {"auc": xgb_auc, "prob": xgb_prob, "pred": xgb_pred}
print(f"    AUC: {xgb_auc:.4f}")

# Save models
joblib.dump(lr,        MODELS_DIR / "logistic_regression.pkl")
joblib.dump(scaler,    MODELS_DIR / "scaler.pkl")
joblib.dump(rf,        MODELS_DIR / "random_forest.pkl")
joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
print("  Models saved to models/")

# ML Performance charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ML Model Performance", fontsize=14, fontweight="bold")

colors = ["#3498db", "#2ecc71", "#e74c3c"]
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=colors[i])
axes[0].plot([0,1],[0,1],"k--", alpha=0.4)
axes[0].set_title("ROC Curves")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend(fontsize=9)

# Confusion matrix for best model (XGBoost)
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
axes[1].set_title("XGBoost Confusion Matrix")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).nlargest(12)
axes[2].barh(feat_imp.index, feat_imp.values, color="#9b59b6")
axes[2].set_title("Top 12 Feature Importances (RF)")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "06_ML_performance.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: reports/06_ML_performance.png")

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Anomaly Detection (Isolation Forest)
# ══════════════════════════════════════════════════════════════════
print("\n[7/11] Running Isolation Forest anomaly detection...")
iso = IsolationForest(contamination=0.002, random_state=42, n_jobs=-1)
iso.fit(X_train_bal)
anomaly_scores = iso.decision_function(X_test)
anomaly_pred   = iso.predict(X_test)
anomaly_bin    = (anomaly_pred == -1).astype(int)
joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(anomaly_scores[y_test == 0], bins=60, alpha=0.6,
             color="#2ecc71", label="Legit")
axes[0].hist(anomaly_scores[y_test == 1], bins=60, alpha=0.6,
             color="#e74c3c", label="Fraud")
axes[0].set_title("Isolation Forest Anomaly Scores")
axes[0].set_xlabel("Score (lower = more anomalous)")
axes[0].legend()

cm_iso = confusion_matrix(y_test, anomaly_bin)
sns.heatmap(cm_iso, annot=True, fmt="d", cmap="Reds", ax=axes[1],
            xticklabels=["Normal","Anomaly"], yticklabels=["Legit","Fraud"])
axes[1].set_title("Isolation Forest Detection")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "07_anomaly_detection.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: reports/07_anomaly_detection.png")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Risk Scoring on Full Dataset (sample)
# ══════════════════════════════════════════════════════════════════
print("\n[8/11] Scoring transactions with risk engine...")

score_df = df_sample.copy()
X_score  = score_df[FEATURES].fillna(0)
X_score_sc = scaler.transform(X_score)

lr_p  = lr.predict_proba(X_score_sc)[:, 1]
rf_p  = rf.predict_proba(X_score)[:, 1]
xgb_p = xgb_model.predict_proba(X_score)[:, 1]

score_df["risk_score"] = (0.2 * lr_p + 0.3 * rf_p + 0.5 * xgb_p).round(4)
score_df["risk_label"] = pd.cut(
    score_df["risk_score"],
    bins=[-0.001, 0.35, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)

risk_counts = score_df["risk_label"].value_counts()
print(f"  Low: {risk_counts.get('Low',0):,}  |  "
      f"Medium: {risk_counts.get('Medium',0):,}  |  "
      f"High: {risk_counts.get('High',0):,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Alerts CSV
# ══════════════════════════════════════════════════════════════════
print("\n[9/11] Generating alerts...")
alerts = score_df[score_df["risk_label"] == "High"][[
    "nameOrig", "nameDest", "type", "amount", "risk_score", "isFraud"
]].copy()
alerts.to_csv(REPORTS_DIR / "alerts.csv", index=False)
print(f"  High-risk alerts: {len(alerts):,} → saved to reports/alerts.csv")

# ══════════════════════════════════════════════════════════════════
# SECTION 10 — Project Summary JSON
# ══════════════════════════════════════════════════════════════════
print("\n[10/11] Writing project summary...")
summary = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": "PaySim1",
    "total_transactions": len(df),
    "sample_used": len(df_sample),
    "total_fraud": int(df["isFraud"].sum()),
    "fraud_rate_pct": round(df["isFraud"].mean() * 100, 3),
    "features_engineered": len(FEATURES),
    "models": {
        "logistic_regression_auc": round(lr_auc, 4),
        "random_forest_auc": round(rf_auc, 4),
        "xgboost_auc": round(xgb_auc, 4)
    },
    "risk_distribution": {
        "Low": int(risk_counts.get("Low", 0)),
        "Medium": int(risk_counts.get("Medium", 0)),
        "High": int(risk_counts.get("High", 0))
    },
    "high_risk_alerts": len(alerts)
}
with open(REPORTS_DIR / "project_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved: reports/project_summary.json")

# ══════════════════════════════════════════════════════════════════
# SECTION 11 — Live HTML Dashboard
# ══════════════════════════════════════════════════════════════════
print("\n[11/11] Building live HTML dashboard...")

risk_dist_js = {k: int(v) for k, v in risk_counts.items()}
type_fraud   = df.groupby("type")["isFraud"].sum().to_dict()
type_counts_d = df["type"].value_counts().to_dict()
model_aucs   = {k: round(v["auc"], 4) for k, v in results.items()}

top_alerts = alerts.nlargest(20, "risk_score")[
    ["nameOrig", "nameDest", "type", "amount", "risk_score", "isFraud"]
].to_dict(orient="records")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>UPI Risk Monitor — Live Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; }}
    header {{ background: linear-gradient(135deg,#1a1f2e,#2d3561);
              padding: 20px 30px; border-bottom: 2px solid #3d5a99; }}
    header h1 {{ font-size: 1.6rem; color: #7eb8f7; }}
    header p  {{ font-size: 0.85rem; color: #8899aa; margin-top: 4px; }}
    .kpi-row  {{ display: flex; gap: 16px; padding: 20px 30px; flex-wrap: wrap; }}
    .kpi      {{ flex: 1; min-width: 160px; background: #1e2235;
                 border-radius: 10px; padding: 18px 20px;
                 border-left: 4px solid #3d5a99; }}
    .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: #7eb8f7; }}
    .kpi .lbl {{ font-size: 0.78rem; color: #8899aa; margin-top: 4px; }}
    .charts   {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(380px,1fr));
                 gap: 20px; padding: 0 30px 20px; }}
    .card     {{ background: #1e2235; border-radius: 10px; padding: 20px; }}
    .card h3  {{ font-size: 0.95rem; color: #a0b4cc; margin-bottom: 14px; }}
    canvas    {{ max-height: 260px; }}
    .table-wrap {{ padding: 0 30px 30px; }}
    table     {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th        {{ background: #2d3561; color: #7eb8f7; padding: 10px 12px;
                 text-align: left; }}
    td        {{ padding: 9px 12px; border-bottom: 1px solid #2a2f45; }}
    tr:hover td {{ background: #252b42; }}
    .badge    {{ padding: 3px 9px; border-radius: 12px; font-size: 0.75rem;
                 font-weight: 600; }}
    .High   {{ background:#4d1515; color:#ff6b6b; }}
    .fraud1 {{ background:#4d1515; color:#ff6b6b; }}
    .fraud0 {{ background:#0d3320; color:#6bffb8; }}
    footer  {{ text-align:center; padding:16px; color:#556; font-size:0.78rem; }}
  </style>
</head>
<body>
<header>
  <h1>UPI Transaction Analytics &amp; Risk Monitoring Platform</h1>
  <p>PaySim1 Dataset &nbsp;|&nbsp; Generated: {summary['generated_at']} &nbsp;|&nbsp;
     Auto-refreshes every 60s</p>
</header>

<div class="kpi-row">
  <div class="kpi"><div class="val">{summary['total_transactions']:,}</div><div class="lbl">Total Transactions</div></div>
  <div class="kpi"><div class="val">{summary['total_fraud']:,}</div><div class="lbl">Fraud Detected</div></div>
  <div class="kpi"><div class="val">{summary['fraud_rate_pct']}%</div><div class="lbl">Fraud Rate</div></div>
  <div class="kpi"><div class="val">{risk_dist_js.get('High',0):,}</div><div class="lbl">High Risk Alerts</div></div>
  <div class="kpi"><div class="val">{round(xgb_auc,3)}</div><div class="lbl">Best Model AUC (XGBoost)</div></div>
</div>

<div class="charts">
  <div class="card"><h3>Risk Distribution</h3><canvas id="riskPie"></canvas></div>
  <div class="card"><h3>Fraud by Transaction Type</h3><canvas id="fraudBar"></canvas></div>
  <div class="card"><h3>Transaction Volume by Type</h3><canvas id="typeBar"></canvas></div>
  <div class="card"><h3>Model AUC Comparison</h3><canvas id="aucBar"></canvas></div>
</div>

<div class="table-wrap">
  <div class="card">
    <h3>Top 20 High-Risk Alerts</h3>
    <table>
      <thead>
        <tr><th>Sender</th><th>Receiver</th><th>Type</th>
            <th>Amount</th><th>Risk Score</th><th>Fraud</th></tr>
      </thead>
      <tbody>
        {"".join(f'''<tr>
          <td>{r["nameOrig"]}</td><td>{r["nameDest"]}</td>
          <td>{r["type"]}</td><td>₹{r["amount"]:,.2f}</td>
          <td><span class="badge High">{r["risk_score"]:.4f}</span></td>
          <td><span class="badge {'fraud1' if r['isFraud'] else 'fraud0'}">
            {'FRAUD' if r['isFraud'] else 'Clean'}</span></td>
        </tr>''' for r in top_alerts)}
      </tbody>
    </table>
  </div>
</div>

<footer>UPI Risk Monitor &mdash; OJT Project &mdash; {datetime.now().year}</footer>

<script>
const riskData = {json.dumps(risk_dist_js)};
const fraudByType = {json.dumps({k: int(v) for k,v in type_fraud.items()})};
const typeCounts  = {json.dumps({k: int(v) for k,v in type_counts_d.items()})};
const modelAucs   = {json.dumps(model_aucs)};

new Chart(document.getElementById("riskPie"), {{
  type: "doughnut",
  data: {{
    labels: Object.keys(riskData),
    datasets: [{{ data: Object.values(riskData),
      backgroundColor: ["#2ecc71","#f39c12","#e74c3c"], borderWidth: 2 }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color:"#ccc" }} }} }} }}
}});

new Chart(document.getElementById("fraudBar"), {{
  type: "bar",
  data: {{
    labels: Object.keys(fraudByType),
    datasets: [{{ label:"Fraud Count", data: Object.values(fraudByType),
      backgroundColor:"#e74c3c", borderRadius: 4 }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color:"#ccc" }} }} }},
    scales: {{ x: {{ ticks: {{ color:"#ccc" }} }}, y: {{ ticks: {{ color:"#ccc" }} }} }} }}
}});

new Chart(document.getElementById("typeBar"), {{
  type: "bar",
  data: {{
    labels: Object.keys(typeCounts),
    datasets: [{{ label:"Transactions", data: Object.values(typeCounts),
      backgroundColor:"#3498db", borderRadius: 4 }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color:"#ccc" }} }} }},
    scales: {{ x: {{ ticks: {{ color:"#ccc" }} }}, y: {{ ticks: {{ color:"#ccc" }} }} }} }}
}});

new Chart(document.getElementById("aucBar"), {{
  type: "bar",
  data: {{
    labels: Object.keys(modelAucs),
    datasets: [{{ label:"AUC-ROC", data: Object.values(modelAucs),
      backgroundColor:["#3498db","#2ecc71","#e74c3c"], borderRadius: 4 }}]
  }},
  options: {{
    plugins: {{ legend: {{ labels: {{ color:"#ccc" }} }} }},
    scales: {{
      x: {{ ticks: {{ color:"#ccc" }} }},
      y: {{ min: 0.5, max: 1.0, ticks: {{ color:"#ccc" }} }}
    }}
  }}
}});

setTimeout(() => location.reload(), 60000);
</script>
</body>
</html>"""

with open(REPORTS_DIR / "live_dashboard.html", "w") as f:
    f.write(html)
print("  Saved: reports/live_dashboard.html")

print("\n" + "=" * 65)
print(f"  PROJECT COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)
print(f"\n  Reports generated:")
print(f"    reports/05_EDA_complete.png")
print(f"    reports/06_ML_performance.png")
print(f"    reports/07_anomaly_detection.png")
print(f"    reports/alerts.csv          ({len(alerts):,} rows)")
print(f"    reports/project_summary.json")
print(f"    reports/live_dashboard.html")
print(f"\n  Open dashboard:")
print(f"    open reports/live_dashboard.html")
