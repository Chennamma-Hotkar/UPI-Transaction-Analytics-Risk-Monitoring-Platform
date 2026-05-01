# UPI Transaction Analytics & Risk Monitoring Platform

A machine learning-powered platform to detect fraudulent UPI transactions, score transaction risk in real time, and provide actionable insights through an analytics dashboard.



## Overview

With UPI payments growing at an unprecedented scale in India, traditional manual fraud detection simply cannot keep up. This project tackles that gap by building an automated pipeline that ingests transaction data, runs it through ML models, assigns risk scores, and surfaces suspicious activity on a live monitoring dashboard.



## Dataset

The project uses a UPI transaction dataset (synthetic/sample) with the following key fields:

| Field | Description |
|-------|-------------|
| `transaction_id` | Unique transaction identifier |
| `amount` | Transaction amount (INR) |
| `timestamp` | Date and time |
| `user_id` | Sender account |
| `merchant_id` | Receiver / payee |
| `device_info` | Device used |
| `location` | Geographic location |

### EDA Findings

- Identified outliers in transaction amount distribution
- Detected peak transaction hours and behavioral differences across weekdays vs weekends
- Flagged users with unusually high transaction frequency
- Found inconsistencies in device and location patterns across sessions
- Key features shortlisted: transaction frequency, time gap between transactions, amount deviation from user average, device/location consistency



## Features

- **Transaction Analytics** — Analyze spending patterns, frequency, time-of-day behavior, and merchant activity
- **Risk Scoring Engine** — Classifies every transaction as Low, Medium, or High risk using a weighted multi-factor model
- **Fraud Detection** — Identifies patterns like high-frequency micro-transactions, sudden amount spikes, and unusual device/location combos
- **Mule Account Detection** — Flags accounts that show layering behavior indicative of money laundering
- **Anomaly Detection** — Unsupervised detection of outlier transactions using Isolation Forest
- **Real-Time Alerts** — Triggers alerts when a transaction breaches configurable risk thresholds
- **Monitoring Dashboard** — Visual overview of transaction volume, risk distribution, and flagged events
- **Exportable Reports** — Download analysis results as CSV or PDF



## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask / FastAPI |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Logistic Regression, Random Forest, Isolation Forest) |
| Database | PostgreSQL / MySQL |
| Caching (optional) | Redis |
| Visualization | Matplotlib, Seaborn, Power BI / Tableau |
| Frontend (optional) | HTML, CSS, JavaScript / React |
| Version Control | Git & GitHub |



## Project Structure

```
upi-risk-monitor/
│
├── data/                        # Raw and processed datasets
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── diagrams/                # Architecture and flow diagrams
│       ├── hld.png              # High-Level Design
│       ├── lld.png              # Low-Level Design
│       ├── dataflow.png         # Data Flow Diagram
│       └── consumerflow.png     # Consumer Flow Diagram
│
├── notebooks/                   # Exploratory analysis and model experiments
│
├── src/
│   ├── preprocessing/           # Data cleaning and feature engineering
│   ├── models/                  # ML model training and evaluation
│   ├── scoring/                 # Risk scoring logic
│   ├── api/                     # Flask / FastAPI backend
│   └── dashboard/               # Visualization and dashboard code
│
├── reports/                     # Generated CSV / PDF reports
├── requirements.txt
└── README.md
```



## Pipeline Flow

```
Raw Transaction Data (CSV / DB)
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
(frequency, time gaps, device patterns, location signals)
        ↓
ML Risk Evaluation
(Logistic Regression + Random Forest + Isolation Forest)
        ↓
Risk Score Assignment  →  Low / Medium / High
        ↓
Alert Trigger (if High risk)
        ↓
Store Results in DB
        ↓
Dashboard + Report Export
```



## Machine Learning Models

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline supervised fraud classifier |
| Random Forest | High-accuracy supervised fraud classifier |
| Isolation Forest | Unsupervised anomaly detection |

### Target Metrics

| Metric | Goal |
|---|---|
| Model Accuracy | ≥ 85% |
| False Positive Rate | < 10% |
| Risk Score Separation | Clear Low / Medium / High bands |
| Response Time | < 2 seconds per transaction |



## Dashboard

The monitoring dashboard provides:
- Live transaction feed with risk labels
- Risk distribution charts (pie / bar)
- User-wise and merchant-wise analytics
- Trend analysis over time
- Flagged transaction drill-down view



## License

This project was developed as part of an On-the-Job Training (OJT) program. All rights reserved by the authors.
