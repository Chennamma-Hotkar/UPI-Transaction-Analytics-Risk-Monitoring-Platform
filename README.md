# UPI Transaction Analytics & Risk Monitoring Platform

A FinTech data analytics and machine learning project focused on detecting fraudulent UPI transactions, scoring risk levels, and monitoring suspicious activity in real time.


## Table of Contents

- [About](#about)
- [Research & Problem Analysis](#research--problem-analysis)
- [Dataset](#dataset)
- [System Design](#system-design)
- [Tech Stack](#tech-stack)
- [Team](#team)


## About

UPI transactions in India have grown at an enormous scale, making manual fraud detection practically impossible. This platform aims to:

- Automatically detect suspicious UPI transactions using machine learning
- Assign a **Low / Medium / High** risk score to each transaction
- Identify mule accounts and money laundering patterns
- Provide a real-time monitoring dashboard for fraud analysts and compliance teams


## Research & Problem Analysis

Key fraud patterns studied and considered for detection:

- **Mule accounts** — accounts used to funnel stolen funds
- **High-frequency micro-transactions** — rapid small transfers to avoid detection
- **Sudden spending spikes** — abnormal deviation from a user's typical behavior
- **Device & location inconsistencies** — transactions from unusual devices or regions
- **Money laundering flows** — layered transactions across multiple accounts

A full **Product Requirements Document (PRD)** has been prepared covering problem scope, user stories, functional requirements, and success metrics.


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


## System Design

### HLD — High-Level Design

Defines the overall system architecture across five layers: data ingestion, preprocessing, ML pipeline, risk scoring, and dashboard/reporting.

### LLD — Low-Level Design

Covers component-level details including database schema, API contracts, model interfaces, and risk scoring logic.

### Dataflow Diagram

Maps how raw transaction data moves through the system — from source ingestion to final risk output and alert generation.

### Consumer Flow Diagram

Maps the end-user journey — how fraud analysts and compliance officers interact with the platform to review flagged transactions and generate reports.


## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.9+ |
| Data Analysis | Pandas, NumPy, SciPy |
| Machine Learning | Scikit-Learn (Logistic Regression, Random Forest, Isolation Forest) |
| Database | PostgreSQL / MySQL |
| Visualization | Matplotlib, Seaborn |
| API | Flask / FastAPI |
| Version Control | Git & GitHub |


## Team

| Name | Roll No |
|------|---------|
| Chennamma Hotkar | PST-25-0248 / 251810700203 |
| Kona Nagadevi | PST-25-0231 / 251810700321 |

**Mentor:** Navneet Nautiyal — Data Analytics, Data Science, Python and its libraries
