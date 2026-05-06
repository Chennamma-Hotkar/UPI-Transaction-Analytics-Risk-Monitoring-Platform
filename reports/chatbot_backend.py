"""
UPI Risk Copilot — FastAPI backend
Capabilities:
  1. Explain risk score  →  Why was this flagged?
  2. NL-to-SQL          →  Show me all fraud from Delhi today
  3. Auto insights      →  What's my fraud trend this week?
  4. Ingest RBI PDFs    →  POST /ingest/pdf
  5. Ingest CSV data    →  POST /ingest/transactions

Run:  uvicorn chatbot_backend:app --reload --port 8000
"""

import io
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import faiss
import numpy as np
import pandas as pd
import pdfplumber
from groq import Groq
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
FAST_MODEL   = "llama-3.1-8b-instant"   # Groq free — 30 RPM, 14400 req/day

HERE     = Path(__file__).parent
DB_PATH  = str(HERE / "upi_transactions.db")
RISK_CSV = HERE / "risk_scored_transactions.csv"
FEAT_CSV = HERE.parent / "data" / "processed" / "featured_data.csv"

_groq = Groq(api_key=GROQ_API_KEY)

def _llm(prompt: str, system: str = "", max_tokens: int = 600) -> str:
    """Call Groq (Llama 3.1 8B) — free, 30 RPM, no billing needed."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _groq.chat.completions.create(
        model=FAST_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="UPI Risk Copilot API", version="2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Globals (RAG) ────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.IndexFlatIP] = None
_rag_chunks: list[str] = []

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad",
]
_CITY_WEIGHTS = [
    0.12, 0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05,
    0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02,
    0.02, 0.02, 0.01, 0.01,
]

# RBI regulatory knowledge base (seeded; extended via /ingest/pdf)
_RBI_KB = [
    "RBI Circular DPSS.CO.PD.No.1030/02.12.004/2020-21: UPI transactions exceeding ₹2 lakhs require additional factor authentication. High-value TRANSFER and CASH_OUT transactions above this limit must be flagged for enhanced due diligence.",
    "RBI Master Direction on KYC (2016): Banks must implement transaction monitoring systems to detect unusual patterns. Accounts showing sudden spikes in transaction volume (>3x baseline), especially involving multiple counterparties, indicate potential money mule activity.",
    "NPCI UPI Fraud Risk Management: Balance drain ratio >0.9 (account emptied in single transaction) is a Tier-1 fraud signal. Systems must auto-block and flag for manual review when origin account balance drops to near-zero post transaction.",
    "RBI Payments Vision 2025: Off-hours transactions (11PM–5AM) carry 2.4x higher fraud risk. Round-number amounts (₹10,000; ₹50,000) in CASH_OUT category suggest structured layering — a money laundering technique to avoid detection thresholds.",
    "FATF Recommendation 16: Wire transfer rules apply to UPI. Destination accounts receiving funds from 5+ unique senders within 24 hours are classified as potential aggregator accounts used in smurfing operations.",
    "RBI Guidelines on Fraud Monitoring (2023): Balance mismatch errors (where old_balance − amount ≠ new_balance) indicate potential transaction manipulation or system compromise. Both-side balance errors simultaneously suggest coordinated fraud.",
    "NPCI Risk Framework: Isolation Forest anomaly score combined with ML fraud probability creates a composite risk score. Transactions scoring >70/100 require immediate analyst review. The 0–50 ML component captures learned fraud patterns; the 0–30 rule component enforces regulatory policies.",
    "RBI Mule Account Detection: Suspected mule accounts show (1) high ratio of balance-unchanged transactions, (2) multiple unique senders in short window, (3) consistent balance errors, (4) high total volume with low individual amounts (structuring).",
    "UPI Circular 2024: Transaction velocity limits — maximum 20 transactions per account per day. Accounts exceeding velocity limits trigger automatic holds. Risk scoring adds 15 points for accounts with z-score >2.",
    "RBI Cyber Security Framework: Real-time fraud scoring latency must be <100 ms for inline transaction blocking. High-risk (score >70) transactions are auto-flagged; analysts must review within 4 hours per regulatory requirement.",
]

# ── Schema exposed to the LLM for NL-to-SQL ─────────────────────────────────
_SCHEMA = """
Table: transactions
Columns:
  transaction_id TEXT    -- unique identifier like TXN0000001
  type           TEXT    -- TRANSFER or CASH_OUT
  amount         REAL    -- transaction amount in INR
  name_orig      TEXT    -- sender account ID
  name_dest      TEXT    -- receiver account ID
  old_balance_orig  REAL
  new_balance_orig  REAL
  old_balance_dest  REAL
  new_balance_dest  REAL
  is_fraud       INTEGER -- 0=legitimate, 1=confirmed fraud
  risk_score     REAL    -- composite 0-100 risk score
  risk_level     TEXT    -- LOW / MEDIUM / HIGH
  ml_fraud_prob  REAL    -- 0-1 ML model probability
  ml_contribution    REAL -- ML component of score (0-50)
  anomaly_contribution REAL -- anomaly component (0-20)
  rule_contribution  REAL -- rule-based component (0-30)
  is_flagged     INTEGER -- 1 = flagged for analyst review
  risk_factors   TEXT    -- JSON array of triggered rule names
  city           TEXT    -- Indian city (e.g. Delhi, Mumbai)
  hour_of_day    INTEGER -- 0-23
  day_of_week    TEXT    -- Monday..Sunday
  date_str       TEXT    -- YYYY-MM-DD (range: 2026-04-29 to 2026-05-06)
"""


# ── Database setup ───────────────────────────────────────────────────────────
def _build_database() -> bool:
    """Merge CSVs, enrich with synthetic city/date, load into SQLite."""
    if not RISK_CSV.exists():
        return False

    risk_df = pd.read_csv(RISK_CSV)
    n = len(risk_df)
    rng = np.random.default_rng(42)

    # Try to pull real transaction fields from featured_data.csv
    if FEAT_CSV.exists():
        feat_df = pd.read_csv(FEAT_CSV)
        # Align lengths
        feat_df = feat_df.iloc[:n].reset_index(drop=True)
        for col in ["type", "amount", "nameOrig", "nameDest",
                    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "step"]:
            if col in feat_df.columns:
                risk_df[col] = feat_df[col].values
        rename = {
            "nameOrig": "name_orig", "nameDest": "name_dest",
            "oldbalanceOrg": "old_balance_orig", "newbalanceOrig": "new_balance_orig",
            "oldbalanceDest": "old_balance_dest", "newbalanceDest": "new_balance_dest",
        }
        risk_df.rename(columns=rename, inplace=True)
    else:
        # Synthesise if featured_data not available
        risk_df["type"]             = rng.choice(["TRANSFER", "CASH_OUT"], n, p=[0.55, 0.45])
        risk_df["amount"]           = rng.exponential(15_000, n).clip(100, 500_000).round(2)
        risk_df["name_orig"]        = [f"C{rng.integers(int(1e9), int(9e9))}" for _ in range(n)]
        risk_df["name_dest"]        = [f"M{rng.integers(int(1e9), int(9e9))}" for _ in range(n)]
        risk_df["old_balance_orig"] = rng.uniform(0, 200_000, n).round(2)
        risk_df["new_balance_orig"] = (risk_df["old_balance_orig"] - risk_df["amount"]).clip(0).round(2)
        risk_df["old_balance_dest"] = rng.uniform(0, 100_000, n).round(2)
        risk_df["new_balance_dest"] = (risk_df["old_balance_dest"] + risk_df["amount"]).round(2)
        risk_df["step"]             = rng.integers(1, 720, n)

    # Synthetic enrichment
    risk_df["transaction_id"] = [f"TXN{i:07d}" for i in range(n)]
    risk_df["city"]           = rng.choice(INDIAN_CITIES, n, p=_CITY_WEIGHTS)
    risk_df["hour_of_day"]    = (risk_df.get("step", rng.integers(0, 23, n)) % 24).astype(int)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    steps = risk_df.get("step", rng.integers(0, 720, n)).astype(int)
    risk_df["day_of_week"] = [days[(int(s) // 24) % 7] for s in steps]

    # Dates clustered in the last 7 days so "today" and "this week" queries hit
    base = pd.Timestamp("2026-04-29")
    risk_df["date_str"] = [
        (base + pd.Timedelta(days=int(rng.integers(0, 8)))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]

    # Rename standard columns
    risk_df.rename(columns={"isFraud": "is_fraud"}, inplace=True)
    if "is_fraud" not in risk_df.columns:
        risk_df["is_fraud"] = 0

    keep = [
        "transaction_id", "type", "amount", "name_orig", "name_dest",
        "old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest",
        "is_fraud", "risk_score", "risk_level", "ml_fraud_prob",
        "ml_contribution", "anomaly_contribution", "rule_contribution",
        "is_flagged", "risk_factors", "city", "hour_of_day", "day_of_week", "date_str",
    ]
    risk_df = risk_df[[c for c in keep if c in risk_df.columns]]

    con = sqlite3.connect(DB_PATH)
    risk_df.to_sql("transactions", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_city ON transactions(city)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_date ON transactions(date_str)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_risk ON transactions(risk_level)")
    con.commit()
    con.close()
    return True


# ── RAG setup ────────────────────────────────────────────────────────────────
def _build_rag() -> None:
    global _embedder, _faiss_index, _rag_chunks
    print("Loading sentence-transformer …")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    _rag_chunks = list(_RBI_KB)
    embs = _embedder.encode(_rag_chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    _faiss_index = faiss.IndexFlatIP(dim)  # inner-product = cosine on L2-normalised vecs
    _faiss_index.add(embs.astype("float32"))
    print(f"RAG index ready — {len(_rag_chunks)} chunks")


def _retrieve(query: str, k: int = 3) -> str:
    if _faiss_index is None or _embedder is None:
        return ""
    q = _embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _, idxs = _faiss_index.search(q, k)
    return "\n\n".join(_rag_chunks[i] for i in idxs[0] if i < len(_rag_chunks))


# ── Intent detection ─────────────────────────────────────────────────────────
_INTENT_PROMPT = (
    "Classify this UPI fraud analyst query into exactly ONE category:\n"
    "  explain_risk — user asks why a transaction was flagged, what the risk score means\n"
    "  nl_to_sql    — user wants to find/filter/count specific transactions (mentions city, date, amount)\n"
    "  insights     — user asks for trends, patterns, summaries, weekly/daily analytics\n"
    "  general      — general question about UPI fraud methodology, RBI rules, risk scoring\n\n"
    "Reply with ONLY the category name.\n\nMessage: {msg}"
)


def _detect_intent(message: str) -> str:
    """Keyword-based intent — no API call, zero quota usage."""
    lo = message.lower()
    if any(w in lo for w in ["why","flag","explain","score","reason","risk","triggered","block","rule"]):
        return "explain_risk"
    if any(w in lo for w in ["show","find","list","from","today","all","query","delhi","mumbai",
                              "bangalore","chennai","hyderabad","city","amount","count","how many"]):
        return "nl_to_sql"
    if any(w in lo for w in ["trend","week","pattern","insight","summary","analytics",
                              "daily","hourly","overall","report","overview"]):
        return "insights"
    return "general"


# ── Handler: explain risk score ───────────────────────────────────────────────
def _explain_risk(transaction_id: Optional[str], message: str) -> dict:
    tx: dict = {}
    if transaction_id:
        con = sqlite3.connect(DB_PATH)
        row = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_id = ? LIMIT 1",
            con, params=[transaction_id],
        )
        con.close()
        if not row.empty:
            tx = row.iloc[0].to_dict()

    if not tx:
        # Fall back to the top-scoring flagged transaction
        con = sqlite3.connect(DB_PATH)
        row = pd.read_sql(
            "SELECT * FROM transactions WHERE is_flagged=1 ORDER BY risk_score DESC LIMIT 1",
            con,
        )
        con.close()
        tx = row.iloc[0].to_dict() if not row.empty else {}

    context = _retrieve(f"fraud risk factors {message}")
    risk_factors = tx.get("risk_factors", "[]")
    if isinstance(risk_factors, str):
        try:
            risk_factors = json.loads(risk_factors)
        except Exception:
            risk_factors = [risk_factors]

    safe_tx = {k: v for k, v in tx.items() if k not in ("name_orig", "name_dest")}

    system = (
        "You are a UPI fraud analyst AI. Explain risk scores in plain English to bank investigators.\n"
        f"Use the RBI regulatory context below to justify triggered rules.\n\n"
        f"RBI CONTEXT:\n{context}\n\n"
        f"Transaction:\n{json.dumps(safe_tx, indent=2, default=str)}"
    )
    prompt = (
        f'User asked: "{message}"\n\n'
        f"Transaction summary:\n"
        f"- Risk Score: {tx.get('risk_score', 'N/A')}/100  →  {tx.get('risk_level', 'N/A')} RISK\n"
        f"- ML Fraud Probability: {float(tx.get('ml_fraud_prob', 0)):.1%}\n"
        f"- Triggered Rules: {risk_factors}\n"
        f"- Score Breakdown: ML({tx.get('ml_contribution', 0)}) + "
        f"Anomaly({tx.get('anomaly_contribution', 0)}) + Rules({tx.get('rule_contribution', 0)})\n\n"
        "Explain in plain English why this transaction was flagged.\n"
        "Use **bullet points**. Reference relevant RBI guidelines where applicable.\n"
        "End with a recommended action: **Block** / **Review** / **Allow**."
    )
    return {
        "answer": _llm(prompt, system=system, max_tokens=600),
        "transaction": tx.get("transaction_id"),
        "type": "explain_risk",
    }


# ── Handler: NL-to-SQL ───────────────────────────────────────────────────────
_SQL_SYSTEM = (
    "You are a SQLite expert for a UPI fraud analytics database.\n"
    "Convert natural language to valid SQLite SQL ONLY — no explanation, no markdown fences.\n\n"
    + _SCHEMA
    + "\nRules:\n"
    "- 'today' = '2026-05-06'   |   'this week' means date_str >= '2026-04-29'\n"
    "- 'fraud' means is_fraud=1 OR risk_level='HIGH' OR is_flagged=1\n"
    "- City names are case-sensitive stored values — use exact city name\n"
    "- Always add LIMIT 50 unless user specifies otherwise\n"
    "- Never use subqueries when a simple JOIN or WHERE suffices\n"
)


def _nl_to_sql(message: str) -> dict:
    sql = _llm(message, system=_SQL_SYSTEM, max_tokens=300).strip().strip("```sql").strip("```").strip()

    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql(sql, con)
        con.close()
        rows = df.to_dict(orient="records")
        count = len(rows)
        sample = rows[:10]
    except Exception as exc:
        return {
            "answer": f"I couldn't run that query: `{exc}`\n\nGenerated SQL:\n```sql\n{sql}\n```",
            "sql": sql,
            "type": "nl_to_sql",
            "rows": [],
            "total_count": 0,
        }

    context = _retrieve(message, k=2)
    sys = f"You are a UPI fraud analyst. Summarise query results concisely.\n\nRBI Context:\n{context}"
    prompt = (
        f"Query: {message}\nSQL: {sql}\n"
        f"Result ({count} rows, showing first {len(sample)}):\n"
        f"{json.dumps(sample, default=str)}\n\n"
        "Provide a **2–3 sentence summary** of the findings. "
        "Highlight any concerning patterns or anomalies."
    )
    return {
        "answer": _llm(prompt, system=sys, max_tokens=400),
        "sql": sql,
        "rows": rows[:20],
        "total_count": count,
        "type": "nl_to_sql",
    }


# ── Handler: auto insights ───────────────────────────────────────────────────
def _insights(message: str) -> dict:
    con = sqlite3.connect(DB_PATH)
    agg: dict = {}

    _queries = {
        "daily_fraud": (
            "SELECT date_str, COUNT(*) total, SUM(is_fraud) fraud_count, "
            "ROUND(AVG(risk_score),1) avg_risk "
            "FROM transactions GROUP BY date_str ORDER BY date_str DESC LIMIT 14"
        ),
        "city_breakdown": (
            "SELECT city, COUNT(*) total, SUM(is_fraud) fraud_count, "
            "ROUND(AVG(risk_score),1) avg_risk "
            "FROM transactions GROUP BY city ORDER BY fraud_count DESC LIMIT 10"
        ),
        "risk_distribution": (
            "SELECT risk_level, COUNT(*) cnt, ROUND(AVG(amount),0) avg_amount "
            "FROM transactions GROUP BY risk_level"
        ),
        "hourly_pattern": (
            "SELECT hour_of_day, COUNT(*) total, SUM(is_fraud) fraud_count "
            "FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day"
        ),
    }
    for name, q in _queries.items():
        try:
            agg[name] = pd.read_sql(q, con).to_dict(orient="records")
        except Exception:
            agg[name] = []
    con.close()

    context = _retrieve(message, k=3)
    sys = ("You are a UPI fraud analytics expert generating executive-level insights.\n\n"
           f"RBI Context:\n{context}")
    prompt = (
        f'User asked: "{message}"\n\n'
        f"Platform analytics:\n"
        f"Daily trend (last 14 days):\n{json.dumps(agg['daily_fraud'], indent=2)}\n\n"
        f"City breakdown (top 10):\n{json.dumps(agg['city_breakdown'], indent=2)}\n\n"
        f"Risk distribution:\n{json.dumps(agg['risk_distribution'], indent=2)}\n\n"
        f"Hourly pattern:\n{json.dumps(agg['hourly_pattern'], indent=2)}\n\n"
        "Generate actionable insights with:\n"
        "1. **Key trend** (2–3 sentences)\n"
        "2. **Bullet points** (3–4 notable patterns)\n"
        "3. **Recommended action** for the fraud team\n"
        "Format with markdown."
    )
    return {"answer": _llm(prompt, system=sys, max_tokens=700), "data": agg, "type": "insights"}


# ── Handler: general RAG ─────────────────────────────────────────────────────
def _general_rag(message: str) -> dict:
    context = _retrieve(message, k=4)
    sys = ("You are a UPI fraud risk expert. Answer based on RBI regulations and fraud-detection best practices.\n\n"
           f"Knowledge Base:\n{context}")
    return {"answer": _llm(message, system=sys, max_tokens=500), "type": "general"}


# ── Pydantic schemas ─────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    transaction_id: Optional[str] = None
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    type: str
    sql: Optional[str] = None
    rows: Optional[list] = None
    total_count: Optional[int] = None
    transaction: Optional[str] = None
    data: Optional[dict] = None


# ── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    ok = _build_database()
    print(f"Database {'ready' if ok else 'SKIPPED (CSV not found)'}")
    _build_rag()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "db": Path(DB_PATH).exists(),
        "rag_chunks": len(_rag_chunks),
        "fast_model": FAST_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        intent = _detect_intent(req.message)
        if intent == "explain_risk":
            result = _explain_risk(req.transaction_id, req.message)
        elif intent == "nl_to_sql":
            result = _nl_to_sql(req.message)
        elif intent == "insights":
            result = _insights(req.message)
        else:
            result = _general_rag(req.message)
        return ChatResponse(**result)
    except Exception as exc:
        return ChatResponse(answer=f"⚠️ Error: {str(exc)[:300]}", type="error")


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest an RBI circular / policy PDF into the RAG vector store."""
    global _rag_chunks, _faiss_index
    if _embedder is None or _faiss_index is None:
        raise HTTPException(503, "RAG engine not ready")
    content = await file.read()
    chunks: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            for i in range(0, len(text), 450):
                chunk = text[i: i + 500].strip()
                if len(chunk) > 80:
                    chunks.append(chunk)
    if not chunks:
        return {"message": "No text extracted", "filename": file.filename}
    embs = _embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    _faiss_index.add(embs)
    _rag_chunks.extend(chunks)
    return {"message": f"Ingested {len(chunks)} chunks", "filename": file.filename, "total_chunks": len(_rag_chunks)}


@app.post("/ingest/transactions")
async def ingest_transactions(file: UploadFile = File(...)):
    """Append a new CSV of transactions to the SQLite database."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    con = sqlite3.connect(DB_PATH)
    df.to_sql("transactions", con, if_exists="append", index=False)
    con.close()
    return {"message": f"Appended {len(df)} rows", "filename": file.filename}
