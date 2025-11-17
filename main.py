from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from analysis import analyze_statement

import numpy as np
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Optional Gemini client -----
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if genai is not None and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---- helper: recursively convert numpy / pandas outputs to plain Python ----
def to_native(obj):
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]

    # dicts
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}

    # lists / tuples / sets
    if isinstance(obj, (list, tuple, set)):
        return [to_native(x) for x in obj]

    # anything else we just pass through
    return obj


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Accept CSV/XLSX, run CashRaaga analysis, return rich JSON.
    """
    try:
        contents = await file.read()
        result = analyze_statement(contents, file.filename)

        # cleaned_csv is bytes -> convert to UTF-8 text for JSON
        csv_bytes = result.get("cleaned_csv", b"")
        if isinstance(csv_bytes, (bytes, bytearray)):
            result["cleaned_csv"] = csv_bytes.decode("utf-8", errors="ignore")
        elif isinstance(csv_bytes, str):
            result["cleaned_csv"] = csv_bytes
        else:
            result["cleaned_csv"] = ""

        # make everything JSON-safe
        safe_result = to_native(result)

        return JSONResponse(content=safe_result)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )


# ---------- CHAT SUPPORT ----------

class ChatRequest(BaseModel):
    message: str
    analysis: dict | None = None


def build_rule_based_reply(message: str, analysis: dict | None) -> str:
    """
    Rule-based explainer that:
    - understands category + month questions (fuel in March, hospital in January, etc.)
    - still answers risk / EMI / UPI / default questions
    It always returns numerically correct values using the cleaned CSV.
    """
    if not analysis:
        return (
            "I don't have your numbers yet. Upload a statement on the dashboard "
            "and then ask again."
        )

    summary = analysis.get("summary", {}) or {}
    future = analysis.get("future_block", {}) or {}
    upi = analysis.get("upi", {}) or {}
    emi = analysis.get("emi", {}) or {}
    cleaned_csv = analysis.get("cleaned_csv", "") or ""

    msg = (message or "").lower()

    # --- local imports (keeps main.py clean) ---
    from io import StringIO
    import pandas as pd

    # Try to rebuild the cleaned df from CSV text
    df = None
    try:
        df = pd.read_csv(StringIO(cleaned_csv))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["month_name"] = df["date"].dt.month_name().str.lower()
        else:
            df["month_name"] = ""
        if "category" in df.columns:
            df["category"] = df["category"].astype(str).str.lower()
        else:
            df["category"] = ""
    except Exception:
        df = None

    # -------------------------
    # 1. Detect month in the question
    # -------------------------
    month_map = {
        "january": "january", "jan": "january",
        "february": "february", "feb": "february",
        "march": "march", "mar": "march",
        "april": "april", "apr": "april",
        "may": "may",
        "june": "june", "jun": "june",
        "july": "july", "jul": "july",
        "august": "august", "aug": "august",
        "september": "september", "sep": "september", "sept": "september",
        "october": "october", "oct": "october",
        "november": "november", "nov": "november",
        "december": "december", "dec": "december",
    }
    detected_month = None
    for key, val in month_map.items():
        if key in msg:
            detected_month = val
            break

    # -------------------------
    # 2. Detect category in the question
    # (keys match categories created in analysis.py)
    # -------------------------
    category_map = {
        "medical": ["medical", "hospital", "pharmacy", "clinic", "doctor"],
        "food & dining": ["food", "eating", "restaurant", "swiggy", "zomato", "lunch", "dinner", "breakfast"],
        "fuel & transport": ["fuel", "petrol", "diesel", "transport", "cab", "uber", "ola", "bus", "train"],
        "shopping": ["shopping", "amazon", "flipkart", "myntra", "purchase"],
        "subscriptions": ["subscription", "subscriptions", "netflix", "hotstar", "prime", "ott"],
        "emi": ["emi", "loan"],
        "rent": ["rent"],
        "mobile & internet": ["mobile", "internet", "recharge", "phone bill", "wifi", "broadband"],
        "salary": ["salary", "payroll", "pay"],
        "others": ["others"],
    }
    detected_category = None
    for cat, words in category_map.items():
        if any(w in msg for w in words):
            detected_category = cat
            break

    # -------------------------
    # 3. Category + month (or overall) – numeric answer
    # -------------------------
    if df is not None and detected_category is not None and "signed_amount" in df.columns:
        base_df = df.copy()
        is_income_cat = detected_category == "salary"

        # Month filter if present
        if detected_month:
            month_exists = (base_df["month_name"] == detected_month).any()
            if not month_exists:
                return (
                    f"I don't see any transactions for **{detected_month.capitalize()}** "
                    "in the uploaded statement."
                )
            base_df = base_df[base_df["month_name"] == detected_month]

        # Category filter
        cat_key = detected_category.lower()
        base_df = base_df[base_df["category"] == cat_key]

        if is_income_cat:
            total = base_df["signed_amount"].where(
                base_df["signed_amount"] > 0, 0
            ).sum()
            kind = "income"
        else:
            total = base_df["signed_amount"].where(
                base_df["signed_amount"] < 0, 0
            ).abs().sum()
            kind = "spend"

        total = float(total or 0.0)

        if detected_month:
            month_label = detected_month.capitalize()
            if total == 0:
                return (
                    f"In **{month_label}**, there are no transactions tagged as "
                    f"**{detected_category}** in this statement."
                )
            else:
                return (
                    f"Your **{month_label} {detected_category} {kind}** in this statement "
                    f"is about **₹{round(total)}**."
                )
        else:
            if total == 0:
                return (
                    f"I don't see any transactions tagged as **{detected_category}** "
                    "in this uploaded statement."
                )
            else:
                return (
                    f"Your total **{detected_category} {kind}** in this statement is "
                    f"about **₹{round(total)}**."
                )

    # -------------------------
    # 4. Overspend risk
    # -------------------------
    overspend = future.get("overspend_risk", {}) or {}
    risk_level = overspend.get("level", "low")
    risk_prob = overspend.get("probability", 0)
    risky_cats = future.get("risky_categories", []) or []

    if "risk" in msg:
        lines: list[str] = []
        lines.append(
            f"Your overspend risk this month is **{risk_level}** "
            f"with probability around **{round(risk_prob * 100)}%**."
        )
        if risky_cats:
            cat_text = ", ".join(
                f"{c['name']} (projected ₹{round(c['projected_amount'])})"
                for c in risky_cats
            )
            lines.append(f"Key categories pushing up risk: {cat_text}.")
        return "\n\n".join(lines)

    # -------------------------
    # 5. EMI questions
    # -------------------------
    if "emi" in msg or "loan" in msg:
        emi_load = emi.get("this_month", 0)
        outflow = summary.get("outflow", 0)
        share = (emi_load / outflow * 100) if outflow else 0
        return (
            f"This month your EMI outflow is about **₹{round(emi_load)}**, "
            f"roughly **{share:.1f}%** of your total spend."
        )

    # -------------------------
    # 6. UPI questions
    # -------------------------
    if "upi" in msg:
        upi_month = upi.get("this_month", 0)
        total_upi = upi.get("total_upi", 0)
        handle = upi.get("top_handle") or "no single dominant handle"
        return (
            f"Your UPI spend this month is about **₹{round(upi_month)}** "
            f"(total UPI in this dataset: **₹{round(total_upi)}**). "
            f"Top handle: **{handle}**."
        )

    # -------------------------
    # 7. Default fallback
    # -------------------------
    savings = summary.get("this_month", {}).get("savings", 0)
    safe_daily = summary.get("safe_daily_spend", 0)

    return (
        f"This month you have savings of about **₹{round(savings)}** "
        f"and a safe daily spend around **₹{round(safe_daily)}**.\n\n"
        "You can ask things like:\n"
        "• hospital / medical expenses in January\n"
        "• fuel spend in March\n"
        "• EMI burden this month\n"
        "• why is my risk high\n"
        "• UPI spend share"
    )

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Conversational AI over the current analysis JSON.

    Flow:
    - build_rule_based_reply() computes correct numeric answer (category/month/risk/etc.)
    - If Gemini is available, it rephrases that answer nicely, without changing numbers.
    - If Gemini is not available or errors, we return the rule-based answer directly.
    """
    # Compact JSON snippet just for context (not used for math)
    analysis_snippet = ""
    if payload.analysis:
        try:
            analysis_snippet = json.dumps(payload.analysis, default=str)[:8000]
        except Exception:
            analysis_snippet = ""

    # Always compute numeric-safe helper answer
    tool_answer = build_rule_based_reply(payload.message, payload.analysis)

    # If Gemini is not available, just return the tool answer
    if not GEMINI_AVAILABLE:
        return {"reply": tool_answer}

    system_prompt = (
        "You are CashRaaga, a friendly Indian personal finance coach. "
        "You receive: (1) a JSON summary of the user's bank statement, "
        "(2) the user's question, and (3) a helper answer that already has correct numeric values. "
        "Your job is to explain things in simple, non-judgmental Indian English, "
        "reusing the same rupee amounts and percentages from the helper answer. "
        "Do NOT change any numbers. You can shorten, clarify, or slightly expand the explanation."
    )

    user_prompt = f"""
Here is the current analysis JSON (for context):

{analysis_snippet}

User question:
{payload.message}

Helper answer with correct numbers (do not alter any numeric values):
{tool_answer}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                system_prompt,
                user_prompt,
            ]
        )
        reply = response.text or ""
        if not reply.strip():
            reply = tool_answer
        return {"reply": reply}
    except Exception as e:
        # If Gemini errors, we still fall back to safe numeric answer
        return {"reply": tool_answer, "note": f"Gemini error: {str(e)}"}
