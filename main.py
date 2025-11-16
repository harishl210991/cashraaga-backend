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
    Simple fallback explainer if Gemini is not configured or errors out.
    Uses the analysis JSON directly.
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

    savings = summary.get("this_month", {}).get("savings", 0)
    safe_daily = summary.get("safe_daily_spend", 0)
    outflow = summary.get("outflow", 0)
    overspend = future.get("overspend_risk", {}) or {}
    risk_level = overspend.get("level", "low")
    risk_prob = overspend.get("probability", 0)
    risky_cats = future.get("risky_categories", []) or []

    lines: list[str] = []

    msg = (message or "").lower()

    if "risk" in msg:
        lines.append(
            f"Your overspend risk this month is **{risk_level}** "
            f"with probability around {round(risk_prob * 100)}%."
        )
        if risky_cats:
            cats = ", ".join(
                f"{c['name']} (projected ₹{round(c['projected_amount'])})"
                for c in risky_cats
            )
            lines.append(f"The main categories pushing up risk are: {cats}.")
        if safe_daily > 0:
            lines.append(
                f"Try to keep daily spends close to your safe limit of about ₹{round(safe_daily)} "
                "for the rest of the month, and trim non-essential spends in those risky categories."
            )
    elif "emi" in msg or "loan" in msg:
        emi_load = emi.get("this_month", 0)
        months_tracked = emi.get("months_tracked", 0)
        share = (emi_load / outflow * 100) if outflow else 0
        lines.append(
            f"This month your EMI outflow is about ₹{round(emi_load)}, "
            f"roughly {share:.1f}% of your total spend."
        )
        lines.append(
            f"We've seen EMIs for {months_tracked} month(s). "
            "If this EMI share goes above ~40–45% of your income, your budget can feel tight."
        )
    elif "upi" in msg:
        upi_month = upi.get("this_month", 0)
        total_upi = upi.get("total_upi", 0)
        top_handle = upi.get("top_handle")
        lines.append(
            f"This month, your UPI outflow is about ₹{round(upi_month)} "
            f"(total UPI in data: ₹{round(total_upi)})."
        )
        if top_handle:
            lines.append(f"You mostly pay via: {top_handle}.")
    else:
        lines.append(
            f"This month you have savings of about ₹{round(savings)} "
            f"and a safe daily spend around ₹{round(safe_daily)}."
        )
        lines.append(
            "Ask specific questions like: "
            "\"Why is my risk high?\", \"How heavy is my EMI?\" or "
            "\"What should I cut this month?\""
        )

    return "\n\n".join(lines)


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Conversational AI over the current analysis JSON.
    Uses Gemini 1.5 Flash when configured, otherwise a rule-based reply.
    """
    # Build a compact JSON snippet
    analysis_snippet = ""
    if payload.analysis:
        try:
            analysis_snippet = json.dumps(payload.analysis, default=str)[:8000]
        except Exception:
            analysis_snippet = ""

    # If Gemini is not available, use fallback logic
    if not GEMINI_AVAILABLE:
        reply = build_rule_based_reply(payload.message, payload.analysis)
        return {"reply": reply}

    system_prompt = (
        "You are CashRaaga, a friendly Indian personal finance coach. "
        "You receive a JSON summary of the user's bank statement analysis and a question. "
        "Explain things in simple, non-judgmental Indian English. "
        "Focus on: savings, EMI burden, UPI behaviour, safe daily spend, and overspend risk. "
        "Never invent numbers that are not present in the JSON. "
        "Use only the JSON as the source of all numeric values."
    )

    user_prompt = f"""
Here is the current analysis JSON:

{analysis_snippet}

User question:
{payload.message}
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
            reply = build_rule_based_reply(payload.message, payload.analysis)
        return {"reply": reply}
    except Exception as e:
        # If Gemini errors, fall back to rule-based
        fallback = build_rule_based_reply(payload.message, payload.analysis)
        return {"reply": fallback, "note": f"Gemini error: {str(e)}"}
