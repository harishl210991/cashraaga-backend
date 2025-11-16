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

# Optional OpenAI client
try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception:
    openai_client = None


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
            # already a string, keep as-is
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


# ---------- CHAT ENDPOINT ----------

class ChatRequest(BaseModel):
  message: str
  analysis: dict | None = None


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Conversational AI over the current analysis JSON.
    """
    if openai_client is None:
        return JSONResponse(
            status_code=500,
            content={
                "error": "LLM backend not configured. Install 'openai' and set OPENAI_API_KEY."
            },
        )

    # Keep JSON short-ish
    analysis_snippet = ""
    if payload.analysis:
        try:
            analysis_snippet = json.dumps(payload.analysis, default=str)[:8000]
        except Exception:
            analysis_snippet = ""

    system_msg = (
        "You are CashRaaga, a friendly Indian personal finance coach. "
        "You receive a JSON summary of the user's bank statement analysis and a question. "
        "Explain things in simple, non-judgmental Indian English. "
        "Focus on: savings, EMI burden, UPI behaviour, safe daily spend, and overspend risk. "
        "Never invent numbers not present in the JSON. Use the JSON as the single source of truth."
    )

    user_msg = f"""
Here is the current analysis JSON:

{analysis_snippet}

User question:
{payload.message}
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Chat backend error: {str(e)}"},
        )
