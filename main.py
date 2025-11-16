from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analysis import analyze_statement

import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

        # 🔧 make everything JSON-safe (kills numpy.int64 problems)
        safe_result = to_native(result)

        return JSONResponse(content=safe_result)

    except Exception as e:
        # send back a simple string, not the whole traceback/list
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
