from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder  # 👈 important

from analysis import analyze_statement

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
        else:
            result["cleaned_csv"] = ""

        # 🔧 make everything JSON-safe (np.int64 -> int, etc.)
        safe_result = jsonable_encoder(result)

        return JSONResponse(content=safe_result)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
