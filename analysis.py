import os
import calendar
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pdfplumber
import joblib

# These imports are needed so joblib can unpickle sklearn Pipelines
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.ensemble import RandomForestRegressor  # noqa: F401


# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------

BANK_MODEL_PATH = os.getenv("CASHRAAGA_BANK_MODEL", "bank_template_model.joblib")
CATEGORY_MODEL_PATH = os.getenv("CASHRAAGA_CATEGORY_MODEL", "category_model.joblib")
SAVINGS_MODEL_PATH = os.getenv("CASHRAAGA_SAVINGS_MODEL", "savings_model.joblib")


_bank_model = None
_bank_model_loaded = False

_category_model = None
_category_model_loaded = False

_savings_model = None
_savings_model_loaded = False


def _get_bank_model():
    global _bank_model, _bank_model_loaded
    if _bank_model_loaded:
        return _bank_model
    _bank_model_loaded = True
    try:
        if os.path.exists(BANK_MODEL_PATH):
            _bank_model = joblib.load(BANK_MODEL_PATH)
    except Exception:
        _bank_model = None
    return _bank_model


def _get_category_model():
    global _category_model, _category_model_loaded
    if _category_model_loaded:
        return _category_model
    _category_model_loaded = True
    try:
        if os.path.exists(CATEGORY_MODEL_PATH):
            _category_model = joblib.load(CATEGORY_MODEL_PATH)
    except Exception:
        _category_model = None
    return _category_model


def _get_savings_model():
    global _savings_model, _savings_model_loaded
    if _savings_model_loaded:
        return _savings_model
    _savings_model_loaded = True
    try:
        if os.path.exists(SAVINGS_MODEL_PATH):
            _savings_model = joblib.load(SAVINGS_MODEL_PATH)
    except Exception:
        _savings_model = None
    return _savings_model


# ---------------------------------------------------------------------------
# RAW LOADER
# ---------------------------------------------------------------------------


def _load_raw(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Load CSV / Excel into a raw grid (no headers).
    """
    name = (file_name or "").lower()

    if name.endswith(".csv"):
        text = file_bytes.decode("utf-8", errors="ignore")
        # header=None to keep the header row as data
        df_raw = pd.read_csv(StringIO(text), header=None, dtype=str)
        return df_raw

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df_raw = pd.read_excel(BytesIO(file_bytes), header=None, dtype=str)
        return df_raw

    raise Exception(f"Unsupported file type for _load_raw: {file_name}")


# ---------------------------------------------------------------------------
# HEADER ROW DETECTION
# ---------------------------------------------------------------------------


def _find_header_row(df_raw: pd.DataFrame) -> int | None:
    """
    Heuristic to find the most likely header row in the first ~60 rows.
    Scores rows by presence of common header tokens.
    """
    best_idx = None
    best_score = -1

    max_rows = min(60, len(df_raw))

    for i in range(max_rows):
        row = df_raw.iloc[i]
        non_empty = row.notna().sum()
        if non_empty < 2:
            continue

        tokens = [str(x).strip().lower() for x in row if str(x).strip()]
        if not tokens:
            continue

        score = 0
        for t in tokens:
            if any(k in t for k in ["date", "value dt", "txn date", "transaction date"]):
                score += 2
            if any(k in t for k in ["narrat", "remark", "descr", "details", "particular"]):
                score += 2
            if any(k in t for k in ["withdraw", "debit"]):
                score += 1
            if any(k in t for k in ["deposit", "credit"]):
                score += 1
            if "balance" in t:
                score += 1

        if 3 <= len(tokens) <= 10:
            score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _extract_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    From a raw grid (no headers) -> use the detected header row
    to create a proper dataframe with named columns.
    """
    hdr = _find_header_row(df_raw)
    if hdr is None:
        raise Exception("Unable to locate header row.")

    header = df_raw.iloc[hdr].fillna("").astype(str).str.strip().tolist()
    data = df_raw.iloc[hdr + 1 :].reset_index(drop=True).copy()
    data.columns = header
    data = data.dropna(axis=1, how="all")
    return data


# ---------------------------------------------------------------------------
# BANK TEMPLATE DETECTION
# ---------------------------------------------------------------------------


def _predict_bank_from_filename(file_name: str) -> str | None:
    """
    Quick heuristics based on file name.
    """
    name = (file_name or "").lower()
    if "icici" in name:
        return "ICICI"
    if "hdfc" in name:
        return "HDFC"
    if "axis" in name:
        return "AXIS"
    if "kotak" in name:
        return "KOTAK"
    if "sbi" in name or "state bank" in name:
        return "SBI"
    if "hsbc" in name:
        return "HSBC"
    return None


def _predict_bank_template(df_raw: pd.DataFrame | None, file_name: str) -> str | None:
    """
    Hybrid detection:
    1. Filename heuristics
    2. ML model (if available) using first few rows as text
    """
    # 1) filename
    bank = _predict_bank_from_filename(file_name)
    if bank:
        return bank

    # 2) ML model
    model = _get_bank_model()
    if model is None or df_raw is None or df_raw.empty:
        return None

    snippet_rows = []
    max_rows = min(10, len(df_raw))
    for i in range(max_rows):
        row = df_raw.iloc[i].fillna("").astype(str).tolist()
        snippet_rows.append(" | ".join(row))

    text = (file_name or "").lower() + "\n" + "\n".join(snippet_rows)

    try:
        pred = model.predict([text])[0]
        return str(pred)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CANONICALIZATION (DATE / DESCRIPTION / SIGNED_AMOUNT)
# ---------------------------------------------------------------------------


def _canonicalize_table(df_in: pd.DataFrame, bank_type: str | None = None) -> pd.DataFrame:
    """
    Normalize various bank formats into a standard schema:

    - date (datetime64[ns])
    - description (string)
    - signed_amount (float)  (+ = inflow, - = outflow)
    """
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    bank_type = (bank_type or "").upper()

    # Bank-specific hints
    if bank_type == "ICICI":
        date_keys = ["date", "value date", "txn date", "transaction date"]
        desc_keys = ["transaction remarks", "narration", "description", "details", "particular"]
        debit_keys = ["withdrawal amt", "withdrawal", "debit"]
        credit_keys = ["deposit amt", "deposit", "credit"]
        amount_keys = ["amount", "amt"]
        type_keys = ["type", "dr/cr"]
    elif bank_type == "HDFC":
        date_keys = ["date", "value dt"]
        desc_keys = ["narration", "description", "details", "particular"]
        debit_keys = ["withdrawal amt", "debit amt", "withdrawal", "debit"]
        credit_keys = ["deposit amt", "credit amt", "deposit", "credit"]
        amount_keys = ["amount", "amt"]
        type_keys = ["type", "chq./ref.no.", "dr/cr"]
    elif bank_type == "HSBC":
        date_keys = ["date", "txn date"]
        desc_keys = ["narration", "transaction details", "description", "details"]
        debit_keys = ["debit", "withdrawal"]
        credit_keys = ["credit", "deposit"]
        amount_keys = ["amount", "amt"]
        type_keys = ["dr/cr", "type"]
    else:
        # Generic
        date_keys = ["date", "txn date", "transaction date", "value dt"]
        desc_keys = ["narration", "description", "details", "remark", "particular"]
        debit_keys = ["debit", "withdraw", "withdrawal"]
        credit_keys = ["credit", "deposit"]
        amount_keys = ["amount", "amt"]
        type_keys = ["type", "dr/cr"]

    def find_col(keywords):
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in keywords):
                return c
        return None

    date_col = find_col(date_keys)
    desc_col = find_col(desc_keys)
    withdraw_col = find_col(debit_keys)
    deposit_col = find_col(credit_keys)
    amount_col = find_col(amount_keys)
    type_col = find_col(type_keys)

    if date_col is None:
        raise Exception("Missing date column.")
    if desc_col is None:
        # fallback: second column if exists, else first
        desc_col = cols[1] if len(cols) > 1 else cols[0]

    def to_num_series(s):
        s = (
            s.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(s, errors="coerce")

    if withdraw_col:
        df[withdraw_col] = to_num_series(df[withdraw_col]).fillna(0)
    if deposit_col:
        df[deposit_col] = to_num_series(df[deposit_col]).fillna(0)
    if amount_col:
        df[amount_col] = to_num_series(df[amount_col]).fillna(0)

    # Build signed_amount
    signed_amount = None

    # Case 1: separate debit & credit
    if withdraw_col and deposit_col:
        debit_vals = df[withdraw_col]
        credit_vals = df[deposit_col]
        signed_amount = credit_vals - debit_vals

    # Case 2: single amount + DR/CR type
    elif amount_col and type_col:
        amt = df[amount_col]
        t = df[type_col].astype(str).str.upper()
        is_credit = t.str.contains("CR")
        signed_amount = np.where(is_credit, amt, -amt)

    # Case 3: single amount with +/- signs
    elif amount_col:
        signed_amount = df[amount_col]

    else:
        raise Exception("Unable to determine amount columns.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["description"] = df[desc_col].astype(str).str.strip()
    out["signed_amount"] = pd.to_numeric(signed_amount, errors="coerce")

    # Drop rows with no date or no amount
    out = out.dropna(subset=["date", "signed_amount"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# PDF PARSER
# ---------------------------------------------------------------------------


def _parse_pdf(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Generic PDF parser:
    - Extract tables with pdfplumber
    - For each table, run header detection + canonicalization
    - Works for HDFC OpTransactionHistory PDFs and generic Date/Description/Amount/Type PDFs.
    """
    all_tables = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for t in tables:
                df_raw = pd.DataFrame(t)
                try:
                    df_tbl = _extract_table(df_raw)
                    all_tables.append(df_tbl)
                except Exception:
                    # Ignore tables that don't look like statements
                    continue

    if not all_tables:
        raise Exception("No valid tables found in PDF.")

    df_tbl = pd.concat(all_tables, ignore_index=True)
    bank_type = _predict_bank_template(df_tbl, file_name) or _predict_bank_from_filename(
        file_name
    )
    df = _canonicalize_table(df_tbl, bank_type=bank_type)
    return df


# ---------------------------------------------------------------------------
# CATEGORY DETECTION (ML + FALLBACK)
# ---------------------------------------------------------------------------


def _detect_category_fallback(desc: str) -> str:
    """
    Rule-based fallback categoriser when ML model is missing or fails.
    """
    d = str(desc).lower()

    mapping = {
        "salary": ["salary", "sal ", "payroll"],
        "rent": ["rent"],
        "emi": ["emi", "loan", "instalment", "installment"],
        "shopping": ["amazon", "flipkart", "myntra"],
        "food & dining": ["swiggy", "zomato", "restaurant", "hotel", "dining"],
        "groceries": ["bigbasket", "dmart", "more", "reliance fresh", "grocer"],
        "fuel & transport": ["petrol", "diesel", "hpcl", "bpcl", "shell", "uber", "ola"],
        "bills & utilities": ["electricity", "eb bill", "tneb", "water tax"],
        "mobile & internet": [
            "airtel",
            "jio",
            "vodafone",
            "idea",
            "vi ",
            "postpaid",
            "prepaid",
            "broadband",
            "wifi",
        ],
        "upi payment": ["upi/"],
        "medical": ["hospital", "pharmacy", "clinic", "lab"],
        "insurance": ["insurance", "premium"],
    }

    for cat, keys in mapping.items():
        if any(k in d for k in keys):
            return cat

    # If looks like income
    if any(k in d for k in ["neft", "rtgs", "imps", "upi", "salary"]):
        return "income"

    return "others"


def _assign_categories(df: pd.DataFrame) -> pd.Series:
    """
    Try ML category model first, then fall back to keyword rules.
    """
    model = _get_category_model()
    desc_series = df["description"].astype(str)

    if model is not None:
        try:
            preds = model.predict(desc_series.tolist())
            return pd.Series(preds, index=df.index).astype(str).str.lower()
        except Exception:
            pass

    # Fallback
    return desc_series.apply(_detect_category_fallback).astype(str).str.lower()


# ---------------------------------------------------------------------------
# FUTURE BLOCK + OPTIONAL SAVINGS MODEL
# ---------------------------------------------------------------------------


def _build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:
    """
    Build a small "future outlook" block:
    - heuristic projected month-end savings
    - risk level
    - optionally: ML-predicted savings using savings_model.joblib
    """
    if df.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
            "ml_predicted_savings": None,
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")

    current_month = df["month"].max()
    df_current = df[df["month"] == current_month]

    if df_current.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
            "ml_predicted_savings": None,
        }

    last_date = df_current["date"].max()
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    days_elapsed = last_date.day
    days_left = max(0, days_in_month - days_elapsed)

    inflow_so_far = df_current[df_current["signed_amount"] > 0]["signed_amount"].sum()
    outflow_so_far = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()
    savings_so_far = inflow_so_far - outflow_so_far

    # Heuristic: assume spend pace continues
    avg_daily_outflow = outflow_so_far / max(days_elapsed, 1)
    projected_extra_outflow = avg_daily_outflow * days_left
    projected_savings = inflow_so_far - (outflow_so_far + projected_extra_outflow)

    low_proj = projected_savings - 0.1 * abs(projected_savings)
    high_proj = projected_savings + 0.1 * abs(projected_savings)

    # Risk: compare projected daily spend vs safe_daily_spend
    risk_ratio = 0.0
    if safe_daily_spend > 0:
        risk_ratio = avg_daily_outflow / safe_daily_spend

    if risk_ratio <= 0.8:
        risk_level = "low"
        risk_prob = 0.15
    elif risk_ratio <= 1.2:
        risk_level = "medium"
        risk_prob = 0.45
    else:
        risk_level = "high"
        risk_prob = 0.75

    # Risky categories (current month top 3 by spend)
    exp_current = df_current[df_current["signed_amount"] < 0]
    risky_cats = (
        exp_current.groupby("category")["signed_amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(3)
        .reset_index()
        .to_dict(orient="records")
    )

    # Optional ML savings model
    ml_pred = None
    model = _get_savings_model()
    if model is not None:
        try:
            # crude split: fixed = rent + emi; variable = others
            fixed_mask = df_current["category"].str.lower().isin(["rent", "emi"])
            fixed_expenses = -df_current[
                (df_current["signed_amount"] < 0) & fixed_mask
            ]["signed_amount"].sum()
            variable_expenses = -df_current[
                (df_current["signed_amount"] < 0) & ~fixed_mask
            ]["signed_amount"].sum()
            total_income = df_current[df_current["signed_amount"] > 0][
                "signed_amount"
            ].sum()

            features = np.array(
                [[float(total_income), float(fixed_expenses), float(variable_expenses)]]
            )
            ml_pred = float(model.predict(features)[0])
        except Exception:
            ml_pred = None

    return {
        "predicted_eom_savings": float(projected_savings),
        "predicted_eom_range": [float(low_proj), float(high_proj)],
        "overspend_risk": {"level": risk_level, "probability": float(risk_prob)},
        "risky_categories": risky_cats,
        "ml_predicted_savings": ml_pred,
    }


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------


def analyze_statement(file_bytes: bytes, file_name: str) -> dict:
    """
    Main function called by the backend.
    Returns a JSON-serialisable dict with:
      - inflow, outflow, net_savings
      - this_month (savings, prev_savings, mom_change)
      - safe_daily_spend
      - upi, emi
      - monthly_savings, category_summary
      - cleaned_csv (for download)
      - future_block
    """
    name = (file_name or "").lower()

    if name.endswith(".pdf"):
        df = _parse_pdf(file_bytes, file_name)
        bank_type = _predict_bank_from_filename(file_name)
    else:
        df_raw = _load_raw(file_bytes, file_name)
        bank_type = _predict_bank_template(df_raw, file_name)
        df_tbl = _extract_table(df_raw)
        df = _canonicalize_table(df_tbl, bank_type=bank_type)

    # Categories & month
    df["category"] = _assign_categories(df)
    df["month"] = df["date"].dt.to_period("M")

    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = -df[df["signed_amount"] < 0]["signed_amount"].sum()
    net_savings = inflow - outflow

    if df["month"].nunique() == 0:
        raise Exception("No valid transactions found.")

    latest_month = df["month"].max()
    df_latest = df[df["month"] == latest_month]

    inflow_latest = df_latest[df_latest["signed_amount"] > 0]["signed_amount"].sum()
    outflow_latest = -df_latest[df_latest["signed_amount"] < 0]["signed_amount"].sum()
    this_month_savings = inflow_latest - outflow_latest

    # previous month
    prev_months = sorted(df["month"].unique())
    prev_savings = 0.0
    if len(prev_months) >= 2:
        prev_month = prev_months[-2]
        df_prev = df[df["month"] == prev_month]
        prev_in = df_prev[df_prev["signed_amount"] > 0]["signed_amount"].sum()
        prev_out = -df_prev[df_prev["signed_amount"] < 0]["signed_amount"].sum()
        prev_savings = prev_in - prev_out

    if prev_savings != 0:
        mom_change = (this_month_savings - prev_savings) / abs(prev_savings) * 100.0
    else:
        mom_change = 0.0

    # Safe daily spend: assume full savings spread across month days
    if not df_latest.empty:
        last_date = df_latest["date"].max()
        days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    else:
        any_date = df["date"].max()
        days_in_month = calendar.monthrange(any_date.year, any_date.month)[1]

    safe_daily = max(0.0, float(this_month_savings)) / float(days_in_month)

    # Monthly savings summary
    def _agg_month(sub: pd.DataFrame) -> pd.Series:
        inflow_m = sub[sub["signed_amount"] > 0]["signed_amount"].sum()
        outflow_m = -sub[sub["signed_amount"] < 0]["signed_amount"].sum()
        savings_m = inflow_m - outflow_m
        return pd.Series(
            {
                "month": str(sub.name),
                "inflow": float(inflow_m),
                "outflow": float(outflow_m),
                "savings": float(savings_m),
            }
        )

    monthly = df.groupby("month").apply(_agg_month).reset_index(drop=True)

    # UPI info (expenses only)
    upi_df = df[df["description"].str.contains("upi", case=False, na=False)]
    upi_df_exp = upi_df[upi_df["signed_amount"] < 0]

    upi_total = -upi_df_exp["signed_amount"].sum()

    upi_month_df = upi_df_exp[upi_df_exp["month"] == latest_month]
    upi_month_total = -upi_month_df["signed_amount"].sum()

    if not upi_month_df.empty:
        top_upi_handle = (
            upi_month_df.groupby("description")["signed_amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .index[0]
        )
    else:
        top_upi_handle = None

    # EMI info
    emi_df = df[df["category"].str.lower() == "emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = -emi_month_df["signed_amount"].sum()
    emi_months = emi_df["month"].nunique()

    # Category summary (expenses only)
    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )

    future_block = _build_future_block(df, safe_daily_spend=safe_daily)

    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    return {
        "inflow": float(inflow),
        "outflow": float(outflow),
        "net_savings": float(net_savings),
        "this_month": {
            "month": str(latest_month),
            "savings": float(this_month_savings),
            "prev_savings": float(prev_savings),
            "mom_change": float(mom_change),
        },
        "safe_daily_spend": float(safe_daily),
        "upi": {
            "this_month": float(upi_month_total),
            "top_handle": top_upi_handle,
            "total_upi": float(upi_total),
        },
        "emi": {
            "this_month": float(emi_load),
            "months_tracked": int(emi_months),
        },
        "monthly_savings": monthly.to_dict(orient="records"),
        "category_summary": cat_summary.to_dict(orient="records"),
        "cleaned_csv": csv_output,
        "future_block": future_block,
    }
