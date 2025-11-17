import os
import json
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import calendar
import pdfplumber


# ----------------- OPTIONAL: Gemini for column mapping -----------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_MODEL = os.getenv("CASHRAAGA_LLM_MODEL", "gemini-1.5-flash")


def llm_map_columns(headers, preview_rows):
    """
    Optional LLM helper to map weird column names to:
    - date
    - description
    - debit (money going out)
    - credit (money coming in)

    Returns a dict like:
    {
        "date": "Transaction Date",
        "description": "Narration",
        "debit": "Withdrawal Amt.",
        "credit": "Deposit Amt."
    }
    """
    if not genai:
        return {}

    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(GEMINI_MODEL)
        sample_json = json.dumps(
            [
                {h: str(preview_rows[0].get(h, "")) for h in headers},
                {h: str(preview_rows[1].get(h, "")) for h in headers}
                if len(preview_rows) > 1
                else {},
            ],
            indent=2,
        )

        prompt = f"""
You are helping map bank statement columns.

Given these headers:
{headers}

And sample rows:
{sample_json}

Your task:
Return a JSON object with keys among:
- "date"
- "description"
- "debit"
- "credit"
and values = EXACT header names from the list.

If a key is unknown, omit it. Do NOT explain, only output JSON.
"""

        resp = model.generate_content(prompt)
        text = resp.text.strip()

        mapping = json.loads(text)
        if not isinstance(mapping, dict):
            return {}

        mapping = {k: v for k, v in mapping.items() if v in headers}
        return mapping
    except Exception:
        return {}


# ----------------- BASIC RAW LOADER -----------------


def _load_raw(file_bytes, file_name):
    """
    Load CSV / Excel into a raw dataframe without making assumptions on columns.
    """
    name = (file_name or "").lower()

    if name.endswith(".csv"):
        data = BytesIO(file_bytes)
        try:
            df = pd.read_csv(data)
        except UnicodeDecodeError:
            data.seek(0)
            df = pd.read_csv(data, encoding="latin1")
        return df

    # Excel variants
    if name.endswith(".xls") or name.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=0, header=None)
        return df

    # Fallback: try CSV first, then Excel
    data = BytesIO(file_bytes)
    try:
        df = pd.read_csv(data)
        return df
    except Exception:
        data.seek(0)
        df = pd.read_excel(data, sheet_name=0, header=None)
        return df


# ----------------- BANK DETECTION -----------------


def _detect_bank(df_raw: pd.DataFrame) -> str:
    """
    Very lightweight bank detection based on presence of key headers.
    """
    text_sample = " ".join(
        df_raw.astype(str).head(15).fillna("").values.ravel().tolist()
    ).lower()

    if "hdfc bank" in text_sample or "withdrawal amt" in text_sample:
        return "HDFC"

    # Could add more banks later
    return "GENERIC"


# ----------------- HDFC PARSER -----------------


def _guess_header_row(df_raw: pd.DataFrame) -> int:
    """
    Try to locate the header row in weird Excel exports.
    We scan first 20 rows and look for one containing typical HDFC columns.
    """
    candidates = ["date", "narration", "value", "withdrawal", "deposit", "balance"]

    for i in range(min(20, len(df_raw))):
        row = df_raw.iloc[i].fillna("").astype(str).str.lower()
        score = sum(any(c in cell for c in candidates) for cell in row)
        if score >= 3:
            return i
    return 0


def parse_hdfc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    HDFC template:

    Date | Narration | Chq./Ref.No. | Value Dt | Withdrawal Amt. | Deposit Amt. | Closing Balance
    """
    hdr_row = _guess_header_row(df_raw)
    header = df_raw.iloc[hdr_row].fillna("").astype(str).str.strip()
    df = df_raw.iloc[hdr_row + 1 :].reset_index(drop=True).copy()
    df.columns = header
    df = df.dropna(axis=1, how="all")

    # Canonicalize column names
    cols = {c: c for c in df.columns}
    lower_cols = {c.lower(): c for c in df.columns}

    def get_col(keys, default=None):
        for k in keys:
            if k in lower_cols:
                return lower_cols[k]
        return default

    date_col = get_col(["date", "transaction date", "tran date"])
    narr_col = get_col(["narration", "description", "transaction remarks"])
    debit_col = get_col(["withdrawal amt.", "withdrawal amount(inr)", "withdrawal amount"])
    credit_col = get_col(["deposit amt.", "deposit amount(inr)", "deposit amount"])

    # If still missing, use LLM to guess
    if not (date_col and narr_col and (debit_col or credit_col)):
        preview_rows = df.head(5).to_dict(orient="records")
        mapping = llm_map_columns(list(df.columns), preview_rows)
        date_col = mapping.get("date", date_col)
        narr_col = mapping.get("description", narr_col)
        debit_col = mapping.get("debit", debit_col)
        credit_col = mapping.get("credit", credit_col)

    if not date_col or not narr_col or not (debit_col or credit_col):
        raise Exception("Could not detect core columns in HDFC statement.")

    # Clean numbers
    for col in [debit_col, credit_col]:
        if col and col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    debit_vals = df[debit_col] if debit_col else 0.0
    credit_vals = df[credit_col] if credit_col else 0.0

    df["amount"] = credit_vals - debit_vals  # +ve credit, -ve debit
    df["signed_amount"] = df["amount"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df["description"] = df[narr_col].astype(str)
    return df[["date", "description", "signed_amount"]]


# ----------------- GENERIC PARSER -----------------


def _normalize_generic(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    For generic CSV/Excel, ensure first row is header if it looks like one.
    """
    # If there is already a header row (stringy, non-numeric), use it.
    first_row = df_raw.iloc[0].astype(str)
    score = sum(any(c.isalpha() for c in str(x)) for x in first_row)

    if score >= max(3, len(first_row) // 2):
        df = df_raw.copy()
        df.columns = df.iloc[0].astype(str)
        df = df[1:].reset_index(drop=True)
        return df

    # Else treat existing columns as header directly
    df = df_raw.copy()
    df.columns = df.columns.astype(str)
    return df


def parse_generic(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_generic(df_raw)
    df.columns = df.columns.map(lambda c: str(c).strip())

    headers = list(df.columns)

    # Heuristic headers
    def find_col(keywords):
        for c in df.columns:
            name = c.lower()
            if any(k in name for k in keywords):
                return c
        return None

    date_col = find_col(["date"])
    desc_col = find_col(["narrat", "desc", "details", "particular", "remark"])
    debit_col = find_col(["debit", "withdraw", "dr"])
    credit_col = find_col(["credit", "deposit", "cr"])

    # If missing, use LLM mapping
    if not (date_col and desc_col and (debit_col or credit_col)):
        preview_rows = df.head(5).to_dict(orient="records")
        mapping = llm_map_columns(headers, preview_rows)
        date_col = mapping.get("date", date_col)
        desc_col = mapping.get("description", desc_col)
        debit_col = mapping.get("debit", debit_col)
        credit_col = mapping.get("credit", credit_col)

    if not date_col or not desc_col or not (debit_col or credit_col):
        raise Exception("Missing required columns (Date, Description, Amount).")

    # Clean amounts
    for col in [debit_col, credit_col]:
        if col and col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    debit_vals = df[debit_col] if debit_col else 0.0
    credit_vals = df[credit_col] if credit_col else 0.0
    df["amount"] = credit_vals - debit_vals
    df["signed_amount"] = df["amount"]
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df["description"] = df[desc_col].astype(str)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df[["date", "description", "signed_amount"]]


# ----------------- CATEGORY DETECTION -----------------


def _detect_category(desc: str) -> str:
    d = str(desc).lower()

    mapping = {
        "salary": ["salary", "sal ", "payroll"],
        "rent": ["rent"],
        "shopping": ["amazon", "flipkart", "myntra", "purchase"],
        "food & dining": ["swiggy", "zomato", "breakfast", "lunch", "dinner"],
        "fuel & transport": ["petrol", "hpcl", "bpcl", "fuel", "uber", "ola"],
        "mobile & internet": ["airtel", "postpaid", "internet", "recharge", "jio"],
        "subscriptions": ["netflix", "hotstar", "prime video", "spotify"],
        "emi": ["emi", "loan", "repay", "instalment", "installment"],
        "medical": ["medical", "pharmacy", "hospital", "clinic"],
    }
    for cat, keys in mapping.items():
        if any(k in d for k in keys):
            return cat.capitalize()
    return "Others"


# ----------------- PDF PARSER (HDFC-style) -----------------


def parse_pdf_statement(file_bytes, file_name=None) -> pd.DataFrame:
    """
    Parse HDFC-style PDF statement into a canonical dataframe
    with columns: date, description, signed_amount.

    This is designed for PDFs like OpTransactionHistory*.pdf where
    the main transaction table has headers such as:
    S No. | Value Date | Transaction Date | Cheque Number |
    Transaction Remarks | Withdrawal Amount(INR) | Deposit Amount(INR) | Balance(INR)
    """
    rows = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Find the header row that looks like the transaction table
                header_idx = None
                for i, row in enumerate(table):
                    if not row:
                        continue
                    joined = " ".join(str(c) for c in row if c)
                    low = joined.lower()
                    if "transaction date" in low and "withdrawal" in low:
                        header_idx = i
                        break

                if header_idx is None:
                    continue

                header = table[header_idx]
                header_lower = [str(c).strip().lower() if c else "" for c in header]

                def find_idx(keywords):
                    for idx, name in enumerate(header_lower):
                        for k in keywords:
                            if k in name:
                                return idx
                    return None

                idx_date = find_idx(["transaction date", "date"])
                idx_desc = find_idx(
                    ["transaction remarks", "narration", "description", "details"]
                )
                idx_wd = find_idx(["withdrawal"])
                idx_dep = find_idx(["deposit"])

                # Need at least date and (withdraw or deposit)
                if idx_date is None or (idx_wd is None and idx_dep is None):
                    continue

                for row in table[header_idx + 1 :]:
                    if not row or idx_date >= len(row):
                        continue

                    date_cell = row[idx_date]
                    if not date_cell:
                        continue

                    ds = str(date_cell).strip()

                    # Validate that this looks like a date
                    try:
                        pd.to_datetime(ds, dayfirst=True, errors="raise")
                    except Exception:
                        continue

                    # Description / remarks
                    desc = ""
                    if idx_desc is not None and idx_desc < len(row):
                        desc = row[idx_desc] or ""

                    # Helpers for numeric conversion
                    def to_num(x):
                        if x is None:
                            return 0.0
                        s = str(x).replace(",", "").strip()
                        if s in ("", "-", "None", "none"):
                            return 0.0
                        try:
                            return float(s)
                        except Exception:
                            return 0.0

                    wd = (
                        to_num(row[idx_wd])
                        if idx_wd is not None and idx_wd < len(row)
                        else 0.0
                    )
                    dep = (
                        to_num(row[idx_dep])
                        if idx_dep is not None and idx_dep < len(row)
                        else 0.0
                    )

                    amount = dep - wd  # deposits positive, withdrawals negative
                    if amount == 0:
                        continue

                    rows.append(
                        {
                            "date": ds,
                            "description": str(desc).replace("\n", " ").strip(),
                            "signed_amount": amount,
                        }
                    )

    if not rows:
        raise Exception(
            "Could not extract transactions from PDF. Please try the bank's CSV/XLS export instead."
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df[["date", "description", "signed_amount"]]


# ----------------- FUTURE / PREDICTION BLOCK (same as before) -----------------


def build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:
    if df.empty or "date" not in df.columns or "signed_amount" not in df.columns:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M")

    if "category" not in df.columns:
        df["category"] = "Others"

    current_month = df["month"].max()
    if pd.isna(current_month):
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    df_current = df[df["month"] == current_month]
    df_prev = df[df["month"] != current_month]

    if df_current.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    last_date = df_current["date"].max()
    year = last_date.year
    month = last_date.month
    days_in_month = calendar.monthrange(year, month)[1]
    days_elapsed = last_date.day

    inflow_to_date = df_current[df_current["signed_amount"] > 0]["signed_amount"].sum()
    outflow_to_date = -df_current[df_current["signed_amount"] < 0][
        "signed_amount"
    ].sum()

    avg_daily_inflow = inflow_to_date / max(days_elapsed, 1)
    avg_daily_outflow = outflow_to_date / max(days_elapsed, 1)

    predicted_eom_savings = float(
        (avg_daily_inflow - avg_daily_outflow) * days_in_month
    )

    low_bound = float(predicted_eom_savings * 0.85)
    high_bound = float(predicted_eom_savings * 1.15)

    monthly_net = df.groupby("month")["signed_amount"].sum().reset_index()

    if safe_daily_spend and safe_daily_spend > 0:
        projected_budget_outflow = safe_daily_spend * days_in_month
        ratio = (
            predicted_eom_savings / projected_budget_outflow
            if projected_budget_outflow
            else 0
        )
        ratio = 1 - ratio
    else:
        prev_months = monthly_net[monthly_net["month"] < current_month]
        if not prev_months.empty:
            avg_net = prev_months["signed_amount"].tail(3).mean()
        else:
            avg_net = predicted_eom_savings
        ratio = (avg_net - predicted_eom_savings) / abs(avg_net) if avg_net != 0 else 1

    ratio = max(0.0, min(1.0, ratio))

    if ratio < 0.33:
        level = "low"
    elif ratio < 0.66:
        level = "medium"
    else:
        level = "high"

    overspend_risk = {
        "level": level,
        "probability": ratio,
    }

    if df_prev.empty:
        risky_categories = []
    else:
        prev_months_count = max(df_prev["month"].nunique(), 1)

        prev_cat = (
            df_prev[df_prev["signed_amount"] < 0]
            .assign(outflow=lambda x: -x["signed_amount"])
            .groupby("category")["outflow"]
            .sum()
            .rename("total_prev")
            .reset_index()
        )
        prev_cat["baseline"] = prev_cat["total_prev"] / prev_months_count

        curr_cat = (
            df_current[df_current["signed_amount"] < 0]
            .assign(outflow=lambda x: -x["signed_amount"])
            .groupby("category")["outflow"]
            .sum()
            .rename("current")
            .reset_index()
        )

        cat_df = prev_cat.merge(curr_cat, on="category", how="outer").fillna(0.0)

        cat_df["projected"] = (
            cat_df["current"] * days_in_month / max(days_elapsed, 1)
        )

        cat_df["ratio"] = np.where(
            cat_df["baseline"] > 0,
            cat_df["projected"] / cat_df["baseline"],
            0,
        )

        risky_df = (
            cat_df[cat_df["ratio"] >= 1.2]
            .sort_values("ratio", ascending=False)
            .head(3)
        )

        risky_categories = [
            {
                "name": r["category"],
                "projected_amount": float(r["projected"]),
                "baseline_amount": float(r["baseline"]),
            }
            for _, r in risky_df.iterrows()
        ]

    return {
        "predicted_eom_savings": predicted_eom_savings,
        "predicted_eom_range": [low_bound, high_bound],
        "overspend_risk": overspend_risk,
        "risky_categories": risky_categories,
    }


# ----------------- MAIN ENTRY -----------------


def analyze_statement(file_bytes, file_name):
    name = (file_name or "").lower() if file_name else ""

    # PDF path goes through dedicated parser
    if name.endswith(".pdf"):
        core = parse_pdf_statement(file_bytes, file_name)
    else:
        df_raw = _load_raw(file_bytes, file_name)
        bank = _detect_bank(df_raw)

        if bank == "HDFC":
            core = parse_hdfc(df_raw)
        else:
            core = parse_generic(df_raw)

    df = core.copy()
    df["category"] = df["description"].apply(_detect_category)
    df["month"] = df["date"].dt.to_period("M")

    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = df[df["signed_amount"] < 0]["signed_amount"].abs().sum()
    net_savings = inflow - outflow

    monthly = df.groupby("month")["signed_amount"].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)

    latest_month = df["month"].max()
    latest_month_str = str(latest_month)

    this_month_df = df[df["month"] == latest_month]
    this_month_savings = this_month_df["signed_amount"].sum()

    prev_month = latest_month - 1
    if prev_month in df["month"].unique():
        prev_savings = df[df["month"] == prev_month]["signed_amount"].sum()
        mom_change = this_month_savings - prev_savings
    else:
        prev_savings = 0.0
        mom_change = 0.0

    upi_mask = df["description"].astype(str).str.lower().str.contains("upi")
    upi_df = df[(upi_mask) & (df["signed_amount"] < 0)]
    upi_total = upi_df["signed_amount"].abs().sum()

    upi_month_df = upi_df[upi_df["month"] == latest_month]
    upi_month_total = upi_month_df["signed_amount"].abs().sum()

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

    emi_df = df[df["category"] == "Emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = emi_month_df["signed_amount"].abs().sum()
    emi_months = emi_df["month"].nunique()

    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
    ).sort_values("signed_amount", ascending=False)

    safe_daily = max(0, this_month_savings) / 30

    future_block = build_future_block(df, safe_daily)

    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    return {
        "summary": {
            "inflow": inflow,
            "outflow": outflow,
            "net_savings": net_savings,
            "this_month": {
                "month": latest_month_str,
                "savings": this_month_savings,
                "prev_savings": prev_savings,
                "mom_change": mom_change,
            },
            "safe_daily_spend": safe_daily,
        },
        "upi": {
            "this_month": upi_month_total,
            "top_handle": top_upi_handle,
            "total_upi": upi_total,
        },
        "emi": {
            "this_month": emi_load,
            "months_tracked": emi_months,
        },
        "monthly_savings": monthly.to_dict(orient="records"),
        "category_summary": cat_summary.to_dict(orient="records"),
        "cleaned_csv": csv_output,
        "future_block": future_block,
    }
