import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import calendar
import pdfplumber


# ---------------------------- RAW LOADER ----------------------------

def _load_raw(file_bytes, file_name):
    name = (file_name or "").lower()

    if name.endswith(".csv"):
        # Read raw text
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # Detect "whole row wrapped in quotes" pattern
        sample = lines[:20]
        quoted = 0
        for ln in sample:
            s = ln.strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and s.count('"') == 2:
                quoted += 1

        # If most lines are like "Date,Description,Amount,Type"
        if quoted >= max(1, len(sample) // 2):
            lines = [ln.strip().strip('"') for ln in lines]
            text = "\n".join(lines)

        df_raw = pd.read_csv(StringIO(text), header=None)
        return df_raw

    elif name.endswith(".xls") or name.endswith(".xlsx"):
        # Read everything as data, we will find the header later
        return pd.read_excel(BytesIO(file_bytes), header=None)

    # Fallback: try CSV then Excel
    try:
        return pd.read_csv(BytesIO(file_bytes), header=None)
    except Exception:
        return pd.read_excel(BytesIO(file_bytes), header=None)


# ---------------------------- HEADER & TABLE EXTRACTION ----------------------------

def _find_header_row(df_raw):
    """
    Scan first ~60 rows and find the row that looks most like the header:
    containing words like date, narration, amount, withdraw, deposit, etc.
    Works for both dummy CSV and HDFC XLS (header around row 20).
    """
    KEYWORDS = [
        "date",
        "narration",
        "remark",
        "description",
        "details",
        "withdraw",
        "deposit",
        "debit",
        "credit",
        "amount",
        "balance",
    ]

    best_idx = None
    best_score = -1

    for i in range(min(60, len(df_raw))):
        row = df_raw.iloc[i].fillna("").astype(str).str.lower().tolist()
        nonempty = [c for c in row if c.strip() != ""]
        if not nonempty:
            continue

        score = 0
        for cell in row:
            if any(k in cell for k in KEYWORDS):
                score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _extract_table(df_raw):
    """
    From a raw grid (no headers) -> use the detected header row to create a
    proper dataframe with named columns.
    """
    hdr = _find_header_row(df_raw)
    if hdr is None:
        raise Exception("Unable to locate header row.")

    header = df_raw.iloc[hdr].fillna("").astype(str).str.strip().tolist()
    data = df_raw.iloc[hdr + 1 :].reset_index(drop=True).copy()
    data.columns = header
    data = data.dropna(axis=1, how="all")
    return data


# ---------------------------- CANONICALIZATION ----------------------------

def _canonicalize_table(df_in):
    """
    Convert any "Date / Narration / Debit / Credit / Amount / Type" layout
    into a canonical dataframe with columns:
    - date
    - description
    - signed_amount  (+ = inflow, - = outflow)
    """
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    def find_col(keywords):
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in keywords):
                return c
        return None

    date_col = find_col(["date"])
    desc_col = find_col(["narrat", "remark", "descr", "details", "particular"])

    withdraw_col = find_col(["withdraw", "debit"])
    deposit_col = find_col(["deposit", "credit"])
    amount_col = find_col(["amount", "amt"])
    type_col = find_col(["type"])

    if date_col is None:
        raise Exception("Missing date column.")

    if desc_col is None:
        # Fallback: second column if we didn't find a better one
        if len(cols) >= 2:
            desc_col = cols[1]
        else:
            desc_col = cols[0]

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
        df[amount_col] = to_num_series(df[amount_col])

    # Decide signed_amount
    if withdraw_col or deposit_col:
        wd = df[withdraw_col] if withdraw_col else 0
        dep = df[deposit_col] if deposit_col else 0
        signed = dep - wd
    elif amount_col:
        signed = df[amount_col]
        # If we have a Type column with CR/DR and amount is mostly positive,
        # use the type to decide the sign.
        if type_col:
            type_series = df[type_col].astype(str).str.upper()
            if signed.dropna().ge(0).mean() > 0.9:
                signed = signed.where(type_series == "CR", -signed.abs())
    else:
        raise Exception("Missing amount/debit/credit columns.")

    df_out = pd.DataFrame()
    df_out["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df_out["description"] = df[desc_col].astype(str)
    df_out["signed_amount"] = signed

    df_out = df_out.dropna(subset=["date"])
    df_out = df_out[df_out["signed_amount"].notna()]
    df_out = df_out.sort_values("date")

    return df_out[["date", "description", "signed_amount"]]


# ---------------------------- PDF PARSER ----------------------------

def _parse_pdf(file_bytes):
    """
    Generic PDF parser:
    - Extract tables with pdfplumber
    - For each table, run the same header detection + canonicalization
    - Works for HDFC OpTransactionHistory PDFs and generic Date/Description/Amount/Type PDFs.
    """
    all_rows = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for t in tables:
                df_raw = pd.DataFrame(t)
                try:
                    df_tbl = _extract_table(df_raw)
                    core = _canonicalize_table(df_tbl)
                    all_rows.append(core)
                except Exception:
                    # Ignore tables that don't look like statements
                    continue

    if not all_rows:
        raise Exception("PDF: Could not parse transaction table.")

    df_all = pd.concat(all_rows, ignore_index=True)
    return df_all


# ---------------------------- CATEGORIES ----------------------------

def _detect_category(desc: str) -> str:
    d = str(desc).lower()
    mapping = {
        "salary": ["salary"],
        "rent": ["rent"],
        "shopping": ["amazon", "flipkart", "myntra"],
        "food & dining": ["swiggy", "zomato", "restaurant", "dining"],
        "fuel & transport": ["petrol", "diesel", "hpcl", "bpcl", "shell", "uber", "ola"],
        "subscriptions": ["netflix", "hotstar", "prime", "spotify"],
        "emi": ["emi", "loan", "instalment", "installment"],
        "medical": ["hospital", "pharmacy", "clinic", "lab"],
        "mobile & internet": ["airtel", "jio", "vodafone", "idea", "vi", "postpaid", "prepaid", "broadband", "wifi"],
    }
    for cat, keys in mapping.items():
        if any(k in d for k in keys):
            return cat
    return "others"


# ---------------------------- FUTURE BLOCK ----------------------------

def _build_future_block(df, safe_daily_spend):
    if df.empty:
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

    current_month = df["month"].max()
    df_current = df[df["month"] == current_month]

    if df_current.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    last_date = df_current["date"].max()
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    days_elapsed = last_date.day

    inflow_to_date = df_current[df_current["signed_amount"] > 0]["signed_amount"].sum()
    outflow_to_date = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()

    avg_inflow = inflow_to_date / max(days_elapsed, 1)
    avg_outflow = outflow_to_date / max(days_elapsed, 1)

    predicted_eom = (avg_inflow - avg_outflow) * days_in_month
    low = predicted_eom * 0.85
    high = predicted_eom * 1.15

    if safe_daily_spend > 0:
        projected_budget_outflow = safe_daily_spend * days_in_month
        if projected_budget_outflow > 0:
            ratio = max(
                0.0,
                min(1.0, float(outflow_to_date) / float(projected_budget_outflow)),
            )
        else:
            ratio = 0.0
    else:
        ratio = 0.5

    level = "low" if ratio < 0.33 else "medium" if ratio < 0.66 else "high"

    df_current_neg = df_current[df_current["signed_amount"] < 0].copy()
    df_current_neg["outflow"] = -df_current_neg["signed_amount"]
    risky = []

    if "category" in df_current_neg.columns and not df_current_neg.empty:
        grp = (
            df_current_neg.groupby("category")["outflow"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        for cat, val in grp.items():
            risky.append(
                {"name": cat, "projected_amount": float(val), "baseline_amount": 0.0}
            )

    return {
        "predicted_eom_savings": float(predicted_eom),
        "predicted_eom_range": [float(low), float(high)],
        "overspend_risk": {"level": level, "probability": float(ratio)},
        "risky_categories": risky,
    }


# ---------------------------- MAIN ENTRY ----------------------------

def analyze_statement(file_bytes, file_name):
    name = (file_name or "").lower()

    if name.endswith(".pdf"):
        df = _parse_pdf(file_bytes)
    else:
        df_raw = _load_raw(file_bytes, file_name)
        df_tbl = _extract_table(df_raw)
        df = _canonicalize_table(df_tbl)

    # Categories & month
    df["category"] = df["description"].apply(_detect_category)
    df["month"] = df["date"].dt.to_period("M")

    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = -df[df["signed_amount"] < 0]["signed_amount"].sum()
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

    # UPI info
    upi_mask = df["description"].str.contains("upi", case=False, na=False)
    upi_df = df[upi_mask & (df["signed_amount"] < 0)]
    upi_total = -upi_df["signed_amount"].sum()

    upi_month_df = upi_df[upi_df["month"] == latest_month]
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
    emi_df = df[df["category"] == "emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = -emi_month_df["signed_amount"].sum()
    emi_months = emi_df["month"].nunique()

    # Category summary
    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
    )

    safe_daily = max(0, this_month_savings) / 30
    future_block = _build_future_block(df, safe_daily)

    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    return {
        "summary": {
            "inflow": float(inflow),
            "outflow": float(outflow),
            "net_savings": float(net_savings),
            "this_month": {
                "month": latest_month_str,
                "savings": float(this_month_savings),
                "prev_savings": float(prev_savings),
                "mom_change": float(mom_change),
            },
            "safe_daily_spend": float(safe_daily),
        },
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
