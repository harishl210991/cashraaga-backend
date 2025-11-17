import os
import json
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import calendar
import pdfplumber


# ---------------------------- RAW LOADER ----------------------------
def _load_raw(file_bytes, file_name):
    name = file_name.lower()

    if name.endswith(".csv"):
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        text = "\n".join(lines)
        return pd.read_csv(StringIO(text), header=None)

    if name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(file_bytes), header=None)

    # fallback
    try:
        return pd.read_csv(BytesIO(file_bytes), header=None)
    except:
        return pd.read_excel(BytesIO(file_bytes), header=None)


# ---------------------------- HDFC HEADER FINDER ----------------------------
def find_hdfc_header_row(df_raw):
    """
    Scan first 50 rows and find the row containing:
    - Date
    - Narration
    - Withdrawal
    - Deposit
    """
    KEYWORDS = [
        "date",
        "narration",
        "withdraw",
        "deposit",
        "value dt",
        "closing"
    ]

    for i in range(min(50, len(df_raw))):
        row = df_raw.iloc[i].fillna("").astype(str).str.lower().tolist()
        score = sum(1 for cell in row if any(k in cell for k in KEYWORDS))
        if score >= 3:
            return i

    return None


# ---------------------------- HDFC PARSER ----------------------------
def parse_hdfc(df_raw):
    hdr = find_hdfc_header_row(df_raw)
    if hdr is None:
        raise Exception("Unable to locate HDFC header row.")

    header = df_raw.iloc[hdr].fillna("").astype(str).str.strip()
    df = df_raw.iloc[hdr + 1:].reset_index(drop=True).copy()
    df.columns = header

    # Canonicalize column names
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def get_col(keys):
        for k in keys:
            if k in lower:
                return lower[k]
        return None

    date_col = get_col(["date", "transaction date"])
    desc_col = get_col(["narration", "transaction remarks", "description"])
    wd_col = get_col(["withdrawal amt.", "withdrawal amount", "withdrawal"])
    dep_col = get_col(["deposit amt.", "deposit amount", "deposit"])

    if not date_col or not desc_col or not (wd_col or dep_col):
        raise Exception("HDFC: Missing Date/Narration/Withdraw/Deposit columns.")

    # clean numeric
    for col in [wd_col, dep_col]:
        if col:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    withdraw = df[wd_col] if wd_col else 0
    deposit = df[dep_col] if dep_col else 0

    df["signed_amount"] = deposit - withdraw
    df["description"] = df[desc_col].astype(str)
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["signed_amount"] != 0]
    df = df.sort_values("date")

    return df[["date", "description", "signed_amount"]]


# ---------------------------- GENERIC PARSER ----------------------------
def parse_generic(df_raw):
    # detect header similar to HDFC-style but simpler
    hdr = None
    for i in range(min(20, len(df_raw))):
        row = df_raw.iloc[i].fillna("").astype(str).str.lower().tolist()
        if "date" in " ".join(row):
            hdr = i
            break

    if hdr is None:
        raise Exception("GENERIC: Missing header row.")

    header = df_raw.iloc[hdr].astype(str).str.strip()
    df = df_raw.iloc[hdr + 1:].reset_index(drop=True).copy()
    df.columns = header
    df.columns = [str(c).strip() for c in df.columns]

    # try find columns
    date_col = None
    desc_col = None
    amt_col = None

    for c in df.columns:
        cl = c.lower()
        if "date" in cl:
            date_col = c
        if "narrat" in cl or "desc" in cl or "detail" in cl:
            desc_col = c
        if "amount" in cl:
            amt_col = c

    if not (date_col and desc_col and amt_col):
        raise Exception("GENERIC: Missing required columns (Date, Description, Amount).")

    df["signed_amount"] = (
        df[amt_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .astype(float)
    )

    df["description"] = df[desc_col].astype(str)
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    return df[["date", "description", "signed_amount"]]


# ---------------------------- PDF PARSER ----------------------------
def parse_pdf_statement(file_bytes):
    rows = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                header = [str(c).lower() for c in table[0]]
                if not ("transaction date" in " ".join(header) and "withdraw" in " ".join(header)):
                    continue

                # find indices
                idx_date = header.index("transaction date") if "transaction date" in header else None
                idx_desc = next((i for i, c in enumerate(header) if "remarks" in c or "narration" in c), None)
                idx_wd = next((i for i, c in enumerate(header) if "withdraw" in c), None)
                idx_dep = next((i for i, c in enumerate(header) if "deposit" in c), None)

                for row in table[1:]:
                    if not row:
                        continue
                    ds = row[idx_date]
                    try:
                        pd.to_datetime(ds, dayfirst=True, errors="raise")
                    except:
                        continue

                    desc = row[idx_desc] if idx_desc is not None else ""
                    wd = float(str(row[idx_wd]).replace(",", "")) if idx_wd is not None else 0
                    dep = float(str(row[idx_dep]).replace(",", "")) if idx_dep is not None else 0
                    amt = dep - wd

                    rows.append({
                        "date": ds,
                        "description": str(desc),
                        "signed_amount": amt
                    })

    if not rows:
        raise Exception("PDF: Could not parse transaction table.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    return df[["date", "description", "signed_amount"]]


# ---------------------------- CATEGORY DETECTOR ----------------------------
def _detect_category(desc: str) -> str:
    d = str(desc).lower()
    mapping = {
        "salary": ["salary"],
        "rent": ["rent"],
        "shopping": ["amazon", "flipkart", "myntra"],
        "food & dining": ["swiggy", "zomato"],
        "fuel & transport": ["petrol", "uber", "ola"],
        "subscriptions": ["netflix", "hotstar", "prime"],
        "emi": ["emi", "loan"],
        "medical": ["hospital", "clinic", "pharmacy"],
    }
    for cat, keys in mapping.items():
        if any(k in d for k in keys):
            return cat
    return "others"


# ---------------------------- MAIN ENTRY ----------------------------
def analyze_statement(file_bytes, file_name):
    name = file_name.lower()

    # PDF Path
    if name.endswith(".pdf"):
        df = parse_pdf_statement(file_bytes)
    else:
        df_raw = _load_raw(file_bytes, file_name)
        bank = "HDFC" if find_hdfc_header_row(df_raw) is not None else "GENERIC"

        if bank == "HDFC":
            df = parse_hdfc(df_raw)
        else:
            df = parse_generic(df_raw)

    df = df.copy()
    df["category"] = df["description"].apply(_detect_category)
    df["month"] = df["date"].dt.to_period("M")

    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = df[df["signed_amount"] < 0]["signed_amount"].abs().sum()
    net = inflow - outflow

    monthly = df.groupby("month")["signed_amount"].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)

    latest = df["month"].max()
    latest_str = str(latest)

    this_month_df = df[df["month"] == latest]
    this_month_savings = this_month_df["signed_amount"].sum()

    safe_daily_spend = max(this_month_savings, 0) / 30

    return {
        "summary": {
            "inflow": inflow,
            "outflow": outflow,
            "net_savings": net,
            "this_month": {
                "month": latest_str,
                "savings": this_month_savings,
            },
            "safe_daily_spend": safe_daily_spend,
        },
        "category_summary": df.groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
        .sort_values("signed_amount", ascending=False)
        .to_dict(orient="records"),
        "monthly_savings": monthly.to_dict(orient="records"),
        "cleaned_csv": df.to_csv(index=False),
    }
