import pandas as pd
import numpy as np
import io
from datetime import datetime


# -----------------------------
# UTILITIES
# -----------------------------

def smart_parse_date(x):
    """
    Robust Indian bank date parser.
    """
    if pd.isna(x):
        return None

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(x).strip(), fmt)
        except:
            pass

    try:
        return pd.to_datetime(x, errors="coerce")
    except:
        return None


def detect_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}

    # DATE
    date_col = None
    for k, v in cols.items():
        if "date" in k:
            date_col = v
            break

    # DESCRIPTION
    desc_col = None
    for k, v in cols.items():
        if any(t in k for t in ["description", "narration", "details", "remarks", "particular"]):
            desc_col = v
            break

    # AMOUNT or CREDIT/DEBIT
    amount_col = None
    credit_col = None
    debit_col = None

    for k, v in cols.items():
        if "amount" in k:
            amount_col = v
        if "credit" in k and "interest" not in k:
            credit_col = v
        if "debit" in k:
            debit_col = v

    return date_col, desc_col, amount_col, credit_col, debit_col


def build_signed_amount(df, amount_col, credit_col, debit_col):
    """
    Creates SignedAmount column:
    - Positive for inflow
    - Negative for outflow
    """

    if amount_col:
        amt = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        # check if Type column exists
        tcol = None
        for c in df.columns:
            if c.lower() in ["type", "dr/cr", "cr/dr", "transaction type"]:
                tcol = c
                break

        if tcol:
            t = df[tcol].astype(str).str.upper()
            dr = t.str.contains("DR")
            cr = t.str.contains("CR")

            # detect if already signed
            if (amt[dr].mean() < 0) and (amt[cr].mean() > 0):
                return amt  # already signed

            # else: assume absolute amounts
            return np.where(cr, amt.abs(), -amt.abs())

        # NO type column = we assume file has signed values
        return amt

    elif credit_col or debit_col:
        cr = pd.to_numeric(df.get(credit_col, 0), errors="coerce").fillna(0)
        dr = pd.to_numeric(df.get(debit_col, 0), errors="coerce").fillna(0)
        return cr - dr

    raise ValueError("Unable to determine amounts in statement.")


# -----------------------------
# CATEGORY ENGINE
# -----------------------------

def classify_category(description):
    d = description.lower()

    if "salary" in d:
        return "Salary"
    if "rent" in d:
        return "Rent"
    if "emi" in d:
        return "EMI"
    if "amazon" in d or "flipkart" in d:
        return "Shopping"
    if "swiggy" in d or "zomato" in d:
        return "Food & Dining"
    if "petrol" in d or "hpcl" in d or "bpcl" in d or "uber" in d:
        return "Fuel & Transport"
    if "airtel" in d or "internet" in d or "postpaid" in d:
        return "Mobile & Internet"
    if "subscription" in d or "netflix" in d or "prime video" in d or "hotstar" in d:
        return "Subscriptions"
    if "medical" in d or "pharmacy" in d:
        return "Medical"

    return "Others"


# -----------------------------
# MAIN ANALYZER
# -----------------------------

def analyze_statement(file_bytes, filename):
    """
    Main entry point called by FastAPI
    """

    # load file
    if filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    date_col, desc_col, amount_col, credit_col, debit_col = detect_columns(df)

    if not date_col or not desc_col:
        raise ValueError("Could not detect Date or Description columns.")

    df = df[[date_col, desc_col] + [c for c in df.columns if c not in [date_col, desc_col]]].copy()
    df.rename(columns={date_col: "Date", desc_col: "Description"}, inplace=True)

    # clean
    df["Date"] = df["Date"].apply(smart_parse_date)
    df = df.dropna(subset=["Date"])
    df["Description"] = df["Description"].astype(str).fillna("").str.strip()

    # SIGNED AMOUNT
    df["SignedAmount"] = build_signed_amount(df, amount_col, credit_col, debit_col)

    # CATEGORY
    df["Category"] = df["Description"].apply(classify_category)

    # MONTH
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # SUMMARY
    inflow = df[df["SignedAmount"] > 0]["SignedAmount"].sum()
    outflow = df[df["SignedAmount"] < 0]["SignedAmount"].abs().sum()
    net_savings = inflow - outflow

    # CURRENT MONTH
    monthly = df.groupby("Month")["SignedAmount"].sum().reset_index()
    monthly.columns = ["Month", "Savings"]
    monthly_sorted = monthly.sort_values("Month")

    if len(monthly_sorted) > 0:
        current_row = monthly_sorted.iloc[-1]
        this_month = current_row["Month"]
        this_savings = current_row["Savings"]
        prev_savings = (
            monthly_sorted.iloc[-2]["Savings"] if len(monthly_sorted) >= 2 else 0
        )
        growth = this_savings - prev_savings
        growth_text = f"{growth:+} vs last month"
    else:
        this_month, this_savings, growth_text = None, 0, "0"

    # SAFE DAILY SPEND
    safe_daily_spend = max(int(this_savings / 30), 0)

    # -----------------------------
    # UPI DETECTION (fix)
    # -----------------------------
    upi_df = df[df["Description"].str.contains("upi", case=False, na=False)]
    upi_net = upi_df[upi_df["SignedAmount"] < 0]["SignedAmount"].abs().sum()

    # top UPI counterparty
    top_upi = (
        upi_df.groupby("Description")["SignedAmount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

    if len(top_upi) > 0:
        top_handle = top_upi.index[0]
    else:
        top_handle = None

    # -----------------------------
    # EMI
    # -----------------------------
    emi_df = df[df["Description"].str.contains("emi", case=False, na=False)]
    emi_by_month = (
        emi_df.groupby("Month")["SignedAmount"].sum().abs().reset_index()
        if not emi_df.empty
        else []
    )
    emi_load = (
        emi_by_month[emi_by_month["Month"] == this_month]["SignedAmount"].sum()
        if this_month
        else 0
    )

    # -----------------------------
    # CATEGORY SUMMARY
    # -----------------------------
    cat_summary = (
        df[df["SignedAmount"] < 0]
        .groupby("Category")["SignedAmount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"Category": "Category", "SignedAmount": "TotalAmount"})
    )

    # -----------------------------
    # CLEANED PREVIEW
    # -----------------------------
    cleaned_preview = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned_preview["Date"] = cleaned_preview["Date"].dt.strftime("%Y-%m-%d")

    cleaned_csv = cleaned_preview.to_csv(index=False)

    # -----------------------------
    # BUILD FINAL JSON
    # -----------------------------
    result = {
        "summary": {
            "total_inflow": int(inflow),
            "total_outflow": int(outflow),
            "savings_total": int(net_savings),
            "current_month": this_month,
            "current_month_savings": int(this_savings),
            "growth_text": growth_text,
            "upi_net_outflow": int(upi_net),
            "emi_load": int(emi_load),
            "safe_daily_spend": int(safe_daily_spend),
        },
        "monthly": monthly_sorted.to_dict(orient="records"),
        "categories": cat_summary.to_dict(orient="records"),
        "upi": {
            "top_counterparties": [
                {"Description": i, "TotalAmount": float(v)}
                for i, v in top_upi.items()
            ]
        },
        "emi": {
            "by_month": emi_by_month.to_dict(orient="records")
            if len(emi_by_month) > 0
            else []
        },
        "forecast": {
            "available": False  # disabled for now
        },
        "cleaned_preview": cleaned_preview.to_dict(orient="records"),
        "cleaned_csv": cleaned_csv,
    }

    return result
