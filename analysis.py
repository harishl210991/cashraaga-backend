import pandas as pd
import numpy as np
from io import BytesIO
import calendar
from datetime import datetime
import pdfplumber  # PDF support added


# --------------------------------------------------
# 1. PDF LOADER (NEW)
# --------------------------------------------------
def load_pdf_to_df(file_bytes: bytes) -> pd.DataFrame:
    """
    Extracts HDFC-style transaction tables from PDF.
    Maps:
      - Transaction Date -> date
      - Transaction Remarks -> description
      - Withdrawal Amount / Deposit Amount -> amount (+/-)
    """
    rows = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []

            for t in tables:
                if not t or len(t) < 2:
                    continue

                header = t[0]
                header_l = [str(h).strip().lower() if h else "" for h in header]

                # Detect columns
                try:
                    idx_date = header_l.index("transaction date")
                    idx_desc = header_l.index("transaction remarks")
                    idx_wd = header_l.index("withdrawal amount")
                    idx_dep = header_l.index("deposit amount")
                except ValueError:
                    # Not a transaction table → skip
                    continue

                # Read rows
                for r in t[1:]:
                    if idx_dep >= len(r):
                        continue

                    d = r[idx_date]
                    desc = r[idx_desc]
                    wd = r[idx_wd]
                    dep = r[idx_dep]

                    # Validate date
                    try:
                        pd.to_datetime(d, dayfirst=True, errors="raise")
                    except Exception:
                        continue

                    # Clean numbers
                    def to_float(x):
                        x = str(x).replace(",", "").strip()
                        if x in ["", "-", "None", "none", None]:
                            return 0.0
                        try:
                            return float(x)
                        except:
                            return 0.0

                    wd = to_float(wd)
                    dep = to_float(dep)

                    amount = dep - wd  # deposits positive, withdrawals negative

                    rows.append({
                        "date": d,
                        "description": str(desc).strip(),
                        "amount": amount
                    })

    if not rows:
        raise Exception("Could not detect transactions in PDF. Upload CSV/Excel instead.")

    return pd.DataFrame(rows)


# --------------------------------------------------
# 2. FUTURE BLOCK (KEEPING EXACTLY SAME FROM analysis14.py)
# --------------------------------------------------
def build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:

    if df.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    if "date" not in df.columns or "signed_amount" not in df.columns:
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
    outflow_to_date = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()

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
        ratio = predicted_eom_savings / projected_budget_outflow if projected_budget_outflow else 0
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
        level = "high"

    overspend_risk = {
        "level": level,
        "probability": ratio,
    }

    # Risky categories (same logic as original)
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


# --------------------------------------------------
# 3. MAIN ANALYSIS FUNCTION (FROM analysis14 + PDF hook)
# --------------------------------------------------
def analyze_statement(file_bytes, file_name):
    lower = (file_name or "").lower()

    # ------------ FILE LOADING (added PDF path) ------------
    if lower.endswith(".pdf"):
        df = load_pdf_to_df(file_bytes)

    elif lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))

    elif lower.endswith(".xls") or lower.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(file_bytes))

    else:
        raise Exception("Unsupported file type. Upload CSV, XLS, XLSX, or PDF.")

    # ------------ REST OF LOGIC BELOW IS EXACT analysis14.py ------------

    df.columns = df.columns.str.strip().str.lower()

    col_map = {}
    for c in df.columns:
        if "date" in c:
            col_map["date"] = c
        if "desc" in c or "narration" in c or "remarks" in c:
            col_map["description"] = c
        if "amount" in c or "withdraw" in c or "deposit" in c:
            col_map["amount"] = c
        if "type" in c:
            col_map["type"] = c

    df = df.rename(columns=col_map)

    if "date" not in df or "description" not in df or "amount" not in df:
        raise Exception("Missing required columns (Date, Description, Amount).")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper()
        df["signed_amount"] = df.apply(
            lambda r: r["amount"] if r["type"] == "CR" else -abs(r["amount"]),
            axis=1,
        )
    else:
        df["signed_amount"] = df["amount"]

    df = df.dropna(subset=["date", "signed_amount"])
    df = df.sort_values("date")

    # CATEGORY DETECTION
    def detect_category(desc):
        d = str(desc).lower()
        mapping = {
            "salary": ["salary"],
            "rent": ["rent"],
            "shopping": ["amazon", "flipkart", "myntra"],
            "food & dining": ["swiggy", "zomato"],
            "fuel & transport": ["petrol", "uber", "ola"],
            "mobile & internet": ["airtel", "jio"],
            "subscriptions": ["netflix", "hotstar", "prime"],
            "emi": ["emi", "loan"],
            "medical": ["medical", "pharmacy"],
        }
        for cat, keys in mapping.items():
            if any(k in d for k in keys):
                return cat.capitalize()
        return "Others"

    df["category"] = df["description"].apply(detect_category)

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
    else:
        prev_savings = 0
    mom_change = this_month_savings - prev_savings

    upi_df = df[df["description"].str.contains("upi", case=False, na=False)]
    upi_total = upi_df[upi_df["signed_amount"] < 0]["signed_amount"].abs().sum()

    upi_month_df = upi_df[upi_df["month"] == latest_month]
    upi_month_total = upi_month_df["signed_amount"].abs().sum() if not upi_month_df.empty else 0

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
        .sort_values("signed_amount", ascending=False)
    )

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
