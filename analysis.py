import pandas as pd
import numpy as np
from io import BytesIO
import calendar
from datetime import datetime
import pdfplumber  # <- for PDF table extraction


def load_pdf_to_df(file_bytes) -> pd.DataFrame:
    """
    Try to extract the main transaction table from a bank-statement PDF.

    Strategy:
    - Use pdfplumber to read all pages.
    - Collect all tables.
    - Pick the table with the maximum rows (likely the transaction table).
    - Use the first row as header and the rest as data.
    """
    try:
        tables = []

        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for t in page_tables:
                    # t is a list of rows (each row is a list of cells)
                    if t and len(t) > 1:
                        tables.append(t)

        if not tables:
            raise Exception(
                "Could not detect any tabular data in PDF. "
                "Try uploading CSV/XLSX export of the statement."
            )

        # Pick the table with the most rows
        best = max(tables, key=lambda tbl: len(tbl))

        header = best[0]
        rows = best[1:]

        # Clean header cells
        header = [str(h).strip().lower() if h is not None else "" for h in header]

        df = pd.DataFrame(rows, columns=header)

        return df

    except Exception as e:
        raise Exception(
            f"Failed to parse PDF statement: {e}. "
            "Please try with the bank's CSV or Excel export instead."
        )


def build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:
    """
    Light-weight prediction block based on current month progression
    and historic behaviour. No heavy ML yet, but shaped like it.
    """

    if df.empty:
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "low", "probability": 0.0},
            "risky_categories": [],
        }

    # Ensure we have required fields
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

    # -----------------------------
    # 1. Identify current & previous months
    # -----------------------------
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

    # -----------------------------
    # 2. Project end-of-month savings
    # -----------------------------
    inflow_to_date = df_current[df_current["signed_amount"] > 0]["signed_amount"].sum()
    outflow_to_date = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()

    avg_daily_inflow = inflow_to_date / max(days_elapsed, 1)
    avg_daily_outflow = outflow_to_date / max(days_elapsed, 1)

    projected_inflow = avg_daily_inflow * days_in_month
    projected_outflow = avg_daily_outflow * days_in_month

    predicted_eom_savings = float(projected_inflow - projected_outflow)

    # Simple confidence band: ±15%
    low_bound = float(predicted_eom_savings * 0.85)
    high_bound = float(predicted_eom_savings * 1.15)

    # -----------------------------
    # 3. Overspend risk vs safe_daily_spend / history
    # -----------------------------
    # Monthly net savings history
    monthly_net = (
        df.groupby("month")["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("month")
    )

    # Overspend risk: either vs safe_daily_spend or vs typical net savings
    if safe_daily_spend and safe_daily_spend > 0:
        projected_budget_outflow = safe_daily_spend * days_in_month
        if projected_budget_outflow <= 0:
            ratio = 0.0
        else:
            ratio = projected_outflow / projected_budget_outflow
    else:
        # Compare predicted savings to average of previous 3 months
        prev_months = monthly_net[monthly_net["month"] < current_month]
        if not prev_months.empty:
            last3 = prev_months["signed_amount"].iloc[-3:]
            avg_net_last3 = last3.mean()
        else:
            avg_net_last3 = monthly_net["signed_amount"].iloc[-1]

        if avg_net_last3 == 0:
            ratio = 1.0
        else:
            ratio = max(0.0, (avg_net_last3 - predicted_eom_savings) / abs(avg_net_last3))

    overspend_prob = float(max(0.0, min(1.0, ratio)))

    if overspend_prob < 0.33:
        level = "low"
    elif overspend_prob < 0.66:
        level = "medium"
    else:
        level = "high"

    overspend_risk = {
        "level": level,
        "probability": overspend_prob,
    }

    # -----------------------------
    # 4. Risky categories (where you may overspend)
    # -----------------------------
    if df_prev.empty:
        risky_categories = []
    else:
        # Average monthly outflow per category from previous months
        prev_months_count = max(df_prev["month"].nunique(), 1)

        prev_cat = (
            df_prev[df_prev["signed_amount"] < 0]
            .assign(outflow=lambda x: -x["signed_amount"])
            .groupby("category")["outflow"]
            .sum()
            .rename("total_outflow_prev")
            .reset_index()
        )
        prev_cat["baseline_outflow"] = prev_cat["total_outflow_prev"] / prev_months_count

        # Current month outflow to date per category
        curr_cat = (
            df_current[df_current["signed_amount"] < 0]
            .assign(outflow=lambda x: -x["signed_amount"])
            .groupby("category")["outflow"]
            .sum()
            .rename("current_outflow_to_date")
            .reset_index()
        )

        cat_df = prev_cat.merge(curr_cat, on="category", how="outer").fillna(0.0)

        if days_elapsed > 0:
            cat_df["projected_outflow"] = (
                cat_df["current_outflow_to_date"] * days_in_month / days_elapsed
            )
        else:
            cat_df["projected_outflow"] = cat_df["current_outflow_to_date"]

        cat_df["overrun_ratio"] = np.where(
            cat_df["baseline_outflow"] > 0,
            cat_df["projected_outflow"] / cat_df["baseline_outflow"],
            np.inf,
        )

        risky_df = (
            cat_df[cat_df["overrun_ratio"] >= 1.2]
            .sort_values("overrun_ratio", ascending=False)
            .head(3)
        )

        risky_categories = [
            {
                "name": row["category"],
                "projected_amount": float(row["projected_outflow"]),
                "baseline_amount": float(row["baseline_outflow"]),
            }
            for _, row in risky_df.iterrows()
        ]

    return {
        "predicted_eom_savings": predicted_eom_savings,
        "predicted_eom_range": [low_bound, high_bound],
        "overspend_risk": overspend_risk,
        "risky_categories": risky_categories,
    }


def analyze_statement(file_bytes, file_name):
    # -----------------------------
    # 1. LOAD FILE (CSV / XLSX / PDF)
    # -----------------------------
    name_lower = (file_name or "").lower()
    ext = ""
    if "." in name_lower:
        ext = name_lower.rsplit(".", 1)[-1]

    if ext == "pdf":
        # New: PDF path
        df = load_pdf_to_df(file_bytes)
    elif ext == "csv":
        df = pd.read_csv(BytesIO(file_bytes))
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        # Fallback: try CSV first, then Excel
        try:
            df = pd.read_csv(BytesIO(file_bytes))
        except Exception:
            try:
                df = pd.read_excel(BytesIO(file_bytes))
            except Exception:
                raise Exception(
                    "Unsupported file format. Please upload a CSV, Excel, or PDF bank statement."
                )

    # -----------------------------
    # 2. STANDARDIZE COLUMNS
    # -----------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Required minimal mapping
    col_map = {}
    for c in df.columns:
        if "date" in c and "value" not in c:
            # Try to avoid mapping "value date" instead of the main "date"
            col_map["date"] = c
        if any(k in c for k in ["desc", "particular", "narration", "details"]):
            col_map["description"] = c
        if "amount" in c or "amt" in c:
            # generic amount column
            col_map["amount"] = c
        if "type" in c or "cr/dr" in c or "dr/cr" in c:
            col_map["type"] = c

    df = df.rename(columns=col_map)

    # Ensure existence
    if "date" not in df or "description" not in df or "amount" not in df:
        raise Exception("Missing required columns (Date, Description, Amount).")

    # -----------------------------
    # 3. CLEAN DATE + AMOUNT
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # if CSV/PDF has commas or ₹ symbol or extra spaces
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("INR", "", case=False, regex=False)
        .str.strip()
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Apply CR/DR correction if provided
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper()
        df["signed_amount"] = df.apply(
            lambda row: row["amount"] if row["type"] == "CR" else -abs(row["amount"]),
            axis=1,
        )
    else:
        # If no type column exists, assume positive = inflow, negative = outflow
        df["signed_amount"] = df["amount"]

    # Remove invalid rows
    df = df.dropna(subset=["date", "signed_amount"])

    # Sort properly
    df = df.sort_values("date")

    # -----------------------------
    # 4. CATEGORY MAPPING
    # -----------------------------
    def detect_category(desc):
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

    df["category"] = df["description"].astype(str).apply(detect_category)

    # -----------------------------
    # 5. AGGREGATES
    # -----------------------------
    df["month"] = df["date"].dt.to_period("M")

    # TOTAL INFLOW / OUTFLOW
    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = df[df["signed_amount"] < 0]["signed_amount"].abs().sum()
    net_savings = inflow - outflow

    # -----------------------------
    # 6. MONTHLY SAVINGS
    # -----------------------------
    monthly = df.groupby("month")["signed_amount"].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)

    # Latest month
    latest_month = df["month"].max()
    latest_month_str = str(latest_month)

    this_month_df = df[df["month"] == latest_month]
    this_month_savings = this_month_df["signed_amount"].sum()

    # Compare to previous month
    prev_month = latest_month - 1
    if prev_month in df["month"].unique():
        prev_savings = df[df["month"] == prev_month]["signed_amount"].sum()
        mom_change = this_month_savings - prev_savings
    else:
        prev_savings = 0
        mom_change = 0

    # -----------------------------
    # 7. UPI OUTFLOW
    # -----------------------------
    upi_mask = df["description"].astype(str).str.lower().str.contains("upi")
    upi_df = df[(upi_mask) & (df["signed_amount"] < 0)]
    upi_total = upi_df["signed_amount"].abs().sum()

    # Latest month UPI
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

    # -----------------------------
    # 8. EMI LOAD
    # -----------------------------
    emi_df = df[df["category"] == "Emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = emi_month_df["signed_amount"].abs().sum()
    emi_months = emi_df["month"].nunique()

    # -----------------------------
    # 9. CATEGORY SUMMARY (DONUT)
    # -----------------------------
    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
    ).sort_values("signed_amount", ascending=False)

    # -----------------------------
    # 10. SAFE DAILY SPEND
    # -----------------------------
    # safe spend = (this month savings) / 30 (if positive)
    safe_daily = max(0, this_month_savings) / 30

    # -----------------------------
    # 11. FUTURE / PREDICTION BLOCK
    # -----------------------------
    future_block = build_future_block(df, safe_daily)

    # -----------------------------
    # 12. CLEANED CSV EXPORT
    # -----------------------------
    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)

    csv_output = cleaned_export.to_csv(index=False)

    # -----------------------------
    # 13. BUILD RESULT JSON
    # -----------------------------
    result = {
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
        # ML-style predictions block
        "future_block": future_block,
    }

    return result
