import pandas as pd
import numpy as np
from io import BytesIO

def analyze_statement(file_bytes, file_name):
    # -----------------------------
    # 1. LOAD FILE
    # -----------------------------
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    # -----------------------------
    # 2. STANDARDIZE COLUMNS
    # -----------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Required minimal mapping
    col_map = {}
    for c in df.columns:
        if "date" in c:
            col_map["date"] = c
        if "desc" in c:
            col_map["description"] = c
        if "amount" in c:
            col_map["amount"] = c
        if "type" in c:
            col_map["type"] = c

    df = df.rename(columns=col_map)

    # Ensure existence
    if "date" not in df or "description" not in df or "amount" not in df:
        raise Exception("Missing required columns (Date, Description, Amount).")

    # -----------------------------
    # 3. CLEAN DATE + AMOUNT
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # if CSV has commas or ₹ symbol
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
    )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Apply CR/DR correction if provided
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper()
        df["signed_amount"] = df.apply(
            lambda row: row["amount"] if row["type"] == "CR" else -abs(row["amount"]),
            axis=1
        )
    else:
        # If no type column exists, assume negative values are outflow
        df["signed_amount"] = df["amount"]

    # Remove invalid rows
    df = df.dropna(subset=["date", "signed_amount"])

    # Sort properly
    df = df.sort_values("date")

    # -----------------------------
    # 4. CATEGORY MAPPING
    # -----------------------------
    def detect_category(desc):
        d = desc.lower()

        mapping = {
            "salary": ["salary"],
            "rent": ["rent"],
            "shopping": ["amazon", "flipkart", "myntra", "purchase"],
            "food & dining": ["swiggy", "zomato", "breakfast", "lunch", "dinner"],
            "fuel & transport": ["petrol", "hpcl", "bpcl", "fuel", "uber", "ola"],
            "mobile & internet": ["airtel", "postpaid", "internet", "recharge"],
            "subscriptions": ["netflix", "hotstar", "prime video"],
            "emi": ["emi", "loan"],
            "medical": ["medical", "pharmacy", "hospital"],
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
    monthly = (
        df.groupby("month")["signed_amount"].sum().reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)

    # Latest month
    latest_month = df["month"].max()
    latest_month_str = str(latest_month)

    this_month_df = df[df["month"] == latest_month]
    this_month_savings = this_month_df["signed_amount"].sum()

    # Compare to previous month
    prev_month = (latest_month - 1)
    if prev_month in df["month"].unique():
        prev_savings = df[df["month"] == prev_month]["signed_amount"].sum()
        mom_change = this_month_savings - prev_savings
    else:
        prev_savings = 0
        mom_change = 0

    # -----------------------------
    # 7. UPI OUTFLOW
    # -----------------------------
    upi_mask = df["description"].str.lower().str.contains("upi")
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
    # 11. CLEANED CSV EXPORT
    # -----------------------------
    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)

    csv_output = cleaned_export.to_csv(index=False)

    # -----------------------------
    # 12. BUILD RESULT JSON
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
                "mom_change": mom_change
            },
            "safe_daily_spend": safe_daily
        },

        "upi": {
            "this_month": upi_month_total,
            "top_handle": top_upi_handle,
            "total_upi": upi_total
        },

        "emi": {
            "this_month": emi_load,
            "months_tracked": emi_months
        },

        "monthly_savings": monthly.to_dict(orient="records"),

        "category_summary": cat_summary.to_dict(orient="records"),

        "cleaned_csv": csv_output
    }

    return result
