import pandas as pd
import numpy as np
from io import BytesIO
import calendar
from datetime import datetime
import pdfplumber


def load_pdf_to_df(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse HDFC-style transaction PDF into a DataFrame with columns:
    - date (string, DD/MM/YYYY)
    - description (transaction remarks)
    - amount (deposit positive, withdrawal negative)
    This is designed for PDFs like OpTransactionHistory*.pdf you uploaded.
    """
    rows = []

    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for t in tables:
                if not t:
                    continue

                # ---- 1) Find the header row (with transaction date / withdrawal / deposit) ----
                header_idx = None
                header = None
                for i, row in enumerate(t):
                    if not row:
                        continue
                    joined = " ".join(str(c) for c in row if c)
                    low = joined.lower()
                    if (
                        "transaction date" in low
                        and "withdrawal" in low
                        and "deposit" in low
                    ):
                        header_idx = i
                        header = row
                        break

                if header_idx is None or header is None:
                    continue

                header_lower = [str(c).lower() if c else "" for c in header]

                def find_idx(keywords):
                    for idx, cell in enumerate(header_lower):
                        for k in keywords:
                            if k in cell:
                                return idx
                    return None

                idx_tran_date = find_idx(["transaction date"])
                idx_remarks = find_idx(["transaction remarks", "remarks"])
                idx_withdraw = find_idx(["withdrawal"])
                idx_deposit = find_idx(["deposit"])

                # Need at least date + (withdraw or deposit)
                if idx_tran_date is None or (idx_withdraw is None and idx_deposit is None):
                    continue

                # ---- 2) Parse all rows after header as transactions ----
                for r in t[header_idx + 1 :]:
                    if not r:
                        continue
                    if idx_tran_date >= len(r):
                        continue

                    date_cell = r[idx_tran_date]
                    if not date_cell:
                        continue

                    ds = str(date_cell).strip()

                    # Make sure this really looks like a date
                    try:
                        pd.to_datetime(ds, dayfirst=True, errors="raise")
                    except Exception:
                        continue

                    # Description / remarks
                    remarks = ""
                    if idx_remarks is not None and idx_remarks < len(r):
                        remarks = r[idx_remarks] or ""

                    # Withdrawal / deposit -> numeric
                    def to_num(x):
                        x = str(x).replace(",", "").strip()
                        if x in ("", "None", "-", "0", "0.00"):
                            return 0.0
                        try:
                            return float(x)
                        except Exception:
                            return 0.0

                    wd = 0.0
                    if idx_withdraw is not None and idx_withdraw < len(r):
                        wd = to_num(r[idx_withdraw])

                    dep = 0.0
                    if idx_deposit is not None and idx_deposit < len(r):
                        dep = to_num(r[idx_deposit])

                    amount = dep - wd  # deposits positive, withdrawals negative

                    rows.append(
                        {
                            "date": ds,
                            "description": str(remarks).replace("\n", " ").strip(),
                            "amount": amount,
                        }
                    )

    if not rows:
        raise Exception(
            "Could not detect transaction table in PDF. "
            "Try uploading the bank's CSV or Excel export instead."
        )

    return pd.DataFrame(rows)


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
    monthly_net = (
        df.groupby("month")["signed_amount"]
        .sum()
        .reset_index()
        .sort_values("month")
    )

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
    # 4. Risky categories
    # -----------------------------
    if df_prev.empty:
        risky_categories = []
    else:
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
    lower = (file_name or "").lower()
    if lower.endswith(".pdf"):
        df = load_pdf_to_df(file_bytes)
    elif lower.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_excel(BytesIO(file_bytes))

    # -----------------------------
    # 2. STANDARDIZE COLUMNS
    # -----------------------------
    df.columns = df.columns.str.strip().str.lower()

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

    if "date" not in df or "description" not in df or "amount" not in df:
        raise Exception("Missing required columns (Date, Description, Amount).")

    # -----------------------------
    # 3. CLEAN DATE + AMOUNT
    # -----------------------------
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

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
            axis=1,
        )
    else:
        # If no type column exists (like our PDF path), use sign in amount
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

    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = df[df["signed_amount"] < 0]["signed_amount"].abs().sum()
    net_savings = inflow - outflow

    # -----------------------------
    # 6. MONTHLY SAVINGS
    # -----------------------------
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
        prev_savings = 0
        mom_change = 0

    # -----------------------------
    # 7. UPI OUTFLOW
    # -----------------------------
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
        "future_block": future_block,
    }

    return result
