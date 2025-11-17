import pandas as pd
import numpy as np
from io import BytesIO
import calendar
from datetime import datetime


# ----------------------------------------------------
# 0. FUTURE / PREDICTION BLOCK
# ----------------------------------------------------
def build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:
    """
    Light-weight prediction block based on current month progression
    and historic behaviour. No heavy ML yet, but shaped like it.
    """

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

    # 1. Identify current & previous months
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

    # 2. Project end-of-month savings
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

    # 3. Overspend risk vs safe_daily_spend / history
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
        prev_months = monthly_net[monthly_net["month"] < current_month]
        if not prev_months.empty:
            last3 = prev_months["signed_amount"].iloc[-3:]
            avg_net_last3 = last3.mean()
        else:
            avg_net_last3 = monthly_net["signed_amount"].iloc[-1]

        if avg_net_last3 == 0:
            ratio = 1.0
        else:
            ratio = max(
                0.0, (avg_net_last3 - predicted_eom_savings) / abs(avg_net_last3)
            )

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

    # 4. Risky categories (where you may overspend)
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


# ----------------------------------------------------
# 1. HELPER: LOAD WITH SMART HEADER DETECTION
# ----------------------------------------------------
def _load_with_header_detection(file_bytes, file_name: str) -> pd.DataFrame:
    """
    Handles messy bank exports (HDFC .xls etc.) which have junk rows
    before the real header. Detects the header row by scanning the first
    few rows for typical column names.
    """

    is_csv = file_name.lower().endswith(".csv")

    # Read raw with NO header
    if is_csv:
        raw = pd.read_csv(BytesIO(file_bytes), header=None)
    else:
        raw = pd.read_excel(BytesIO(file_bytes), header=None)

    header_row = None
    max_scan = min(20, len(raw))

    keywords = [
        "date",
        "txn",
        "transaction",
        "value date",
        "narration",
        "description",
        "details",
        "withdrawal",
        "deposit",
        "debit",
        "credit",
        "amount",
    ]

    for i in range(max_scan):
        row_values = raw.iloc[i].tolist()
        row_str = " ".join(str(x).lower() for x in row_values)
        if any(k in row_str for k in keywords):
            header_row = i
            break

    # Fallback: assume first row is header
    if header_row is None:
        header_row = 0

    # Re-read file with correct header row
    if is_csv:
        df = pd.read_csv(BytesIO(file_bytes), header=header_row)
    else:
        df = pd.read_excel(BytesIO(file_bytes), header=header_row)

    return df


# ----------------------------------------------------
# 2. MAIN ENTRY: ANALYZE STATEMENT
# ----------------------------------------------------
def analyze_statement(file_bytes, file_name):
    # 2.1 LOAD FILE (with header detection)
    df = _load_with_header_detection(file_bytes, file_name)

    # 2.2 STANDARDIZE COLUMNS
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # Initial column mapping
    col_map = {}
    for c in df.columns:
        if "date" in c and "value" not in c:
            col_map.setdefault("date", c)
        if (
            "desc" in c
            or "narration" in c
            or "particular" in c
            or "details" in c
            or "remark" in c
        ):
            col_map.setdefault("description", c)
        if "amount" in c:
            col_map.setdefault("amount", c)
        if "type" in c or "cr/dr" in c or "dr/cr" in c:
            col_map.setdefault("type", c)

    df = df.rename(columns=col_map)

    # If description still missing, best-effort fallback
    if "description" not in df.columns:
        for fallback in ["narration", "description", "details", "particulars", "remarks"]:
            if fallback in df.columns:
                df = df.rename(columns={fallback: "description"})
                break

    # 2.3 HDFC-style DEBIT / CREDIT COLUMNS -> single "amount"
    if "amount" not in df.columns:
        debit_candidates = [
            c
            for c in df.columns
            if any(
                key in c
                for key in [
                    "debit",
                    "withdrawal",
                    "withdr",
                    "wdl",
                    "dr",
                    "withdrawal amt",
                ]
            )
        ]
        credit_candidates = [
            c
            for c in df.columns
            if any(
                key in c
                for key in [
                    "credit",
                    "deposit",
                    "cr",
                    "cr.",
                    "deposit amt",
                ]
            )
        ]

        debit_col = debit_candidates[0] if debit_candidates else None
        credit_col = credit_candidates[0] if credit_candidates else None

        if debit_col or credit_col:
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

            # Positive for credit, negative for debit
            df["amount"] = credit_vals - debit_vals
            df["type"] = np.where(df["amount"] >= 0, "CR", "DR")

    # 2.4 Ensure existence of core fields
    if "date" not in df.columns:
        # FRONTEND expects this message
        raise Exception("Missing required date column.")

    if "description" not in df.columns:
        raise Exception("Missing required transaction description column.")

    if "amount" not in df.columns:
        raise Exception(
            "Missing required amount information. Looked for 'amount' or debit/credit columns."
        )

    # ------------------------------------------------
    # 3. CLEAN DATE + AMOUNT
    # ------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # ------------------------------------------------
    # 4. SIGNED AMOUNT
    # ------------------------------------------------
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper()

        def _signed(row):
            t = str(row.get("type", "")).upper()
            amt = row["amount"]
            if t in ("CR", "CREDIT", "C"):
                return amt
            elif t in ("DR", "DEBIT", "D"):
                return -abs(amt)
            return amt

        df["signed_amount"] = df.apply(_signed, axis=1)
    else:
        df["signed_amount"] = df["amount"]

    # Drop obviously invalid rows
    df = df.dropna(subset=["date", "signed_amount"])
    df = df.sort_values("date")

    # ------------------------------------------------
    # 5. BASIC CATEGORY MAPPING (RULE-BASED)
    # ------------------------------------------------
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

    df["category"] = df.get("description", "").astype(str).apply(detect_category)

    # ------------------------------------------------
    # 6. AGGREGATES & METRICS
    # ------------------------------------------------
    df["month"] = df["date"].dt.to_period("M")

    # Total inflow/outflow
    inflow = df[df["signed_amount"] > 0]["signed_amount"].sum()
    outflow = df[df["signed_amount"] < 0]["signed_amount"].abs().sum()
    net_savings = inflow - outflow

    # Monthly savings
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

    # UPI outflow
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

    # EMI load
    emi_df = df[df["category"] == "Emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = emi_month_df["signed_amount"].abs().sum()
    emi_months = emi_df["month"].nunique()

    # Category summary (for donut chart)
    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
    ).sort_values("signed_amount", ascending=False)

    # Safe daily spend (simple version)
    safe_daily = max(0.0, this_month_savings) / 30

    # ------------------------------------------------
    # 7. FUTURE / PREDICTION BLOCK
    # ------------------------------------------------
    future_block = build_future_block(df, safe_daily)

    # ------------------------------------------------
    # 8. CLEANED CSV EXPORT
    # ------------------------------------------------
    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    # ------------------------------------------------
    # 9. BUILD RESULT JSON
    # ------------------------------------------------
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
