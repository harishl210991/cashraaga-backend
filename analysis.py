import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import calendar
from datetime import datetime


# ======================================================
# 1. LOADING + HEADER DETECTION
# ======================================================

def _load_raw_table(file_bytes, file_name):
    """
    Load CSV / Excel into a raw dataframe with header=None and dtype=str.
    Handles fully-quoted CSV lines like in dummy_bank - Copy.csv.
    """
    name = file_name.lower()

    if name.endswith(".csv"):
        # Decode bytes to text
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # Detect "every line fully in double quotes" pattern
        sample = lines[:20]
        quoted = 0
        for ln in sample:
            s = ln.strip()
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                quoted += 1

        # If majority of the sample lines are fully quoted, strip outer quotes
        if sample and quoted / len(sample) > 0.6:
            lines = [
                ln.strip()[1:-1]
                if ln.strip().startswith('"') and ln.strip().endswith('"')
                else ln
                for ln in lines
            ]
            text = "\n".join(lines)

        # Read as raw, header=None so we can detect the real header row
        df_raw = pd.read_csv(StringIO(text), header=None, dtype=str)

    else:
        # Excel (.xlsx or .xls) – load everything as raw text, header=None
        df_raw = pd.read_excel(BytesIO(file_bytes), header=None, dtype=str)

    return df_raw


def _guess_header_row(df_raw: pd.DataFrame) -> int:
    """
    Heuristic to find the row that looks like the actual header:
    - contains "date"
    - and/or something like desc/narration/details/particular
    - and/or amount/balance/debit/credit
    """
    best_idx = 0
    best_score = -1
    max_rows = min(30, len(df_raw))

    for i in range(max_rows):
        row = df_raw.iloc[i].fillna("").astype(str).str.strip().str.lower()
        non_empty = [c for c in row if c not in ("", "nan", "none")]
        if not non_empty:
            continue

        score = 0
        cells = list(row)

        if any("date" in c for c in cells):
            score += 3

        if any(
            any(k in c for k in ("desc", "narrat", "details", "particular", "remark"))
            for c in cells
        ):
            score += 2

        if any(
            any(
                k in c
                for k in (
                    "amount",
                    "amt",
                    "balance",
                    "debit",
                    "credit",
                    "withdraw",
                    "deposit",
                )
            )
            for c in cells
        ):
            score += 2

        if len(non_empty) >= 2:
            score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _normalize_table(file_bytes, file_name) -> pd.DataFrame:
    """
    Returns a dataframe with the correct header row applied and
    all-empty columns dropped.
    """
    df_raw = _load_raw_table(file_bytes, file_name)
    header_row = _guess_header_row(df_raw)

    header = df_raw.iloc[header_row].fillna("").astype(str).str.strip()
    df = df_raw.iloc[header_row + 1 :].reset_index(drop=True).copy()
    df.columns = header
    df = df.dropna(axis=1, how="all")

    return df


def _find_col(df: pd.DataFrame, keywords) -> str | None:
    """
    Find first column whose lowercased name contains any of the given keywords.
    """
    for c in df.columns:
        name = str(c).strip().lower()
        for k in keywords:
            if k in name:
                return c
    return None


# ======================================================
# 2. FUTURE / PREDICTION BLOCK (same behaviour as before)
# ======================================================

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
    outflow_to_date = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()

    avg_daily_inflow = inflow_to_date / max(days_elapsed, 1)
    avg_daily_outflow = outflow_to_date / max(days_elapsed, 1)

    projected_inflow = avg_daily_inflow * days_in_month
    projected_outflow = avg_daily_outflow * days_in_month

    predicted_eom_savings = float(projected_inflow - projected_outflow)
    low_bound = float(predicted_eom_savings * 0.85)
    high_bound = float(predicted_eom_savings * 1.15)

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


# ======================================================
# 3. MAIN ANALYSIS
# ======================================================

def analyze_statement(file_bytes, file_name):
    # --- 1) Normalize table & detect key columns ---
    df = _normalize_table(file_bytes, file_name)

    # standardize header
    df.columns = df.columns.map(lambda c: str(c).strip().lower())

    date_col = _find_col(df, ["date"])
    desc_col = _find_col(df, ["desc", "narrat", "details", "particular"])
    amount_col = _find_col(df, ["amount", "amt"])
    type_col = _find_col(df, ["type", "dr/cr", "cr/dr"])

    if date_col is None:
        raise Exception("Missing required date column.")
    if desc_col is None:
        raise Exception("Missing required description / narration column.")
    if amount_col is None:
        raise Exception("Missing required amount column.")

    col_ren = {
        date_col: "date",
        desc_col: "description",
        amount_col: "amount",
    }
    if type_col is not None:
        col_ren[type_col] = "type"

    df = df.rename(columns=col_ren)

    # --- 2) Clean date & amount ---
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
            lambda row: row["amount"] if row["type"] == "CR" else -abs(row["amount"]),
            axis=1,
        )
    else:
        df["signed_amount"] = df["amount"]

    df = df.dropna(subset=["date", "signed_amount"])
    df = df.sort_values("date")

    # --- 3) Simple category tagging (rule-based) ---
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

    # --- 4) Aggregates ---
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
        prev_savings = 0
        mom_change = 0

    # --- 5) UPI metrics ---
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

    # --- 6) EMI load ---
    emi_df = df[df["category"] == "Emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]

    emi_load = emi_month_df["signed_amount"].abs().sum()
    emi_months = emi_df["month"].nunique()

    # --- 7) Category summary for donut ---
    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
    ).sort_values("signed_amount", ascending=False)

    # --- 8) Safe daily spend ---
    safe_daily = max(0, this_month_savings) / 30

    # --- 9) Future / prediction block ---
    future_block = build_future_block(df, safe_daily)

    # --- 10) Cleaned CSV export ---
    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    # --- 11) Final JSON ---
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
