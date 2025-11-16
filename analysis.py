import io
from datetime import datetime
from calendar import monthrange
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS – allow your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    col_map = {}

    for c in df.columns:
        if "date" in c and "value" not in c and "statement" not in c:
            col_map["date"] = c
        if "desc" in c or "narration" in c or "details" in c:
            col_map["description"] = c
        if "amount" in c and "balance" not in c:
            col_map["amount"] = c
        if "type" in c or "dr/cr" in c or "debit/credit" in c:
            col_map["type"] = c
        if "category" in c:
            col_map["category"] = c

    df = df.rename(columns=col_map)

    if "date" not in df or "description" not in df or "amount" not in df:
        raise HTTPException(
            status_code=400,
            detail="Missing required columns. Need something like Date, Description, Amount.",
        )

    # parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # clean description
    df["description"] = df["description"].astype(str)

    # clean amount
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", "0")
        .astype(float)
    )

    # handle type if present (CR/DR convention)
    if "type" in df.columns:
        t = df["type"].astype(str).str.lower()
        # if looks like CR/DR type, adjust sign
        mask_dr = t.str.contains("dr")
        mask_cr = t.str.contains("cr")
        # for DR, make amount negative; for CR, positive
        df.loc[mask_dr, "amount"] = -df.loc[mask_dr, "amount"].abs()
        df.loc[mask_cr, "amount"] = df.loc[mask_cr, "amount"].abs()

    return df


def extract_month_info(df: pd.DataFrame):
    df = df.copy()
    df["month_period"] = df["date"].dt.to_period("M")
    df["month_str"] = df["month_period"].astype(str)

    last_date = df["date"].max()
    current_period = last_date.to_period("M")
    current_month_str = str(current_period)

    # previous month
    prev_period = current_period - 1
    prev_month_str = str(prev_period)

    return df, current_period, current_month_str, prev_month_str, last_date


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    df, current_period, current_month_str, prev_month_str, last_date = extract_month_info(
        df
    )

    # inflow/outflow overall
    inflow = df.loc[df["amount"] > 0, "amount"].sum()
    outflow = -df.loc[df["amount"] < 0, "amount"].sum()  # make positive
    net_savings = inflow - outflow

    # monthly savings = inflow - outflow per month (using signed amount)
    monthly_group = df.groupby("month_str")["amount"].sum().reset_index()
    monthly_group = monthly_group.sort_values("month_str")

    # this month + prev
    this_row = monthly_group[monthly_group["month_str"] == current_month_str]
    prev_row = monthly_group[monthly_group["month_str"] == prev_month_str]

    this_savings = float(this_row["amount"].iloc[0]) if not this_row.empty else 0.0
    prev_savings = float(prev_row["amount"].iloc[0]) if not prev_row.empty else 0.0

    if prev_savings == 0:
        mom_change = 0.0
    else:
        mom_change = (this_savings - prev_savings) / abs(prev_savings) * 100

    # safe daily spend: assume you can safely spend income - 20% savings target
    current_month_df = df[df["month_str"] == current_month_str]
    month_income = current_month_df.loc[current_month_df["amount"] > 0, "amount"].sum()

    year = last_date.year
    month = last_date.month
    days_in_month = monthrange(year, month)[1]

    target_savings = 0.2 * month_income
    safe_spend_budget = max(month_income - target_savings, 0.0)
    safe_daily_spend = safe_spend_budget / days_in_month if days_in_month > 0 else 0.0

    return {
        "summary": {
            "inflow": float(round(inflow)),
            "outflow": float(round(outflow)),
            "net_savings": float(round(net_savings)),
            "this_month": {
                "month": last_date.strftime("%b %Y"),
                "savings": float(round(this_savings)),
                "prev_savings": float(round(prev_savings)),
                "mom_change": float(round(mom_change)),
            },
            "safe_daily_spend": float(round(safe_daily_spend)),
        },
        "monthly_group": monthly_group,
        "current_month_str": current_month_str,
        "prev_month_str": prev_month_str,
        "last_date": last_date,
        "days_in_month": days_in_month,
    }


def compute_upi_info(df: pd.DataFrame, current_month_str: str) -> Dict[str, Any]:
    df = df.copy()
    df["month_str"] = df["date"].dt.to_period("M").astype(str)

    # identify UPI-like transactions
    upi_mask = df["description"].str.contains("upi|vpa|@paytm|@oksbi|@okicici|@okaxis", case=False, na=False)
    upi_df = df[upi_mask]

    this_month = upi_df[upi_df["month_str"] == current_month_str]

    # outflow via UPI = negative amounts
    total_upi = -upi_df.loc[upi_df["amount"] < 0, "amount"].sum()
    this_month_upi = -this_month.loc[this_month["amount"] < 0, "amount"].sum()

    # extract handles
    def extract_handle(desc: str) -> str:
        import re

        m = re.findall(r"([\w\.\-]+@\w+)", desc)
        return m[0].lower() if m else ""

    if not upi_df.empty:
        upi_df["handle"] = upi_df["description"].apply(extract_handle)
        handles = upi_df["handle"]
        handles = handles[handles != ""]
        if not handles.empty:
            top_handle = handles.value_counts().idxmax()
        else:
            top_handle = None
    else:
        top_handle = None

    return {
        "this_month": float(round(this_month_upi)),
        "top_handle": top_handle,
        "total_upi": float(round(total_upi)),
    }


def compute_emi_info(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["month_str"] = df["date"].dt.to_period("M").astype(str)

    # crude EMI detection
    emi_mask = df["description"].str.contains(
        "emi|loan|instalment|installment", case=False, na=False
    )
    emi_df = df[emi_mask & (df["amount"] < 0)]

    if emi_df.empty:
        return {"this_month": 0.0, "months_tracked": 0}

    last_month_str = str(df["date"].max().to_period("M"))
    this_month_emi = -emi_df.loc[emi_df["month_str"] == last_month_str, "amount"].sum()

    months_tracked = emi_df["month_str"].nunique()

    return {
        "this_month": float(round(this_month_emi)),
        "months_tracked": int(months_tracked),
    }


def compute_monthly_savings_series(monthly_group: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for _, row in monthly_group.iterrows():
        out.append(
            {
                "month": row["month_str"],
                "signed_amount": float(round(row["amount"])),
            }
        )
    return out


def compute_category_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.copy()

    # if we already have category, use it; else one bucket
    if "category" in df.columns:
        cat_group = (
            df.groupby("category")["amount"].sum().reset_index().sort_values(
                "amount"
            )
        )
        cat_group["signed_amount"] = -cat_group["amount"]  # positive spend
        cat_group = cat_group[cat_group["signed_amount"] > 0]
        rows = []
        for _, row in cat_group.iterrows():
            rows.append(
                {
                    "category": str(row["category"]),
                    "signed_amount": float(round(row["signed_amount"])),
                }
            )
        return rows

    # fallback: single "Spend" category = all negative amounts
    total_spend = -df.loc[df["amount"] < 0, "amount"].sum()
    if total_spend <= 0:
        return []
    return [
        {
            "category": "Spends",
            "signed_amount": float(round(total_spend)),
        }
    ]


# -----------------------------
# Future / risk logic
# -----------------------------


def compute_overspend_risk(
    safe_monthly_limit: float, projected_month_spend: float
) -> Dict[str, Any]:
    """
    Turn safe vs projected spend into:
    - risk level
    - probability 0–1
    """
    if safe_monthly_limit <= 0:
        return {"level": "medium", "probability": 0.5}

    ratio = projected_month_spend / safe_monthly_limit  # >1 means overshoot

    # discrete bucket
    if ratio <= 0.9:
        level = "low"
    elif ratio <= 1.1:
        level = "medium"
    else:
        level = "high"

    # smooth probability: 0.7x safe -> 0, 1.4x safe -> 1
    prob = (ratio - 0.7) / (1.4 - 0.7)
    prob = max(0.0, min(1.0, prob))

    return {"level": level, "probability": prob}


def compute_risky_categories(
    df: pd.DataFrame,
    current_month_str: str,
    days_in_month: int,
) -> List[Dict[str, Any]]:
    """
    Compare current month's pace vs baseline average across past months per category.
    Requires a 'category' column; if not present, returns [].
    """
    if "category" not in df.columns:
        return []

    df = df.copy()
    df["month_str"] = df["date"].dt.to_period("M").astype(str)

    current_df = df[df["month_str"] == current_month_str]
    past_df = df[df["month_str"] != current_month_str]

    if current_df.empty or past_df.empty:
        return []

    last_date = current_df["date"].max()
    days_elapsed = last_date.day
    if days_elapsed <= 0:
        return []

    # only consider spend (negative amounts)
    current_spends = (
        current_df[current_df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .reset_index()
    )
    current_spends["current_spend"] = -current_spends["amount"]  # positive

    past_spends = (
        past_df[past_df["amount"] < 0]
        .groupby(["month_str", "category"])["amount"]
        .sum()
        .reset_index()
    )
    if past_spends.empty:
        return []

    past_spends["spend_abs"] = -past_spends["amount"]
    baseline = (
        past_spends.groupby("category")["spend_abs"].mean().reset_index()
    )  # avg per month

    merged = current_spends.merge(baseline, on="category", how="left")
    merged = merged.rename(columns={"spend_abs": "baseline_monthly"})

    rows = []
    for _, row in merged.iterrows():
        cat = str(row["category"])
        current_spend = float(row["current_spend"])
        baseline_monthly = float(row.get("baseline_monthly", 0.0))

        # project for full month
        projected_monthly = current_spend * days_in_month / days_elapsed

        if baseline_monthly <= 0:
            continue

        # mark risky if projected > 120% of baseline
        if projected_monthly > 1.2 * baseline_monthly:
            rows.append(
                {
                    "name": cat,
                    "projected_amount": float(round(projected_monthly)),
                    "baseline_amount": float(round(baseline_monthly)),
                }
            )

    # sort by how bad it is
    rows = sorted(
        rows,
        key=lambda r: r["projected_amount"] - r["baseline_amount"],
        reverse=True,
    )

    return rows[:5]


def build_future_block(
    df: pd.DataFrame,
    summary: Dict[str, Any],
    current_month_str: str,
    days_in_month: int,
) -> Dict[str, Any]:
    df = df.copy()
    df["month_str"] = df["date"].dt.to_period("M").astype(str)

    current_df = df[df["month_str"] == current_month_str]
    if current_df.empty:
        # nothing to predict
        return {
            "predicted_eom_savings": 0.0,
            "predicted_eom_range": [0.0, 0.0],
            "overspend_risk": {"level": "medium", "probability": 0.5},
            "risky_categories": [],
            "diagnostics": {
                "safe_monthly_limit": 0.0,
                "projected_month_spend": 0.0,
                "current_month_spend": 0.0,
            },
        }

    last_date = current_df["date"].max()
    days_elapsed = last_date.day
    if days_elapsed <= 0:
        days_elapsed = 1

    # current spend (outflow) this month
    current_month_spend = -current_df.loc[current_df["amount"] < 0, "amount"].sum()
    # current income this month
    current_month_income = current_df.loc[current_df["amount"] > 0, "amount"].sum()

    # project spend for entire month from pace
    projected_month_spend = current_month_spend * days_in_month / days_elapsed

    # safe monthly limit from summary.safe_daily_spend
    safe_daily = summary.get("safe_daily_spend", 0.0) or 0.0
    safe_monthly_limit = safe_daily * days_in_month

    # predicted month-end savings = income - projected spend (simplified)
    predicted_eom_savings = current_month_income - projected_month_spend

    # create a fuzzy ±20% range around prediction
    low = predicted_eom_savings * 0.8
    high = predicted_eom_savings * 1.2

    overspend_risk = compute_overspend_risk(
        safe_monthly_limit=safe_monthly_limit,
        projected_month_spend=projected_month_spend,
    )

    risky_categories = compute_risky_categories(
        df=df,
        current_month_str=current_month_str,
        days_in_month=days_in_month,
    )

    future_block = {
        "predicted_eom_savings": float(round(predicted_eom_savings)),
        "predicted_eom_range": [
            float(round(low)),
            float(round(high)),
        ],
        "overspend_risk": overspend_risk,
        "risky_categories": risky_categories,
        "diagnostics": {
            "safe_monthly_limit": float(round(safe_monthly_limit)),
            "projected_month_spend": float(round(projected_month_spend)),
            "current_month_spend": float(round(current_month_spend)),
        },
    }

    return future_block


# -----------------------------
# Main analysis
# -----------------------------


def analyze_statement(file_bytes: bytes, file_name: str) -> Dict[str, Any]:
    # 1. Load
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))

    # 2. Normalize columns
    df = normalize_columns(df)

    # 3. Summary
    summary_info = compute_summary(df)
    summary = summary_info["summary"]
    monthly_group = summary_info["monthly_group"]
    current_month_str = summary_info["current_month_str"]
    days_in_month = summary_info["days_in_month"]

    # 4. UPI / EMI / monthly savings / categories
    upi_info = compute_upi_info(df, current_month_str=current_month_str)
    emi_info = compute_emi_info(df)
    monthly_savings = compute_monthly_savings_series(monthly_group)
    category_summary = compute_category_summary(df)

    # 5. Future / risk block
    future_block = build_future_block(
        df=df,
        summary=summary,
        current_month_str=current_month_str,
        days_in_month=days_in_month,
    )

    # 6. Cleaned CSV output
    cleaned_csv = df.to_csv(index=False)

    result = {
        "summary": summary,
        "upi": upi_info,
        "emi": emi_info,
        "monthly_savings": monthly_savings,
        "category_summary": category_summary,
        "cleaned_csv": cleaned_csv,
        "future_block": future_block,
    }

    return result


# -----------------------------
# FastAPI route
# -----------------------------


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = analyze_statement(contents, file.filename)
        return data
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
