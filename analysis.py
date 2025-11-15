import io
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# ============================
#       UTIL FUNCTIONS
# ============================

def _read_statement(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame."""
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Find a column by matching lowercase names."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    col = _find_column(df, ["Date", "Txn Date", "Transaction Date"])
    if col is None:
        raise ValueError("Date column not found. Expected: Date / Txn Date / Transaction Date.")

    df["Date"] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).copy()
    return df


def _ensure_description(df: pd.DataFrame) -> pd.DataFrame:
    col = _find_column(df, ["Description", "Narration", "Transaction Details"])
    if col is None:
        raise ValueError("Description column not found.")
    df["Description"] = df[col].astype(str)
    return df


# ============================
#   SIGNED AMOUNT LOGIC
# ============================

def _build_signed_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SignedAmount:
    + positive = credit
    + negative = debit
    Priority order:
    1. Amount + Type (CR/DR)
    2. Credit + Debit columns
    3. Existing SignedAmount
    """

    # ---- Option A: Amount + Type (CR/DR) ----
    amt_col = _find_column(df, ["Amount", "Amt"])
    type_col = _find_column(df, ["Type", "DR/CR", "CR/DR"])

    if amt_col is not None and type_col is not None:
        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
        df = df.dropna(subset=[amt_col]).copy()

        types = df[type_col].astype(str).str.lower().str.strip()
        credit_markers = {"cr", "credit", "c", "in"}

        df["SignedAmount"] = np.where(
            types.isin(credit_markers),
            df[amt_col],
            -abs(df[amt_col]),
        )
        return df

    # ---- Option B: Credit & Debit columns ----
    credit_col = _find_column(df, ["Credit", "Cr", "Cr Amount"])
    debit_col = _find_column(df, ["Debit", "Dr", "Dr Amount"])

    if credit_col is not None and debit_col is not None:
        df[credit_col] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        df[debit_col] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0)
        df["SignedAmount"] = df[credit_col] - df[debit_col]
        return df

    # ---- Option C: Already has SignedAmount ----
    if "SignedAmount" in df.columns:
        df["SignedAmount"] = pd.to_numeric(df["SignedAmount"], errors="coerce")
        df = df.dropna(subset=["SignedAmount"]).copy()
        return df

    raise ValueError("Could not compute SignedAmount. Upload contains unsupported columns.")


# ============================
#     CATEGORY DETECTION
# ============================

def _categorise(desc: str) -> str:
    d = desc.lower()

    if "salary" in d: return "Salary"
    if "interest" in d: return "Interest"
    if "rent" in d and "credit" not in d: return "Rent"
    if "emi" in d or "loan" in d: return "EMI"
    if "petrol" in d or "hpcl" in d or "bpcl" in d or "uber" in d or "ola" in d: return "Fuel & Transport"
    if "airtel" in d or "jio" in d or "vi " in d or "internet" in d or "recharge" in d: return "Mobile & Internet"
    if "netflix" in d or "prime video" in d or "hotstar" in d or "subscription" in d: return "Subscriptions"
    if "swiggy" in d or "zomato" in d or "lunch" in d or "dinner" in d or "breakfast" in d: return "Food & Dining"
    if "amazon" in d or "flipkart" in d or "myntra" in d: return "Shopping"
    if "school fee" in d or "school fees" in d or "tuition" in d: return "Education"
    if "upi" in d: return "UPI / Wallet"
    return "Others"


# ============================
#   UPI COUNTERPARTY FIX
# ============================

def _upi_counterparty(desc: str) -> str:
    """
    Extract counterparty from ANY UPI format:
    - Upi_Zomato Lunch
    - UPI-Petrol HPCL
    - BHIMUPI*1234*
    - GPay_UPI Zomato
    """
    d = desc.lower()

    if "upi" not in d:
        return desc[:40]

    # Very flexible split pattern
    parts = re.split(r"(?i)upi[\W_]*", desc, maxsplit=1)

    if len(parts) == 2:
        after = parts[1].strip()
        words = after.split()
        if words:
            return " ".join(words[:3]).title()

    return desc[:40]


# ============================
#      MAIN ANALYSIS
# ============================

def analyze_statement(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    df = _read_statement(file_bytes, filename)
    df = _ensure_date(df)
    df = _ensure_description(df)
    df = _build_signed_amount(df)

    df["Category"] = df["Description"].apply(_categorise)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # ===================================
    #        SUMMARY NUMBERS
    # ===================================
    inflow = df.loc[df["SignedAmount"] > 0, "SignedAmount"].sum()
    outflow = -df.loc[df["SignedAmount"] < 0, "SignedAmount"].sum()
    savings_total = df["SignedAmount"].sum()

    monthly = (
        df.groupby("Month")
        .agg(
            TotalInflow=("SignedAmount", lambda s: s[s > 0].sum()),
            TotalOutflow=("SignedAmount", lambda s: -s[s < 0].sum()),
        )
        .reset_index()
    )
    monthly["Savings"] = monthly["TotalInflow"] - monthly["TotalOutflow"]

    monthly_records = [
        {
            "Month": row["Month"],
            "TotalInflow": float(row["TotalInflow"]),
            "TotalOutflow": float(row["TotalOutflow"]),
            "Savings": float(row["Savings"]),
        }
        for _, row in monthly.iterrows()
    ]

    current_month = None
    if not df.empty:
        current_month = df["Date"].max().to_period("M").strftime("%Y-%m")

    # Growth text
    if len(monthly_records) >= 2:
        last = monthly_records[-1]["Savings"]
        prev = monthly_records[-2]["Savings"]
        base = prev if prev != 0 else 1
        pct = (last - prev) / abs(base) * 100
        sign = "+" if pct >= 0 else ""
        growth_text = f"{sign}{pct:.0f}% vs last month"
    else:
        growth_text = "First month tracked"

    safe_daily_spend = max((savings_total / max(len(monthly_records), 1)) / 30, 0)

    # ===================================
    #         CATEGORY SPEND
    # ===================================
    spend = df[df["SignedAmount"] < 0]
    cat = (
        spend.groupby("Category", as_index=False)["SignedAmount"]
        .sum()
    )
    cat["TotalAmount"] = cat["SignedAmount"].abs()
    categories = [
        {"Category": r["Category"], "TotalAmount": float(r["TotalAmount"])}
        for _, r in cat.iterrows()
    ]

    # ===================================
    #        UPI ANALYSIS (FIXED)
    # ===================================
    upi_mask = df["Description"].str.contains(r"(?i)upi", regex=True, na=False)
    upi_df = df[upi_mask].copy()

    upi_outflow_month = 0.0
    upi_tops = []

    if not upi_df.empty:
        upi_df["UPILabel"] = upi_df["Description"].apply(_upi_counterparty)
        upi_df["Month"] = upi_df["Date"].dt.to_period("M").astype(str)

        if current_month:
            cur = upi_df[(upi_df["Month"] == current_month) & (upi_df["SignedAmount"] < 0)]
            upi_outflow_month = float(-cur["SignedAmount"].sum())

        top = (
            upi_df[upi_df["SignedAmount"] < 0]
            .groupby("UPILabel", as_index=False)["SignedAmount"]
            .sum()
        )
        top["TotalAmount"] = top["SignedAmount"].abs()
        top = top.sort_values("TotalAmount", ascending=False).head(5)

        for _, row in top.iterrows():
            upi_tops.append({
                "Description": row["UPILabel"],
                "TotalAmount": float(row["TotalAmount"]),
            })

    # ===================================
    #          EMI ANALYSIS
    # ===================================
    emi_df = df[df["Description"].str.contains("emi", case=False, na=False)]
    emi_df = emi_df[emi_df["SignedAmount"] < 0].copy()

    emi_load = 0.0
    emi_by_desc = []
    emi_by_month = []

    if not emi_df.empty:
        # By description
        d = (
            emi_df.groupby("Description", as_index=False)["SignedAmount"]
            .sum()
        )
        d["TotalEMI"] = d["SignedAmount"].abs()
        emi_by_desc = [
            {"Description": r["Description"], "TotalEMI": float(r["TotalEMI"])}
            for _, r in d.iterrows()
        ]

        # By month
        emi_df["Month"] = emi_df["Date"].dt.to_period("M").astype(str)
        m = (
            emi_df.groupby("Month", as_index=False)["SignedAmount"]
            .sum()
        )
        m["TotalEMI"] = m["SignedAmount"].abs()

        emi_by_month = [
            {"Month": r["Month"], "TotalEMI": float(r["TotalEMI"])}
            for _, r in m.iterrows()
        ]

        if current_month:
            row = m[m["Month"] == current_month]
            if not row.empty:
                emi_load = float(row["TotalEMI"].iloc[0])

    # ===================================
    #         FORECAST LOGIC
    # ===================================
    forecast = {"available": False}
    if len(monthly_records) >= 3:
        last3 = [m["Savings"] for m in monthly_records[-3:]]
        forecast.update({
            "available": True,
            "next_month": float(sum(last3) / 3),
            "history": [{"month": m["Month"], "savings": m["Savings"]} for m in monthly_records]
        })

    # ===================================
    #       CLEANED EXPORT
    # ===================================
    cleaned = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned["Date"] = cleaned["Date"].dt.strftime("%Y-%m-%d")
    cleaned_preview = cleaned.sort_values("Date", ascending=False).head(100)
    cleaned_csv_bytes = cleaned.to_csv(index=False).encode("utf-8")

    # ===================================
    #        FINAL RESULT DICT
    # ===================================
    summary = {
        "total_inflow": float(inflow),
        "total_outflow": float(outflow),
        "savings_total": float(savings_total),
        "current_month": current_month,
        "current_month_savings": float(monthly_records[-1]["Savings"] if monthly_records else 0),
        "growth_text": growth_text,
        "upi_net_outflow": float(upi_outflow_month),
        "emi_load": float(emi_load),
        "safe_daily_spend": float(safe_daily_spend),
    }

    return {
        "summary": summary,
        "monthly": monthly_records,
        "categories": categories,
        "upi": {"top_counterparties": upi_tops},
        "emi": {"by_desc": emi_by_desc, "by_month": emi_by_month},
        "forecast": forecast,
        "cleaned_preview": cleaned_preview.to_dict(orient="records"),
        "cleaned_csv": cleaned_csv_bytes,
    }
