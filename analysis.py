import io
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# =========================
#   LOW-LEVEL HELPERS
# =========================

def _read_statement(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV / XLSX into a DataFrame."""
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)

    # Normalise names (strip only – keep original case for display)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Find a column by case-insensitive name from a list of candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Date column exists and is parsed."""
    date_col = _find_column(df, [
        "Date",
        "Txn Date",
        "Transaction Date",
        "Transaction_Date",
    ])
    if date_col is None:
        raise ValueError(
            "Could not find a Date column. "
            "Expected one of: Date, Txn Date, Transaction Date."
        )

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).copy()
    return df


def _ensure_description(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Description column exists."""
    desc_col = _find_column(df, [
        "Description",
        "Narration",
        "Transaction Details",
        "Particulars",
    ])
    if desc_col is None:
        raise ValueError(
            "Could not find a Description/Narration column."
        )
    df["Description"] = df[desc_col].astype(str)
    return df


def _build_signed_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SignedAmount:
    +ve = credit, -ve = debit.

    Priority:
    1) If Amount + Type (CR/DR) exist -> ALWAYS use those.
    2) Else if Credit + Debit exist -> Credit - Debit.
    3) Else if SignedAmount already exists -> trust it.
    """

    # ---- 1) Amount + Type (CR/DR) ----
    amount_col = _find_column(df, ["Amount", "Amt", "Transaction Amount"])
    type_col = _find_column(df, ["Type", "DR/CR", "CR/DR"])

    if amount_col is not None and type_col is not None:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        df = df.dropna(subset=[amount_col]).copy()

        types_lower = df[type_col].astype(str).str.strip().str.lower()
        credit_markers = {"cr", "credit", "c", "in"}

        df["SignedAmount"] = np.where(
            types_lower.isin(credit_markers),
            df[amount_col],          # credit => +ve
            -df[amount_col],         # debit  => -ve
        )

        # Safety: if somehow everything came out >= 0, flip DR explicitly
        if (df["SignedAmount"] < 0).sum() == 0:
            df["SignedAmount"] = np.where(
                types_lower.isin(credit_markers),
                df[amount_col],
                -abs(df[amount_col]),
            )
        return df

    # ---- 2) Separate Credit / Debit columns ----
    credit_col = _find_column(df, ["Credit", "Cr Amount", "Cr"])
    debit_col = _find_column(df, ["Debit", "Dr Amount", "Dr"])

    if credit_col is not None and debit_col is not None:
        df[credit_col] = pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        df[debit_col] = pd.to_numeric(df[debit_col], errors="coerce").fillna(0)
        df["SignedAmount"] = df[credit_col] - df[debit_col]
        return df

    # ---- 3) Already has SignedAmount ----
    if "SignedAmount" in df.columns:
        df["SignedAmount"] = pd.to_numeric(df["SignedAmount"], errors="coerce")
        df = df.dropna(subset=["SignedAmount"]).copy()
        return df

    raise ValueError(
        "Could not infer amounts. Expected either:\n"
        "- 'Amount' + 'Type' (CR/DR), OR\n"
        "- separate Credit / Debit columns, OR\n"
        "- existing SignedAmount column."
    )


def _categorise(desc: str) -> str:
    """Simple rule-based category from description."""
    d = desc.lower()

    # Income
    if "salary" in d:
        return "Salary"
    if "interest" in d:
        return "Interest"

    # Housing
    if "rent" in d and "credit" not in d:
        return "Rent"

    # EMI / loans
    if "emi" in d or "loan" in d:
        return "EMI"

    # Fuel / transport
    if "petrol" in d or "hpcl" in d or "bpcl" in d or "uber" in d or "ola" in d:
        return "Fuel & Transport"

    # Telecom / internet
    if "airtel" in d or "jio" in d or "vi " in d or "internet" in d or "recharge" in d:
        return "Mobile & Internet"

    # Subscriptions
    if "netflix" in d or "prime video" in d or "hotstar" in d or "subscription" in d:
        return "Subscriptions"

    # Food
    if "swiggy" in d or "zomato" in d or "lunch" in d or "dinner" in d or "breakfast" in d:
        return "Food & Dining"

    # Shopping
    if "amazon" in d or "flipkart" in d or "myntra" in d:
        return "Shopping"

    # Education / fees
    if "school fee" in d or "school fees" in d or "tuition" in d:
        return "Education"

    # Generic UPI
    if "upi" in d:
        return "UPI / Wallet"

    return "Others"


def _upi_counterparty(desc: str) -> str:
    """Try to extract a short UPI counterparty label."""
    d = desc.lower()
    if "upi" in d:
        # e.g. "Upi_Zomato Lunch" -> "Zomato Lunch"
        parts = re.split(r"upi[_\-\s]+", d, maxsplit=1)
        if len(parts) == 2:
            words = parts[1].split()
            if words:
                return " ".join(words[:2]).title()
    return desc[:40]


# =========================
#   MAIN ANALYSIS
# =========================

def analyze_statement(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Core analyzer used by the FastAPI backend.
    Returns a JSON-serialisable dict.
    """
    # 1) Read + basic normalisation
    df = _read_statement(file_bytes, filename)
    df = _ensure_date(df)
    df = _ensure_description(df)
    df = _build_signed_amount(df)

    # 2) Categorise + month column
    df["Category"] = df["Description"].apply(_categorise)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # ======================
    #   SUMMARY NUMBERS
    # ======================
    inflow = df.loc[df["SignedAmount"] > 0, "SignedAmount"].sum()
    outflow = -df.loc[df["SignedAmount"] < 0, "SignedAmount"].sum()
    savings_total = df["SignedAmount"].sum()

    # Monthly aggregates
    monthly = (
        df.groupby("Month")
        .agg(
            TotalInflow=("SignedAmount", lambda s: s[s > 0].sum()),
            TotalOutflow=("SignedAmount", lambda s: -s[s < 0].sum()),
        )
        .reset_index()
    )
    monthly["Savings"] = monthly["TotalInflow"] - monthly["TotalOutflow"]

    monthly_records: List[Dict[str, Any]] = [
        {
            "Month": row["Month"],
            "TotalInflow": float(row["TotalInflow"]),
            "TotalOutflow": float(row["TotalOutflow"]),
            "Savings": float(row["Savings"]),
        }
        for _, row in monthly.iterrows()
    ]

    # Current month info
    current_month = None
    if not df.empty:
        current_month = df["Date"].max().to_period("M").strftime("%Y-%m")

    if monthly_records:
        last = monthly_records[-1]
        current_month_savings = last["Savings"]

        if len(monthly_records) >= 2:
            prev = monthly_records[-2]["Savings"]
            base = prev if prev != 0 else 1.0
            growth_pct = (current_month_savings - prev) / abs(base) * 100
            sign = "+" if growth_pct >= 0 else ""
            growth_text = f"{sign}{growth_pct:.0f}% vs last month"
        else:
            growth_text = "First month tracked"
    else:
        current_month_savings = 0.0
        growth_text = "No history"

    # Simple safe daily spend: avg monthly savings / 30
    months_count = max(len(monthly_records), 1)
    avg_monthly_savings = savings_total / months_count
    safe_daily_spend = max(avg_monthly_savings / 30.0, 0.0)

    # ======================
    #   CATEGORY SPEND
    # ======================
    spend_df = df[df["SignedAmount"] < 0].copy()
    cat = (
        spend_df.groupby("Category", as_index=False)["SignedAmount"]
        .sum()
    )
    cat["TotalAmount"] = cat["SignedAmount"].abs()
    cat = cat[["Category", "TotalAmount"]].sort_values(
        "TotalAmount", ascending=False
    )

    categories: List[Dict[str, Any]] = [
        {
            "Category": row["Category"],
            "TotalAmount": float(row["TotalAmount"]),
        }
        for _, row in cat.iterrows()
    ]
    total_spend = float(cat["TotalAmount"].sum()) if not cat.empty else 0.0

    # ======================
    #   UPI ANALYSIS
    # ======================
    upi_mask = df["Description"].str.contains("upi", case=False, na=False)
    upi_df = df[upi_mask].copy()

    upi_net_outflow_month = 0.0
    upi_top_counterparties: List[Dict[str, Any]] = []

    if not upi_df.empty:
        # restrict to current month for headline number
        if current_month:
            upi_df["Month"] = upi_df["Date"].dt.to_period("M").astype(str)
            upi_this_month = upi_df[upi_df["Month"] == current_month]
        else:
            upi_this_month = upi_df

        upi_net_outflow_month = float(
            -upi_this_month.loc[upi_this_month["SignedAmount"] < 0, "SignedAmount"].sum()
        )

        # counterparties overall
        upi_df["UPILabel"] = upi_df["Description"].apply(_upi_counterparty)
        upi_grp = (
            upi_df[upi_df["SignedAmount"] < 0]
            .groupby("UPILabel", as_index=False)["SignedAmount"]
            .sum()
        )
        upi_grp["TotalAmount"] = upi_grp["SignedAmount"].abs()
        upi_grp = upi_grp.sort_values("TotalAmount", ascending=False).head(5)

        for _, row in upi_grp.iterrows():
            upi_top_counterparties.append(
                {
                    "Description": row["UPILabel"],
                    "TotalAmount": float(row["TotalAmount"]),
                }
            )

    # ======================
    #   EMI ANALYSIS
    # ======================
    emi_mask = df["Description"].str.contains("emi", case=False, na=False)
    emi_df = df[emi_mask & (df["SignedAmount"] < 0)].copy()

    emi_by_desc: List[Dict[str, Any]] = []
    emi_by_month: List[Dict[str, Any]] = []
    emi_load_current = 0.0

    if not emi_df.empty:
        # by description
        by_desc = (
            emi_df.groupby("Description", as_index=False)["SignedAmount"]
            .sum()
        )
        by_desc["TotalEMI"] = by_desc["SignedAmount"].abs()
        by_desc = by_desc[["Description", "TotalEMI"]].sort_values(
            "TotalEMI", ascending=False
        )
        for _, row in by_desc.iterrows():
            emi_by_desc.append(
                {
                    "Description": row["Description"],
                    "TotalEMI": float(row["TotalEMI"]),
                }
            )

        # by month
        emi_df["Month"] = emi_df["Date"].dt.to_period("M").astype(str)
        by_month = (
            emi_df.groupby("Month", as_index=False)["SignedAmount"]
            .sum()
        )
        by_month["TotalEMI"] = by_month["SignedAmount"].abs()
        by_month = by_month[["Month", "TotalEMI"]].sort_values("Month")

        for _, row in by_month.iterrows():
            rec = {
                "Month": row["Month"],
                "TotalEMI": float(row["TotalEMI"]),
            }
            emi_by_month.append(rec)
            if current_month and row["Month"] == current_month:
                emi_load_current = rec["TotalEMI"]

    # ======================
    #   FORECAST (simple)
    # ======================
    forecast: Dict[str, Any] = {"available": False}
    if len(monthly_records) >= 3:
        last_three = [m["Savings"] for m in monthly_records[-3:]]
        avg_next = float(sum(last_three) / len(last_three))
        forecast["available"] = True
        forecast["next_month"] = avg_next
        forecast["history"] = [
            {"month": m["Month"], "savings": m["Savings"]}
            for m in monthly_records
        ]

    # ======================
    #   CLEANED EXPORT
    # ======================
    cleaned_export = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned_export["Date"] = cleaned_export["Date"].dt.strftime("%Y-%m-%d")

    cleaned_preview = cleaned_export.sort_values(
        "Date", ascending=False
    ).head(100)

    cleaned_csv_bytes = cleaned_export.to_csv(index=False).encode("utf-8")

    # ======================
    #   FINAL JSON RESULT
    # ======================
    summary = {
        "total_inflow": float(inflow),
        "total_outflow": float(outflow),
        "savings_total": float(savings_total),
        "current_month": current_month,
        "current_month_savings": float(current_month_savings),
        "growth_text": growth_text,
        "upi_net_outflow": float(upi_net_outflow_month),
        "emi_load": float(emi_load_current),
        "safe_daily_spend": float(safe_daily_spend),
    }

    result: Dict[str, Any] = {
        "summary": summary,
        "monthly": monthly_records,
        "categories": categories,
        "upi": {"top_counterparties": upi_top_counterparties},
        "emi": {
            "by_desc": emi_by_desc,
            "by_month": emi_by_month,
        },
        "forecast": forecast,
        "cleaned_preview": cleaned_preview.to_dict(orient="records"),
        "cleaned_csv": cleaned_csv_bytes,  # main.py converts to text
    }

    return result
