import io
import re
from typing import Dict, Any, List

import pandas as pd
import numpy as np


# ------------ Helpers -------------------------------------------------


def _read_statement(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame."""
    buf = io.BytesIO(file_bytes)

    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        # fall back to CSV
        df = pd.read_csv(buf)

    # Standardise column names (strip + lower)
    df.columns = [str(c).strip() for c in df.columns]

    return df


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a Date column exists and is parsed."""
    # try common date column names
    candidates = [
        "Date",
        "Txn Date",
        "Transaction Date",
        "DATE",
        "Transaction_Date",
    ]

    date_col = None
    for c in candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(
            "Could not find a Date column. Expected one of: "
            "Date, Txn Date, Transaction Date"
        )

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).copy()

    return df


def _ensure_description(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a Description column exists."""
    candidates = [
        "Description",
        "Narration",
        "description",
        "Transaction Details",
    ]

    desc_col = None
    for c in candidates:
        if c in df.columns:
            desc_col = c
            break

    if desc_col is None:
        raise ValueError(
            "Could not find a Description / Narration column."
        )

    df["Description"] = df[desc_col].astype(str)
    return df


def _ensure_signed_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build SignedAmount:
    +ve = credit, -ve = debit.
    Supports:
    - Amount + Type (CR/DR)
    - separate Credit / Debit columns
    - already present SignedAmount
    """
    if "SignedAmount" in df.columns:
        df["SignedAmount"] = pd.to_numeric(
            df["SignedAmount"], errors="coerce"
        )
        df = df.dropna(subset=["SignedAmount"]).copy()
        return df

    # Case 1: Amount + Type
    amount_cols = [c for c in df.columns if c.lower() in ["amount", "amt"]]
    type_cols = [c for c in df.columns if c.lower() in ["type", "dr/cr", "cr/dr"]]

    if amount_cols and type_cols:
        amt_col = amount_cols[0]
        typ_col = type_cols[0]

        df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
        df = df.dropna(subset=[amt_col]).copy()

        credits = {"cr", "credit", "c"}
        types_lower = df[typ_col].astype(str).str.strip().str.lower()

        df["SignedAmount"] = np.where(
            types_lower.isin(credits),
            df[amt_col],
            -df[amt_col],
        )
        return df

    # Case 2: separate credit / debit columns
    credit_candidates = [c for c in df.columns if "credit" in c.lower()]
    debit_candidates = [c for c in df.columns if "debit" in c.lower()]

    if credit_candidates and debit_candidates:
        c_col = credit_candidates[0]
        d_col = debit_candidates[0]
        df[c_col] = pd.to_numeric(df[c_col], errors="coerce").fillna(0)
        df[d_col] = pd.to_numeric(df[d_col], errors="coerce").fillna(0)
        df["SignedAmount"] = df[c_col] - df[d_col]
        return df

    raise ValueError(
        "Could not infer amount. Expected either 'Amount' + 'Type' "
        "columns, or separate Credit / Debit columns, or SignedAmount."
    )


def _categorise(desc: str) -> str:
    """Basic rule-based categorisation from description."""
    d = desc.lower()

    # Income
    if "salary" in d or "salary credit" in d:
        return "Salary"
    if "interest" in d:
        return "Interest"

    # Housing
    if "rent" in d and "credit" not in d:
        return "Rent"

    # EMI / Loans
    if "emi" in d or "loan" in d:
        return "EMI"

    # Fuel / transport
    if "petrol" in d or "hpcl" in d or "bpcl" in d or "uber" in d or "ola" in d:
        return "Fuel & Transport"

    # Telecom / internet
    if "airtel" in d or "jio" in d or "vi " in d or "internet" in d:
        return "Mobile & Internet"

    # Subscriptions
    if (
        "netflix" in d
        or "prime video" in d
        or "hotstar" in d
        or "subscription" in d
    ):
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

    # UPI generic
    if "upi" in d:
        return "UPI / Wallet"

    return "Others"


def _upi_handle_from_desc(desc: str) -> str:
    """
    Try to extract a UPI handle / counterparty.
    This is heuristic; we just want a readable label.
    """
    d = desc.lower()
    # Example formats: "Upi_Zomato Lunch", "upi_Airtel Recharge"
    if "upi" in d:
        # split on space or hyphen after 'upi'
        # keep a short readable bit
        after = re.split(r"upi[_\-\s]+", d, maxsplit=1)
        if len(after) == 2:
            # take first 2 words
            words = after[1].split()
            if words:
                label = " ".join(words[:2])
                return label.title()
    return desc[:40]


# ------------ Core analysis -------------------------------------------


def analyze_statement(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Main analysis entrypoint.
    Returns a JSON-serializable dictionary consumed by the frontend.
    """

    df = _read_statement(file_bytes, filename)
    df = _ensure_date(df)
    df = _ensure_description(df)
    df = _ensure_signed_amount(df)

    # categorise
    df["Category"] = df["Description"].astype(str).apply(_categorise)

    # standard month string
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # ---- inflow / outflow / savings ---------------------------------
    inflow = df.loc[df["SignedAmount"] > 0, "SignedAmount"].sum()
    outflow = -df.loc[df["SignedAmount"] < 0, "SignedAmount"].sum()
    savings_total = float(df["SignedAmount"].sum())

    # monthly aggregates
    monthly = (
        df.groupby("Month")
        .agg(
            TotalInflow=("SignedAmount", lambda s: s[s > 0].sum()),
            TotalOutflow=("SignedAmount", lambda s: -s[s < 0].sum()),
        )
        .reset_index()
    )
    monthly["Savings"] = monthly["TotalInflow"] - monthly["TotalOutflow"]

    monthly_records: List[Dict[str, Any]] = []
    for _, row in monthly.iterrows():
        monthly_records.append(
            {
                "Month": row["Month"],
                "TotalInflow": float(row["TotalInflow"]),
                "TotalOutflow": float(row["TotalOutflow"]),
                "Savings": float(row["Savings"]),
            }
        )

    # ---- category spend (only outflows) ------------------------------
    spend_df = df[df["SignedAmount"] < 0].copy()
    cat = (
        spend_df.groupby("Category", as_index=False)["SignedAmount"]
        .sum()
    )
    cat["TotalAmount"] = cat["SignedAmount"].abs()
    cat = cat[["Category", "TotalAmount"]].sort_values(
        "TotalAmount", ascending=False
    )

    categories: List[Dict[str, Any]] = []
    for _, row in cat.iterrows():
        categories.append(
            {
                "Category": row["Category"],
                "TotalAmount": float(row["TotalAmount"]),
            }
        )

    total_spend = float(cat["TotalAmount"].sum()) if not cat.empty else 0.0

    # ---- UPI analysis -----------------------------------------------
    upi_mask = df["Description"].str.contains("upi", case=False, na=False)
    upi_rows = df[upi_mask].copy()

    # current month = latest Month in data
    current_month = None
    if not df.empty:
        current_month = (
            df["Date"].max().to_period("M").strftime("%Y-%m")
        )

    upi_net_outflow_month = 0.0
    upi_top_counterparties: List[Dict[str, Any]] = []

    if not upi_rows.empty:
        # restrict to current month for the main number
        if current_month is not None:
            upi_rows["Month"] = upi_rows["Date"].dt.to_period("M").astype(
                str
            )
            upi_rows_month = upi_rows[upi_rows["Month"] == current_month]
        else:
            upi_rows_month = upi_rows

        upi_net_outflow_month = float(
            -upi_rows_month.loc[
                upi_rows_month["SignedAmount"] < 0, "SignedAmount"
            ].sum()
        )

        # top counterparties across whole period
        upi_rows["UPILabel"] = upi_rows["Description"].apply(
            _upi_handle_from_desc
        )
        upi_group = (
            upi_rows[upi_rows["SignedAmount"] < 0]
            .groupby("UPILabel", as_index=False)["SignedAmount"]
            .sum()
        )
        upi_group["TotalAmount"] = upi_group["SignedAmount"].abs()
        upi_group = upi_group.sort_values(
            "TotalAmount", ascending=False
        ).head(5)

        for _, row in upi_group.iterrows():
            upi_top_counterparties.append(
                {
                    "Description": row["UPILabel"],
                    "TotalAmount": float(row["TotalAmount"]),
                }
            )

    # ---- EMI analysis -----------------------------------------------
    emi_mask = df["Description"].str.contains("emi", case=False, na=False)
    emi_df = df[emi_mask & (df["SignedAmount"] < 0)].copy()

    emi_by_desc: List[Dict[str, Any]] = []
    emi_by_month: List[Dict[str, Any]] = []

    if not emi_df.empty:
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

        emi_df["Month"] = emi_df["Date"].dt.to_period("M").astype(str)
        by_month = (
            emi_df.groupby("Month", as_index=False)["SignedAmount"]
            .sum()
        )
        by_month["TotalEMI"] = by_month["SignedAmount"].abs()
        by_month = by_month[["Month", "TotalEMI"]].sort_values(
            "Month"
        )

        for _, row in by_month.iterrows():
            emi_by_month.append(
                {
                    "Month": row["Month"],
                    "TotalEMI": float(row["TotalEMI"]),
                }
            )

    emi_load_current = 0.0
    if current_month and emi_by_month:
        for row in emi_by_month:
            if row["Month"] == current_month:
                emi_load_current = row["TotalEMI"]
                break

    # ---- simple forecast from monthly savings -----------------------
    forecast: Dict[str, Any] = {"available": False}
    if len(monthly_records) >= 3:
        last_savings = [m["Savings"] for m in monthly_records[-3:]]
        avg_last = float(sum(last_savings) / len(last_savings))
        forecast["available"] = True
        forecast["next_month"] = avg_last
        forecast["delta_vs_last"] = float(
            last_savings[-1] - last_savings[-2]
        )
        forecast["history"] = [
            {"month": m["Month"], "savings": m["Savings"]}
            for m in monthly_records
        ]

    # ---- safe daily spend -------------------------------------------
    months_count = max(len(monthly_records), 1)
    avg_monthly_savings = savings_total / months_count
    safe_daily_spend = max(avg_monthly_savings / 30.0, 0.0)

    # current month savings & growth text
    if monthly_records:
        last_month = monthly_records[-1]
        current_month_savings = last_month["Savings"]
        if len(monthly_records) >= 2:
            prev = monthly_records[-2]["Savings"] or 1.0
            growth_pct = (
                (current_month_savings - prev) / abs(prev)
            ) * 100.0
            sign = "+" if growth_pct >= 0 else ""
            growth_text = f"{sign}{growth_pct:.0f}% vs last month"
        else:
            growth_text = "First month tracked"
    else:
        current_month_savings = 0.0
        growth_text = "No history"

    # ---- summary ----------------------------------------------------
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

    # ---- cleaned preview & CSV --------------------------------------
    cleaned_export = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned_export["Date"] = cleaned_export["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    cleaned_preview = cleaned_export.sort_values(
        "Date", ascending=False
    ).head(100)

    cleaned_csv_bytes = cleaned_export.to_csv(index=False).encode("utf-8")

    result: Dict[str, Any] = {
        "summary": summary,
        "monthly": monthly_records,
        "categories": categories,
        "upi": {
            "top_counterparties": upi_top_counterparties,
        },
        "emi": {
            "by_desc": emi_by_desc,
            "by_month": emi_by_month,
        },
        "forecast": forecast,
        "cleaned_preview": cleaned_preview.to_dict(orient="records"),
        "cleaned_csv": cleaned_csv_bytes,  # main.py converts to text
    }

    return result
