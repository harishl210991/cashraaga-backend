from io import BytesIO
from typing import Dict, Any, Tuple

import pandas as pd

# Optional forecast library
try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False


def categorize(description: str) -> str:
    """Simple rule-based category from transaction description."""
    d = str(description).lower()

    if any(k in d for k in ["swiggy", "zomato", "restaurant", "food", "dining"]):
        return "Food & Dining"
    if any(k in d for k in ["amazon", "flipkart", "myntra", "ajio", "meesho"]):
        return "Shopping"
    if "rent" in d:
        return "Rent"
    if any(k in d for k in ["petrol", "fuel", "shell", "hpcl", "bpcl", "indianoil"]):
        return "Fuel & Transport"
    if any(k in d for k in ["recharge", "jio", "airtel", "vi ", "vodafone", "bsnl"]):
        return "Mobile & Internet"
    if any(k in d for k in ["electricity", "eb", "tneb", "gas bill", "power bill"]):
        return "Utilities"
    if any(k in d for k in ["salary", "payroll", "salary credit", "sal "]):
        return "Salary"
    if any(k in d for k in ["emi", "loan", "repayment"]):
        return "Loans & EMI"
    if any(k in d for k in ["hospital", "pharmacy", "medical", "clinic"]):
        return "Health & Medical"
    if any(k in d for k in ["school", "college", "fees", "tuition"]):
        return "Education"
    if any(k in d for k in ["netflix", "hotstar", "prime video", "prime ", "spotify", "wynk"]):
        return "Entertainment & Subscriptions"
    return "Others"


def detect_columns(df_raw: pd.DataFrame) -> Tuple[str, str, str, str]:
    """
    Try to detect:
      - date column
      - description column
      - signed amount column name (we will create if needed)
      - strategy: "amount" or "credit_debit"

    Returns: (date_col, desc_col, signed_col, strategy)
    """
    cols = list(df_raw.columns)
    lower_map = {c.lower().strip(): c for c in cols}

    # Date
    date_candidates = [
        "date",
        "txn date",
        "transaction date",
        "trans date",
        "value date",
        "posting date",
    ]
    date_col = None
    for cand in date_candidates:
        if cand in lower_map:
            date_col = lower_map[cand]
            break
    if date_col is None:
        # fallback: any column with 'date'
        for lc, orig in lower_map.items():
            if "date" in lc:
                date_col = orig
                break
    if date_col is None:
        raise ValueError("Could not detect date column")

    # Description / narration
    desc_candidates = [
        "description",
        "narration",
        "details",
        "transaction details",
        "particulars",
        "remarks",
    ]
    desc_col = None
    for cand in desc_candidates:
        if cand in lower_map:
            desc_col = lower_map[cand]
            break
    if desc_col is None:
        for lc, orig in lower_map.items():
            if any(k in lc for k in ["descr", "narr", "details", "particular"]):
                desc_col = orig
                break
    if desc_col is None:
        raise ValueError("Could not detect description/narration column")

    # Amount / credit / debit
    # 1) direct "amount"
    amount_col = None
    if "amount" in lower_map:
        amount_col = lower_map["amount"]
        return date_col, desc_col, amount_col, "amount"

    # 2) credit + debit style
    credit_candidates = [
        "credit",
        "cr",
        "cr amount",
        "credit amount",
        "deposit",
        "deposit amount",
    ]
    debit_candidates = [
        "debit",
        "dr",
        "dr amount",
        "debit amount",
        "withdrawal",
        "withdrawal amount",
    ]

    credit_col = None
    debit_col = None

    for cand in credit_candidates:
        if cand in lower_map:
            credit_col = lower_map[cand]
            break
    if credit_col is None:
        for lc, orig in lower_map.items():
            if any(k in lc for k in ["credit", "cr", "deposit"]):
                credit_col = orig
                break

    for cand in debit_candidates:
        if cand in lower_map:
            debit_col = lower_map[cand]
            break
    if debit_col is None:
        for lc, orig in lower_map.items():
            if any(k in lc for k in ["debit", "dr", "withdraw"]):
                debit_col = orig
                break

    if credit_col and debit_col:
        # We'll create SignedAmount = credit - debit
        return date_col, desc_col, f"{credit_col}__{debit_col}__signed", "credit_debit"

    raise ValueError(
        "Could not detect Amount / Credit / Debit columns. "
        "Please ensure the file has an Amount column or Credit/Debit columns."
    )


def normalize_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Take raw df and return a normalized frame with:
      - Date (datetime)
      - Description (str)
      - SignedAmount (float, +inflow, -outflow)
    """
    date_col, desc_col, signed_col, strategy = detect_columns(df_raw)

    df = pd.DataFrame()
    df["Date"] = pd.to_datetime(df_raw[date_col], errors="coerce")
    df["Description"] = df_raw[desc_col].astype(str).fillna("")

    if strategy == "amount":
        df["SignedAmount"] = pd.to_numeric(df_raw[signed_col], errors="coerce")
    else:
        # credit_debit
        credit_col, debit_col, _ = signed_col.split("__")
        credit = pd.to_numeric(df_raw[credit_col], errors="coerce").fillna(0)
        debit = pd.to_numeric(df_raw[debit_col], errors="coerce").fillna(0)
        df["SignedAmount"] = credit - debit  # positive inflow, negative outflow

    df = df.dropna(subset=["Date", "SignedAmount"])
    return df


def analyze_statement(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Core CashRaaga analysis (no Streamlit, only logic).

    Auto-detects common Indian bank formats for Date / Description / Amount.
    """

    # ---------- READ FILE ----------
    buf = BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        df_raw = pd.read_csv(buf)
    else:
        df_raw = pd.read_excel(buf)

    if df_raw.empty:
        raise ValueError("Uploaded file is empty")

    # ---------- NORMALISATION ----------
    df = normalize_dataframe(df_raw)
    if df.empty:
        raise ValueError("No valid rows after cleaning")

    df["Category"] = df["Description"].apply(categorize)
    df["Month"] = df["Date"].dt.strftime("%Y-%m")

    # ---------- AGGREGATES ----------
    total_inflow = df.loc[df["SignedAmount"] > 0, "SignedAmount"].sum()
    total_outflow = df.loc[df["SignedAmount"] < 0, "SignedAmount"].sum()
    savings_total = total_inflow + total_outflow  # outflow is negative

    # Monthly inflow/outflow/savings
    monthly_inflow = df[df["SignedAmount"] > 0].groupby("Month")["SignedAmount"].sum()
    monthly_outflow = (
        df[df["SignedAmount"] < 0].groupby("Month")["SignedAmount"].sum().abs()
    )

    all_months = sorted(set(monthly_inflow.index) | set(monthly_outflow.index))
    monthly_inflow = monthly_inflow.reindex(all_months, fill_value=0)
    monthly_outflow = monthly_outflow.reindex(all_months, fill_value=0)
    monthly_savings = monthly_inflow - monthly_outflow

    monthly_df = pd.DataFrame(
        {
            "Month": all_months,
            "TotalInflow": monthly_inflow.values,
            "TotalOutflow": monthly_outflow.values,
            "Savings": monthly_savings.values,
        }
    )

    # ---------- UPI & EMI ----------
    upi_df = df[df["Description"].str.contains("UPI", case=False, na=False)].copy()
    emi_mask = df["Description"].str.contains("EMI|LOAN", case=False, na=False)
    emi_df = df[(df["SignedAmount"] < 0) & emi_mask].copy()

    # Current month snapshot (last month in data)
    if not monthly_df.empty:
        last_row = monthly_df.iloc[-1]
        current_month = last_row["Month"]
        this_savings = float(last_row["Savings"])

        if len(monthly_df) > 1:
            prev_savings = float(monthly_df.iloc[-2]["Savings"])
            if prev_savings != 0:
                growth_pct = (this_savings - prev_savings) / abs(prev_savings) * 100
                growth_txt = f"{growth_pct:+.0f}% vs last month"
            else:
                growth_txt = "first month in data"
        else:
            prev_savings = None
            growth_txt = "first month in data"

        # UPI net outflow current month
        if not upi_df.empty:
            upi_current = upi_df[upi_df["Date"].dt.strftime("%Y-%m") == current_month]
            upi_net = upi_current["SignedAmount"].sum()
            upi_net_out = abs(upi_net) if upi_net < 0 else 0
        else:
            upi_net_out = 0.0

        # EMI total current month
        if not emi_df.empty:
            emi_current = emi_df[emi_df["Date"].dt.strftime("%Y-%m") == current_month]
            emi_load = (
                abs(emi_current["SignedAmount"].sum()) if not emi_current.empty else 0.0
            )
        else:
            emi_load = 0.0

        safe_daily = max(this_savings, 0.0) / 30 if this_savings > 0 else 0.0
    else:
        current_month = None
        this_savings = 0.0
        prev_savings = None
        growth_txt = "no history"
        upi_net_out = 0.0
        emi_load = 0.0
        safe_daily = 0.0

    # ---------- TOP UPI & EMI ----------
    if not upi_df.empty:
        upi_top = (
            upi_df.groupby("Description")["SignedAmount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"SignedAmount": "TotalAmount"})
        )
        upi_top_records = upi_top.to_dict(orient="records")
    else:
        upi_top_records = []

    if not emi_df.empty:
        emi_by_desc = (
            emi_df.groupby("Description")["SignedAmount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"SignedAmount": "TotalEMI"})
        )
        emi_by_month = (
            emi_df.groupby("Month")["SignedAmount"]
            .sum()
            .abs()
            .reset_index()
            .rename(columns={"SignedAmount": "TotalEMI"})
        )
        emi_by_desc_records = emi_by_desc.to_dict(orient="records")
        emi_by_month_records = emi_by_month.to_dict(orient="records")
    else:
        emi_by_desc_records = []
        emi_by_month_records = []

    # ---------- CATEGORY BREAKUP ----------
    cat_breakup = (
        df.groupby("Category")["SignedAmount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"SignedAmount": "TotalAmount"})
    )
    cat_records = cat_breakup.to_dict(orient="records")

    # ---------- FORECAST ----------
    forecast_info: Dict[str, Any] = {"available": False}
    monthly_series = monthly_df.set_index("Month")["Savings"]
    if HAS_PMDARIMA and len(monthly_series) >= 3:
        try:
            model = auto_arima(
                monthly_series,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
            )
            n_future = 3
            forecast = model.predict(n_periods=n_future)
            next_month_pred = float(forecast[0])
            last_val = float(monthly_series.iloc[-1])
            delta_vs_last = next_month_pred - last_val

            forecast_info = {
                "available": True,
                "next_month": next_month_pred,
                "delta_vs_last": delta_vs_last,
                "history": [
                    {"month": m, "savings": float(v)}
                    for m, v in monthly_series.items()
                ],
                "future": [
                    {"step": i + 1, "predicted_savings": float(v)}
                    for i, v in enumerate(forecast)
                ],
            }
        except Exception as e:
            forecast_info = {"available": False, "error": str(e)}

    # ---------- CLEANED PREVIEW ----------
    cleaned_preview = df.sort_values("Date", ascending=False).head(20)[
        ["Date", "Description", "SignedAmount", "Category"]
    ].copy()
    cleaned_preview["Date"] = cleaned_preview["Date"].dt.strftime("%Y-%m-%d")
    cleaned_preview_records = cleaned_preview.to_dict(orient="records")

    # ---------- CLEANED EXPORT CSV ----------
    cleaned_export = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned_export["Date"] = cleaned_export["Date"].dt.strftime("%Y-%m-%d")
    cleaned_export.rename(columns={"SignedAmount": "Amount"}, inplace=True)
    csv_bytes = cleaned_export.to_csv(index=False).encode("utf-8")

    return {
        "summary": {
            "total_inflow": float(total_inflow),
            "total_outflow": float(total_outflow),
            "savings_total": float(savings_total),
            "current_month": current_month,
            "current_month_savings": float(this_savings),
            "growth_text": growth_txt,
            "upi_net_outflow": float(upi_net_out),
            "emi_load": float(emi_load),
            "safe_daily_spend": float(safe_daily),
        },
        "monthly": monthly_df.to_dict(orient="records"),
        "categories": cat_records,
        "upi": {
            "top_counterparties": upi_top_records,
        },
        "emi": {
            "by_desc": emi_by_desc_records,
            "by_month": emi_by_month_records,
        },
        "forecast": forecast_info,
        "cleaned_preview": cleaned_preview_records,
        # raw CSV bytes – main.py will turn this into text
        "cleaned_csv": csv_bytes,
    }
