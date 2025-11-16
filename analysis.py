import pandas as pd
import numpy as np
from io import BytesIO
import calendar

# Optional ML import – safe fallback if missing
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None


# =========================
# FUTURE / PREDICTION BLOCK
# =========================
def build_future_block(df: pd.DataFrame, safe_daily_spend: float) -> dict:
    """
    Prediction block based on current month progression + simple ML model.
    If scikit-learn is available, we fit a tiny LinearRegression on
    monthly inflow/outflow vs net savings and use that to refine the
    projected end-of-month savings.
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

    # 1. Current / previous month split
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

    # 2. Baseline end-of-month projection
    inflow_to_date = df_current[df_current["signed_amount"] > 0]["signed_amount"].sum()
    outflow_to_date = -df_current[df_current["signed_amount"] < 0]["signed_amount"].sum()

    avg_daily_inflow = inflow_to_date / max(days_elapsed, 1)
    avg_daily_outflow = outflow_to_date / max(days_elapsed, 1)

    projected_inflow = avg_daily_inflow * days_in_month
    projected_outflow = avg_daily_outflow * days_in_month

    predicted_eom_savings = float(projected_inflow - projected_outflow)

    # 2b. ML refinement (if sklearn available)
    if LinearRegression is not None:
        monthly_ml = (
            df.assign(
                inflow=lambda x: x["signed_amount"].where(x["signed_amount"] > 0, 0.0),
                outflow=lambda x: x["signed_amount"].where(x["signed_amount"] < 0, 0.0),
            )
            .groupby("month")
            .agg(
                total_inflow=("inflow", "sum"),
                total_outflow=("outflow", "sum"),  # negative
                net=("signed_amount", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )

        if len(monthly_ml) >= 3:
            X = monthly_ml[["total_inflow", "total_outflow"]].to_numpy()
            y = monthly_ml["net"].to_numpy()

            try:
                model = LinearRegression()
                model.fit(X, y)

                X_new = np.array([[projected_inflow, -projected_outflow]])
                y_pred = model.predict(X_new)[0]
                predicted_eom_savings = float(y_pred)
            except Exception:
                # fall back to heuristic value
                pass

    # Confidence band ±15%
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

    overspend_risk = {"level": level, "probability": overspend_prob}

    # 4. Risky categories
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


# ===================
# HEADER DETECTION
# ===================
def _looks_like_header(row: pd.Series) -> bool:
    """Heuristic to detect the real header row in Excel exports."""
    vals = [str(v).strip().lower() for v in row.tolist()]

    has_date = any("date" in v for v in vals)

    desc_keys = ["desc", "narrat", "narration", "particular", "details"]
    has_desc = any(any(k in v for k in desc_keys) for v in vals)

    amt_keys = ["amount", "amt", "withdrawal", "deposit", "debit", "credit"]
    has_amt = any(any(k in v for k in amt_keys) for v in vals)

    return has_date and has_desc and has_amt


# ===================
# MAIN ANALYSIS
# ===================
def analyze_statement(file_bytes, file_name):
    name = file_name.lower()

    # -----------------------------
    # 1. LOAD FILE
    # -----------------------------
    if name.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))

        # CSV rescue: case where everything is quoted into one column
        if df.shape[1] == 1 and "," in str(df.columns[0]):
            text = file_bytes.decode("utf-8", errors="ignore")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            rows = [ln.strip('"') for ln in lines]  # remove wrapping quotes
            split_rows = [r.split(",") for r in rows]

            header, data_rows = split_rows[0], split_rows[1:]
            df = pd.DataFrame(data_rows, columns=[h.strip() for h in header])

    elif name.endswith(".xlsx"):
        df_raw = pd.read_excel(BytesIO(file_bytes), header=None)
        header_row_idx = None
        max_scan = min(60, len(df_raw))
        for i in range(max_scan):
            if _looks_like_header(df_raw.iloc[i]):
                header_row_idx = i
                break

        if header_row_idx is None:
            raise Exception(
                "Could not locate header row in Excel. "
                "Try exporting the statement again as CSV or simpler XLSX."
            )

        header = df_raw.iloc[header_row_idx].astype(str).tolist()
        df = df_raw.iloc[header_row_idx + 1 :].copy()
        df.columns = header

    elif name.endswith(".xls"):
        # old Excel – needs xlrd installed
        df_raw = pd.read_excel(BytesIO(file_bytes), header=None, engine="xlrd")
        header_row_idx = None
        max_scan = min(60, len(df_raw))
        for i in range(max_scan):
            if _looks_like_header(df_raw.iloc[i]):
                header_row_idx = i
                break

        if header_row_idx is None:
            raise Exception(
                "Could not locate header row in .xls file. "
                "Try exporting the statement again as CSV or XLSX."
            )

        header = df_raw.iloc[header_row_idx].astype(str).tolist()
        df = df_raw.iloc[header_row_idx + 1 :].copy()
        df.columns = header

    else:
        raise Exception("Unsupported file type. Please upload CSV or Excel.")

    # -----------------------------
    # 2. STANDARDIZE COLUMNS
    # -----------------------------
    df.columns = df.columns.str.strip().str.lower()

    # Detect core columns
    date_col = next((c for c in df.columns if "date" in c), None)

    desc_col = next(
        (
            c
            for c in df.columns
            if any(
                k in c
                for k in ["desc", "narrat", "narration", "particular", "details"]
            )
        ),
        None,
    )

    amount_col = next((c for c in df.columns if "amount" in c or "amt" in c), None)

    # Also look for withdrawal / deposit style columns for banks like HDFC
    debit_col = next(
        (c for c in df.columns if "withdrawal" in c or "debit" in c), None
    )
    credit_col = next(
        (c for c in df.columns if "deposit" in c or "credit" in c), None
    )

    # If we don't have a single amount column, synthesize from debit/credit/withdrawal/deposit
    if not amount_col and (debit_col or credit_col):
        df["amount"] = 0.0
        if debit_col:
            df["amount"] -= pd.to_numeric(df[debit_col], errors="coerce").fillna(0)
        if credit_col:
            df["amount"] += pd.to_numeric(df[credit_col], errors="coerce").fillna(0)
        amount_col = "amount"

    # Final validation
    if not date_col or not desc_col or not amount_col:
        raise Exception(
            "Missing required columns (Date, Description, Amount). "
            f"Found: {list(df.columns)}"
        )

    # Rename to the unified names used later
    rename_map = {
        date_col: "date",
        desc_col: "description",
        amount_col: "amount",
    }

    type_col = next((c for c in df.columns if "type" in c), None)
    if type_col:
        rename_map[type_col] = "type"

    df = df.rename(columns=rename_map)

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
