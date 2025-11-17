import os
import json
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import calendar

# ----------------- OPTIONAL: Gemini for column mapping -----------------
try:
    import google.generativeai as genai
except ImportError:
    genai = None

GEMINI_MODEL = os.getenv("CASHRAAGA_LLM_MODEL", "gemini-1.5-flash")


def llm_map_columns(headers, preview_rows):
    """
    Optional LLM helper to map weird column names to:
    - date
    - description
    - debit (money going out)
    - credit (money coming in)

    Used ONLY when heuristics fail.
    """
    if genai is None or not os.getenv("GEMINI_API_KEY"):
        return {}

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
You are helping to read a bank statement table.

Headers:
{headers}

Preview rows (as CSV):
{preview_rows}

Your task:
Return a JSON object with keys among:
- "date"
- "description"
- "debit"
- "credit"
and values = EXACT header names from the list.

If a key is unknown, omit it. Do NOT explain, only output JSON.
"""

        resp = model.generate_content(prompt)
        text = resp.text.strip()
        # Try to extract JSON
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        js = text[start : end + 1]
        mapping = json.loads(js)
        # Ensure values are valid headers
        mapping = {k: v for k, v in mapping.items() if v in headers}
        return mapping
    except Exception:
        return {}


# ----------------- FUTURE / PREDICTION BLOCK (same as before) -----------------


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


# ----------------- LOADING + BANK TEMPLATES -----------------


def _load_raw(file_bytes, file_name):
    """
    Load CSV / Excel as raw string table (header=None).
    """
    name = file_name.lower()

    if name.endswith(".csv"):
        text = file_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # Fix 'all fields wrapped in ""' pattern
        sample = lines[:20]
        quoted = sum(
            1
            for ln in sample
            if len(ln.strip()) >= 2
            and ln.strip().startswith('"')
            and ln.strip().endswith('"')
        )
        if sample and quoted / len(sample) > 0.6:
            lines = [
                ln.strip()[1:-1]
                if ln.strip().startswith('"') and ln.strip().endswith('"')
                else ln
                for ln in lines
            ]
            text = "\n".join(lines)

        df_raw = pd.read_csv(StringIO(text), header=None, dtype=str)
    else:
        df_raw = pd.read_excel(BytesIO(file_bytes), header=None, dtype=str)

    return df_raw


def _guess_header_row(df_raw: pd.DataFrame) -> int:
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
            any(k in c for k in ("narrat", "desc", "details", "particular", "remark"))
            for c in cells
        ):
            score += 2
        if any(
            any(
                k in c
                for k in ("amount", "amt", "balance", "debit", "credit", "withdraw", "deposit")
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


def _normalize_generic(df_raw: pd.DataFrame) -> pd.DataFrame:
    hdr_row = _guess_header_row(df_raw)
    header = df_raw.iloc[hdr_row].fillna("").astype(str).str.strip()
    df = df_raw.iloc[hdr_row + 1 :].reset_index(drop=True).copy()
    df.columns = header
    df = df.dropna(axis=1, how="all")
    return df


def _detect_bank(df_raw: pd.DataFrame) -> str:
    """
    Very simple detector: look for HDFC-style headers or text.
    """
    first_text = (
        " ".join(
            df_raw.iloc[:10]
            .fillna("")
            .astype(str)
            .to_numpy()
            .flatten()
            .tolist()
        )
        .lower()
    )

    if "hdfc bank" in first_text or "withdrawal amt" in first_text:
        return "HDFC"

    # Can add more: ICICI, SBI, etc.
    return "GENERIC"


# ----------------- BANK-SPECIFIC PARSERS -----------------


def parse_hdfc(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    HDFC template:

    Date | Narration | Chq./Ref.No. | Value Dt | Withdrawal Amt. | Deposit Amt. | Closing Balance
    """
    hdr_row = _guess_header_row(df_raw)
    header = df_raw.iloc[hdr_row].fillna("").astype(str).str.strip()
    df = df_raw.iloc[hdr_row + 1 :].reset_index(drop=True).copy()
    df.columns = header
    df = df.dropna(axis=1, how="all")

    # Canonicalize column names
    cols = {c: c for c in df.columns}
    lower_cols = {c.lower(): c for c in df.columns}

    # Date column (usually "Date")
    date_col = None
    for c in df.columns:
        if "date" in c.lower() and "value" not in c.lower():
            date_col = c
            break

    narr_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ("narration", "description", "details")):
            narr_col = c
            break

    # Withdrawal / Deposit with aggressive normalization
    def norm_name(c):
        return c.replace(".", "").replace(" ", "").lower()

    debit_col = None
    credit_col = None
    for c in df.columns:
        n = norm_name(c)
        if any(k in n for k in ("withdrawalamt", "withdrawal", "debit", "dr")):
            debit_col = c
        if any(k in n for k in ("depositamt", "deposit", "credit", "cr")):
            credit_col = c

    if date_col is None:
        raise Exception("Missing required date column.")
    if narr_col is None:
        raise Exception("Missing required transaction description column.")
    if debit_col is None and credit_col is None:
        raise Exception("Missing withdrawal/deposit columns for HDFC template.")

    # Clean up
    df = df[[c for c in [date_col, narr_col, debit_col, credit_col] if c is not None]].copy()
    df.rename(
        columns={
            date_col: "date",
            narr_col: "description",
        },
        inplace=True,
    )

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

    df["amount"] = credit_vals - debit_vals  # +ve credit, -ve debit
    df["signed_amount"] = df["amount"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df[["date", "description", "signed_amount"]]


def parse_generic(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_generic(df_raw)
    df.columns = df.columns.map(lambda c: str(c).strip())

    headers = list(df.columns)

    # Heuristic headers
    def find_col(keywords):
        for c in df.columns:
            name = c.lower()
            if any(k in name for k in keywords):
                return c
        return None

    date_col = find_col(["date"])
    desc_col = find_col(["narrat", "desc", "details", "particular", "remark"])
    amount_col = find_col(["amount", "amt"])
    debit_col = find_col(["debit", "withdraw", "dr"])
    credit_col = find_col(["credit", "deposit", "cr"])

    # If still missing, ask LLM as last resort
    if date_col is None or (amount_col is None and (debit_col is None or credit_col is None)):
        preview_csv = df.head(10).to_csv(index=False)
        mapping = llm_map_columns(headers, preview_csv)
        # LLM can return debit/credit instead of amount
        if date_col is None and "date" in mapping:
            date_col = mapping["date"]
        if desc_col is None and "description" in mapping:
            desc_col = mapping["description"]
        if "debit" in mapping:
            debit_col = mapping["debit"]
        if "credit" in mapping:
            credit_col = mapping["credit"]

    if date_col is None:
        raise Exception("Missing required date column.")
    if desc_col is None:
        raise Exception("Missing required transaction description column.")

    # If debit/credit available, build amount
    if amount_col is None and (debit_col or credit_col):
        for col in [debit_col, credit_col]:
            if col:
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
        df["amount"] = credit_vals - debit_vals
        amount_col = "amount"

    if amount_col is None:
        raise Exception("Missing required amount column.")

    df = df[[date_col, desc_col, amount_col]].copy()
    df.rename(
        columns={
            date_col: "date",
            desc_col: "description",
            amount_col: "amount",
        },
        inplace=True,
    )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["signed_amount"] = df["amount"]
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df[["date", "description", "signed_amount"]]


# ----------------- CATEGORY & SUMMARY -----------------


def _detect_category(desc: str) -> str:
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


# ----------------- MAIN ENTRY -----------------


def analyze_statement(file_bytes, file_name):
    df_raw = _load_raw(file_bytes, file_name)
    bank = _detect_bank(df_raw)

    if bank == "HDFC":
        core = parse_hdfc(df_raw)
    else:
        core = parse_generic(df_raw)

    df = core.copy()
    df["category"] = df["description"].apply(_detect_category)
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
        prev_savings = 0.0
        mom_change = 0.0

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

    emi_df = df[df["category"] == "Emi"]
    emi_month_df = emi_df[emi_df["month"] == latest_month]
    emi_load = emi_month_df["signed_amount"].abs().sum()
    emi_months = emi_df["month"].nunique()

    cat_summary = (
        df[df["signed_amount"] < 0]
        .groupby("category")["signed_amount"]
        .sum()
        .abs()
        .reset_index()
    ).sort_values("signed_amount", ascending=False)

    safe_daily = max(0.0, this_month_savings) / 30
    future_block = build_future_block(df, safe_daily)

    cleaned_export = df[["date", "description", "signed_amount", "category"]].copy()
    cleaned_export["date"] = cleaned_export["date"].astype(str)
    csv_output = cleaned_export.to_csv(index=False)

    return {
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
