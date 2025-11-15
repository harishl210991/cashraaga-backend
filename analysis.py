import pandas as pd
import numpy as np
import io
from datetime import datetime


# -----------------------------
# UTILITIES
# -----------------------------

def smart_parse_date(x):
    """
    Robust Indian bank date parser.
    """
    if pd.isna(x):
        return None

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(str(x).strip(), fmt)
        except:
            pass

    try:
        # Attempt standard pandas datetime conversion as a last resort
        return pd.to_datetime(x, errors="coerce")
    except:
        return None


def detect_columns(df):
    """
    Detects key columns (Date, Description, Amount/Credit/Debit) based on common headers.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    # DATE
    date_col = None
    for k, v in cols.items():
        if "date" in k:
            date_col = v
            break

    # DESCRIPTION
    desc_col = None
    for k, v in cols.items():
        if any(t in k for t in ["description", "narration", "details", "remarks", "particular"]):
            desc_col = v
            break

    # AMOUNT or CREDIT/DEBIT
    amount_col = None
    credit_col = None
    debit_col = None

    for k, v in cols.items():
        if "amount" in k and not any(t in k for t in ["opening", "closing", "balance"]):
            amount_col = v
        if "credit" in k and "interest" not in k:
            credit_col = v
        if "debit" in k:
            debit_col = v
            
    # Priority: Credit/Debit columns are often cleaner than a generic 'Amount' column
    # If both single 'Amount' and 'Credit/Debit' exist, C/D will be used inside build_signed_amount
    
    return date_col, desc_col, amount_col, credit_col, debit_col


def clean_and_convert_to_numeric(series):
    """
    Aggressively cleans string series from common bank statement formatting issues 
    (currency symbols, commas, spaces, parentheses for negative) and converts to numeric.
    """
    if series is None or len(series) == 0:
        return pd.Series([], dtype='float64')

    s_str = series.astype(str).str.strip()

    # 1. Remove currency symbols (₹, Rs), and non-breaking spaces
    # Note: Using regex to handle 'Rs.' or 'Rs' variations
    s_str = s_str.str.replace(r'₹|Rs\.?|\s', '', regex=True)

    # 2. Handle parentheses for debit/negative: (100.00) -> -100.00
    # This regex captures content inside parentheses and prepends a minus sign.
    s_str = s_str.str.replace(r'^\((.*)\)$', r'-\1', regex=True)

    # 3. Remove commas, which are likely thousand separators that break parsing
    # Must be done after handling parentheses to avoid issues like (-1,000)
    s_str = s_str.str.replace(',', '', regex=False)
    
    # 4. Convert to numeric. Errors (e.g., remaining non-numeric text) go to NaN, then fill 0.
    return pd.to_numeric(s_str, errors="coerce").fillna(0)


def build_signed_amount(df, amount_col, credit_col, debit_col):
    """
    Creates SignedAmount column:
    - Positive for inflow (Credit)
    - Negative for outflow (Debit)
    - Uses cleaned numeric conversion.
    """

    # Case 2: Separate Credit/Debit columns (Most reliable for sign)
    if credit_col or debit_col:
        # Pass the original columns to the cleaner
        cr = clean_and_convert_to_numeric(df.get(credit_col, pd.Series([0])))
        dr = clean_and_convert_to_numeric(df.get(debit_col, pd.Series([0])))
        # Credit - Debit logic is always correct if columns are clean
        return cr - dr

    # Case 1: Single Amount column
    elif amount_col:
        # Use cleaning utility for the amount column
        amt = clean_and_convert_to_numeric(df[amount_col])
        
        # Check if a Type (DR/CR) column exists to determine sign
        tcol = None
        for c in df.columns:
            if c.lower() in ["type", "dr/cr", "cr/dr", "transaction type"]:
                tcol = c
                break

        if tcol:
            t = df[tcol].astype(str).str.upper()
            dr = t.str.contains("DR")
            cr = t.str.contains("CR")

            # Check if values are already signed (e.g., -100 for DR, 100 for CR)
            # This check prevents double-signing
            # Note: Checking mean is a heuristic. It's safer to just rely on DR/CR for sign.
            if (amt[dr].mean() < 0) and (amt[cr].mean() > 0):
                return amt  # already signed

            # Else: assume absolute amounts and use DR/CR column to apply sign
            return np.where(cr, amt.abs(), -amt.abs())

        # Case 1c: NO type column - we assume file has signed values
        return amt

    raise ValueError("Unable to determine amounts in statement.")


# -----------------------------
# CATEGORY ENGINE
# -----------------------------

def classify_category(description):
    """
    A basic, rule-based classifier for transaction descriptions.
    (UPDATED with Groceries, Investments, and Cash)
    """
    d = description.lower()

    if "salary" in d or "income" in d or "transfer from" in d:
        # Mark all inflows (positive SignedAmount) as 'Income' in final step
        return "INCOME_MARKER" 
    
    if "rent" in d or "landlord" in d:
        return "Rent"
    if "emi" in d or "loan" in d:
        return "EMI"
    if "amazon" in d or "flipkart" in d or "myntra" in d or "shop" in d:
        return "Shopping"
    if "swiggy" in d or "zomato" in d or "dine" in d or "restaurant" in d:
        return "Food & Dining"
    if "petrol" in d or "hpcl" in d or "bpcl" in d or "uber" in d or "ola" in d or "metro" in d:
        return "Fuel & Transport"
    if "airtel" in d or "vodafone" in d or "internet" in d or "postpaid" in d or "jio" in d:
        return "Mobile & Internet"
    if "subscription" in d or "netflix" in d or "prime video" in d or "hotstar" in d or "spotify" in d:
        return "Subscriptions"
    if "medical" in d or "pharmacy" in d or "hospital" in d or "dr." in d or "clinic" in d:
        return "Medical"
    if "credit card" in d or "cc pmt" in d:
        return "Credit Card Repayment"
    if "insurance" in d or "lic" in d:
        return "Insurance"
    
    # NEW CATEGORIES
    if "store" in d or "supermarket" in d or "grocer" in d or "reliance fresh" in d or "dmart" in d:
        return "Groceries"
    if "atm" in d or "cash withdrawal" in d:
        return "Cash Withdrawal"
    if "sip" in d or "mutual fund" in d or "nps" in d or "invest" in d:
        return "Investments"

    return "Others"


# -----------------------------
# MAIN ANALYZER
# -----------------------------

def analyze_statement(file_bytes, filename):
    """
    Main entry point called by FastAPI
    """

    # load file
    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            # Try to read as CSV, allowing for various delimiters if possible
            df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not load file: {e}")


    date_col, desc_col, amount_col, credit_col, debit_col = detect_columns(df)

    if not date_col or not desc_col:
        raise ValueError("Could not detect Date or Description columns.")
        
    # Drop rows that are entirely NaN/header-like before proceeding
    df.dropna(how='all', inplace=True)

    # Reorder/rename columns for internal consistency
    cols_to_keep = [date_col, desc_col] + [c for c in df.columns if c not in [date_col, desc_col]]
    df = df[cols_to_keep].copy()
    df.rename(columns={date_col: "Date", desc_col: "Description"}, inplace=True)

    # Date Cleaning
    df["Date"] = df["Date"].apply(smart_parse_date)
    # Drop rows where Date could not be successfully parsed (often non-transactional rows)
    df = df.dropna(subset=["Date"]) 
    df.sort_values(by="Date", inplace=True)
    df["Description"] = df["Description"].astype(str).fillna("").str.strip()

    # SIGNED AMOUNT (CRITICAL FIX APPLIED HERE)
    df["SignedAmount"] = build_signed_amount(df, amount_col, credit_col, debit_col)
    
    # Filter out zero transactions (e.g., failed transfers) and sort by date
    df = df[df["SignedAmount"].abs() > 0.01].copy()

    # CATEGORY
    df["Category"] = df["Description"].apply(classify_category)
    # Apply "Income" category for all positive amounts
    df.loc[df["SignedAmount"] > 0, "Category"] = "Income"
    
    # MONTH
    # Filter only necessary data for summary after cleaning.
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # -----------------------------
    # SUMMARY
    # -----------------------------
    
    # Inflow/Outflow are calculated on the entire filtered/cleaned dataset
    inflow = df[df["SignedAmount"] > 0]["SignedAmount"].sum()
    outflow = df[df["SignedAmount"] < 0]["SignedAmount"].abs().sum()
    net_savings = inflow - outflow

    # CURRENT MONTH
    monthly = df.groupby("Month")["SignedAmount"].sum().reset_index()
    monthly.columns = ["Month", "Savings"]
    monthly_sorted = monthly.sort_values("Month")

    this_month, this_savings, growth_text, safe_daily_spend = None, 0, "0", 0

    if len(monthly_sorted) > 0:
        current_row = monthly_sorted.iloc[-1]
        this_month = current_row["Month"]
        this_savings = current_row["Savings"]
        
        # Calculate growth vs last month
        prev_savings = (
            monthly_sorted.iloc[-2]["Savings"] if len(monthly_sorted) >= 2 else 0
        )
        growth = this_savings - prev_savings
        growth_text = f"{growth:,.0f} vs last month" # Format with comma
        
        # SAFE DAILY SPEND: Only calculate if savings are positive (or close to it)
        safe_daily_spend = max(int(this_savings / 30), 0)
        
    else:
        this_month, this_savings, growth_text = None, 0, "0"

    # Calculate average monthly outflow (NEW METRIC)
    num_months = len(monthly_sorted)
    avg_outflow = outflow / num_months if num_months > 0 else 0


    # -----------------------------
    # UPI DETECTION
    # -----------------------------
    upi_df = df[df["Description"].str.contains("upi", case=False, na=False)]
    upi_net = upi_df[upi_df["SignedAmount"] < 0]["SignedAmount"].abs().sum()

    # top UPI counterparty
    top_upi = (
        upi_df[upi_df["SignedAmount"] < 0] # Only look at outflows for counterparty analysis
        .groupby("Description")["SignedAmount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(5) # Limit to top 5
    )

    if len(top_upi) > 0:
        # Extract the description that has the highest outflow
        top_handle = top_upi.index[0] 
    else:
        top_handle = None

    # -----------------------------
    # EMI
    # -----------------------------
    # Use Category assigned EMI to be more robust than just checking description
    emi_df = df[df["Category"] == "EMI"] 
    emi_by_month = (
        emi_df.groupby("Month")["SignedAmount"].sum().abs().reset_index()
        if not emi_df.empty
        else pd.DataFrame(columns=["Month", "SignedAmount"])
    )
    
    emi_load = (
        emi_by_month[emi_by_month["Month"] == this_month]["SignedAmount"].sum()
        if this_month
        else 0
    )

    # -----------------------------
    # CATEGORY SUMMARY
    # -----------------------------
    # Only show outflow categories
    cat_summary = (
        df[df["SignedAmount"] < 0]
        .groupby("Category")["SignedAmount"]
        .sum()
        .abs()
        .reset_index()
        .rename(columns={"Category": "Category", "SignedAmount": "TotalAmount"})
        .sort_values(by="TotalAmount", ascending=False)
    )

    # -----------------------------
    # CLEANED PREVIEW
    # -----------------------------
    cleaned_preview = df[["Date", "Description", "SignedAmount", "Category"]].copy()
    cleaned_preview["Date"] = cleaned_preview["Date"].dt.strftime("%Y-%m-%d")

    cleaned_csv = cleaned_preview.to_csv(index=False)

    # -----------------------------
    # BUILD FINAL JSON
    # -----------------------------
    result = {
        "summary": {
            "total_inflow": round(inflow, 2),
            "total_outflow": round(outflow, 2),
            "savings_total": round(net_savings, 2),
            "current_month": this_month,
            "current_month_savings": round(this_savings, 2),
            "growth_text": growth_text,
            "upi_net_outflow": round(upi_net, 2),
            "emi_load": round(emi_load, 2),
            "safe_daily_spend": int(safe_daily_spend),
            "average_monthly_outflow": round(avg_outflow, 2), # NEW FIELD
        },
        "monthly": monthly_sorted.to_dict(orient="records"),
        "categories": cat_summary.to_dict(orient="records"),
        "upi": {
            "top_counterparties": [
                {"Description": i, "TotalAmount": round(float(v), 2)}
                for i, v in top_upi.items()
            ],
            "top_handle_description": top_handle
        },
        "emi": {
            "by_month": emi_by_month.to_dict(orient="records")
            if len(emi_by_month) > 0
            else []
        },
        "forecast": {
            "available": False
        },
        "cleaned_preview": cleaned_preview.to_dict(orient="records"),
        "cleaned_csv": cleaned_csv,
    }

    return result
