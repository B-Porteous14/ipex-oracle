"""
build_transformer_timeseries.py

Creates per-project payment time-series for the transformer model.
Takes raw transaction-level data and outputs a clean, numerical,
chronologically ordered dataset.

INPUT:  ../training_model/Transaction data_with_synthetic.csv
OUTPUT: ./transformer_timeseries.csv
"""

import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
from pathlib import Path

RAW_FILE = Path("Transaction data_with_synthetic.csv")
OUT_FILE = Path("transformer_timeseries.csv")


COL_PROJECT_ID = "project_id"
COL_PAYMENT_DATE = "payment_date"
COL_PAYMENT_AMT = "payment_amount"
COL_PROJ_VALUE = "project_value"


# ---------- HELPERS ----------
def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Convert '$', commas, blanks, etc → float."""
    return (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(" ", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )


# ---------- MAIN ----------
def build_timeseries():
    print("[transformer] Starting build_timeseries()")
    print(f"[transformer] Loading CSV: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    print(f"[transformer] Raw shape: {df.shape}")

    use_cols = [COL_PROJECT_ID, COL_PAYMENT_DATE, COL_PAYMENT_AMT, COL_PROJ_VALUE]
    for c in use_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    df = df[use_cols].copy()

    # ---- Clean numeric ----
    df[COL_PAYMENT_AMT] = clean_numeric_column(df[COL_PAYMENT_AMT])
    df[COL_PROJ_VALUE] = clean_numeric_column(df[COL_PROJ_VALUE])

    # ---- Drop rows missing essential fields ----
    before = len(df)
    df = df.dropna(subset=[COL_PROJECT_ID, COL_PAYMENT_DATE])
    print(f"[transformer] Removed {before - len(df)} rows missing project_id or payment_date")

    # ---- Parse dates ----
    df[COL_PAYMENT_DATE] = pd.to_datetime(df[COL_PAYMENT_DATE], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[COL_PAYMENT_DATE])
    print(f"[transformer] Removed {before - len(df)} invalid dates")

    # ---- Sort ----
    df = df.sort_values([COL_PROJECT_ID, COL_PAYMENT_DATE])

    # ---- Time index per project ----
    df["t"] = df.groupby(COL_PROJECT_ID).cumcount()

    # ---- Days since first payment ----
    first_date = df.groupby(COL_PROJECT_ID)[COL_PAYMENT_DATE].transform("min")
    df["days_since_first_payment"] = (df[COL_PAYMENT_DATE] - first_date).dt.days

    # ---- Cumulative + pct paid ----
    df["cum_project_paid"] = df.groupby(COL_PROJECT_ID)[COL_PAYMENT_AMT].cumsum()
    df["pct_project_paid"] = df["cum_project_paid"] / df[COL_PROJ_VALUE].replace(0, pd.NA)

    # ---- Output final CSV ----
    out_cols = [
        COL_PROJECT_ID,
        "t",
        "days_since_first_payment",
        COL_PAYMENT_AMT,
        "cum_project_paid",
        "pct_project_paid",
        COL_PROJ_VALUE,
    ]

    df_out = df[out_cols].dropna().reset_index(drop=True)

    print(f"[transformer] Final time-series shape: {df_out.shape}")
    df_out.to_csv(OUT_FILE, index=False)
    print(f"[transformer] Saved to: {OUT_FILE.resolve()}")
    print("[transformer] Done!")


if __name__ == "__main__":
    build_timeseries()
