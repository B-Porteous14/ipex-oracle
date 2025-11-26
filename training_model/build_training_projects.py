import pandas as pd

# =========================================
# CONFIG – change this if your file is named differently
# =========================================
RAW_CSV_FILE = "raw_transactions.csv"   # or "fake_raw_transactions.csv"
OUTPUT_TRAINING_FILE = "training_projects.csv"


def load_and_fix_header(path):
    """
    Load the raw CSV and fix the header row if needed.
    If project_id is not a column name, we assume the first row contains
    the real column names (like in your template file).
    """
    df = pd.read_csv(path)

    if "project_id" not in df.columns:
        new_header = df.iloc[0]
        df = df[1:].copy()
        df.columns = new_header

    return df


def build_training_projects(df):
    """
    Take the raw transactions DataFrame and aggregate to one row per project
    with all the fields we want for training.
    """

    # ---- 1. Convert types ----
    # Dates (day comes first, e.g. 23/07/2021)
    date_cols = [
        "project_contract_start_date",
        "project_completion_date",
        "payment_date",
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Numbers
    df["project_value"] = pd.to_numeric(df["project_value"], errors="coerce")
    df["payment_amount"] = pd.to_numeric(df["payment_amount"], errors="coerce")

    # ---- 2. Aggregate to project level ----
    # A) project-level aggregates
    payments_agg = df.groupby("project_id").agg(
        planned_cost=("project_value", "first"),
        planned_finish_date=("project_completion_date", "first"),
        contract_start_date=("project_contract_start_date", "first"),
        actual_cost=("payment_amount", "sum"),
        actual_finish_date=("payment_date", "max"),
        num_payments=("payment_amount", "count"),
    ).reset_index()

    # B) trade-level aggregation (for num_trades and concentration)
    trade_payments = (
        df.groupby(["project_id", "trade_work_type"])["payment_amount"]
        .sum()
        .reset_index()
    )

    trade_agg = trade_payments.groupby("project_id").agg(
        num_trades=("trade_work_type", "nunique"),
        max_trade_paid=("payment_amount", "max"),
    ).reset_index()

    # ---- 3. Join project + trade aggregates ----
    projects = payments_agg.merge(trade_agg, on="project_id", how="left")

    # ---- 4. Compute derived fields (targets + features) ----
    # Overruns
    projects["cost_overrun_amount"] = projects["actual_cost"] - projects["planned_cost"]
    projects["cost_overrun_percent"] = (
        projects["cost_overrun_amount"] / projects["planned_cost"]
    )

    # Durations
    projects["planned_duration_days"] = (
        projects["planned_finish_date"] - projects["contract_start_date"]
    ).dt.days

    projects["actual_duration_days"] = (
        projects["actual_finish_date"] - projects["contract_start_date"]
    ).dt.days

    # Weeks late
    projects["weeks_late"] = (
        projects["actual_finish_date"] - projects["planned_finish_date"]
    ).dt.days / 7.0

    # Payment stats
    projects["avg_payment_amount"] = (
        projects["actual_cost"] / projects["num_payments"]
    )

    # Trade concentration
    projects["trade_cost_concentration"] = (
        projects["max_trade_paid"] / projects["actual_cost"]
    )

    # ---- 5. Select columns and order ----
    training_cols = [
        "project_id",
        "planned_cost",
        "actual_cost",
        "cost_overrun_amount",
        "cost_overrun_percent",
        "planned_finish_date",
        "actual_finish_date",
        "weeks_late",
        "planned_duration_days",
        "actual_duration_days",
        "num_payments",
        "avg_payment_amount",
        "num_trades",
        "max_trade_paid",
        "trade_cost_concentration",
    ]

    training_df = projects[training_cols].copy()
    return training_df


def main():
    print(f"Loading raw data from: {RAW_CSV_FILE}")
    df_raw = load_and_fix_header(RAW_CSV_FILE)

    print("Building training_projects table...")
    training_df = build_training_projects(df_raw)

    print(f"Saving training table to: {OUTPUT_TRAINING_FILE}")
    training_df.to_csv(OUTPUT_TRAINING_FILE, index=False)

    print("Done.")
    print("Rows in training_projects:", len(training_df))
    print("Columns:", list(training_df.columns))


if __name__ == "__main__":
    main()
