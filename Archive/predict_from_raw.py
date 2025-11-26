import os
import glob
import pandas as pd
import joblib

# Folder where you drop raw project exports
RAW_DIR = "raw_projects"

# Model files (already created by train_models.py)
MODEL_WEEKS_FILE = "model_weeks_late.pkl"
MODEL_OVERRUN_FILE = "model_overrun_percent.pkl"


def load_models():
    model_weeks = joblib.load(MODEL_WEEKS_FILE)
    model_overrun = joblib.load(MODEL_OVERRUN_FILE)
    return model_weeks, model_overrun


def choose_raw_file() -> str:
    """
    Look in RAW_DIR for .csv or .xlsx files.
    Let the user choose one by number.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(RAW_DIR, "*.csv"))
        + glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
    )

    if not files:
        print(f"\nNo raw files found in '{RAW_DIR}'.")
        print("→ Export a raw project from IPEX and save it as CSV or Excel")
        print(f"→ Then put it into the '{RAW_DIR}' folder and run this again.\n")
        return ""

    print("\nFound the following raw project files:")
    for i, path in enumerate(files, start=1):
        print(f"  {i}. {os.path.basename(path)}")

    while True:
        choice = input("\nType the number of the file you want to use (e.g. 1): ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(files):
            return files[idx - 1]
        else:
            print("Number out of range, try again.")


def build_features_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw transactions (one or more projects) into
    the features the model expects, using 'so far' data only.
    """

    # Fix header if the first row contains column names
    if "project_id" not in df_raw.columns:
        df_raw.columns = df_raw.iloc[0]
        df_raw = df_raw[1:].copy()

    # Convert columns
    date_cols = [
        "project_contract_start_date",
        "project_completion_date",
        "payment_date",
    ]
    for col in date_cols:
        df_raw[col] = pd.to_datetime(df_raw[col], dayfirst=True, errors="coerce")

    df_raw["project_value"] = pd.to_numeric(df_raw["project_value"], errors="coerce")
    df_raw["payment_amount"] = pd.to_numeric(df_raw["payment_amount"], errors="coerce")

    # Drop rows without payments
    df_raw = df_raw.dropna(subset=["payment_amount"])

    rows = []

    # One row per project_id
    for project_id, g in df_raw.groupby("project_id"):
        planned_cost = g["project_value"].iloc[0]
        contract_start = g["project_contract_start_date"].iloc[0]
        planned_finish = g["project_completion_date"].iloc[0]

        num_payments = g["payment_amount"].count()
        actual_cost_so_far = g["payment_amount"].sum()
        avg_payment = actual_cost_so_far / num_payments if num_payments > 0 else 0.0

        trade_totals = g.groupby("trade_work_type")["payment_amount"].sum()
        num_trades = trade_totals.count()
        max_trade_paid = trade_totals.max() if num_trades > 0 else 0.0
        trade_conc = (
            max_trade_paid / actual_cost_so_far if actual_cost_so_far > 0 else 0.0
        )

        if pd.isna(contract_start) or pd.isna(planned_finish):
            planned_duration_days = None
        else:
            planned_duration_days = (planned_finish - contract_start).days

        rows.append(
            {
                "project_id": project_id,
                "planned_cost": planned_cost,
                "planned_duration_days": planned_duration_days,
                "num_payments": num_payments,
                "avg_payment_amount": avg_payment,
                "num_trades": num_trades,
                "max_trade_paid": max_trade_paid,
                "trade_cost_concentration": trade_conc,
            }
        )

    features = pd.DataFrame(rows)
    features = features.dropna()  # simple clean
    return features


def main():
    print("\n=== IPEX Oracle – Predict from raw file ===")

    # Choose which raw file to use
    raw_path = choose_raw_file()
    if not raw_path:
        return

    print(f"\nUsing file: {raw_path}\n")

    # Load the raw file
    if raw_path.lower().endswith(".csv"):
        df_raw = pd.read_csv(raw_path)
    else:
        df_raw = pd.read_excel(raw_path)

    # Build features
    features = build_features_from_raw(df_raw)

    if features.empty:
        print("Could not build features from this file (no data after cleaning).")
        return

    # Load models
    model_weeks, model_overrun = load_models()

    feature_cols = [
        "planned_cost",
        "planned_duration_days",
        "num_payments",
        "avg_payment_amount",
        "num_trades",
        "max_trade_paid",
        "trade_cost_concentration",
    ]

    X = features[feature_cols]

    # Predict
    pred_weeks = model_weeks.predict(X)
    pred_overrun = model_overrun.predict(X)  # decimal (0.12 = 12%)

    # Show results per project
    print("\n--- Predictions ---")
    for i, row in features.iterrows():
        project_id = row["project_id"]
        w = pred_weeks[i]
        o = pred_overrun[i] * 100  # to %
        print(f"Project {project_id}:")
        print(f"  Estimated delay: {w:.2f} weeks")
        print(f"  Estimated cost overrun: {o:.2f}%")
        print("")

    print("Done.\n")


if __name__ == "__main__":
    main()
