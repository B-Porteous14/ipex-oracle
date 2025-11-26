import os
import glob
import pandas as pd
import joblib
import shap

# These are the feature columns the model was trained on
FEATURE_COLS = [
    "planned_cost",
    "planned_duration_days",
    "num_payments",
    "avg_payment_amount",
    "num_trades",
    "max_trade_paid",
    "trade_cost_concentration",
]

# Folder that holds your raw unfinished CSVs
RAW_FOLDER = "raw_projects"


def list_raw_files():
    """Return list of CSV files in the raw_projects folder."""
    pattern = os.path.join(RAW_FOLDER, "*.csv")
    files = glob.glob(pattern)
    return files


def choose_file(files):
    """Ask you to pick one file by number."""
    print("Available raw project files:\n")
    for i, path in enumerate(files, start=1):
        print(f"[{i}] {os.path.basename(path)}")

    print("")
    choice = int(input("Type the number of the file you want to predict: "))
    idx = choice - 1

    if idx < 0 or idx >= len(files):
        raise ValueError("Invalid choice")

    return files[idx]


def build_features_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw project CSV into ONE row of features
    that match the training data columns.
    """

    # ---- tweak column names here if your raw file uses slightly different ones ----

    # Contract value (planned cost) – same for all rows in the project
    planned_cost = df_raw["project_value"].iloc[0]

    # Planned duration in days
    contract_start = pd.to_datetime(
        df_raw["project_contract_start_date"].iloc[0],
        dayfirst=True,  # your dates look like dd/mm/yyyy
    )
    contract_finish = pd.to_datetime(
        df_raw["project_completion_date"].iloc[0],
        dayfirst=True,
    )
    planned_duration_days = (contract_finish - contract_start).days

    # Number of payment events
    num_payments = len(df_raw)

    # Payment stats
    avg_payment_amount = df_raw["payment_amount"].mean()

    # Trade stats
    num_trades = df_raw["trade_work_type"].nunique()

    # Total paid per trade
    per_trade = df_raw.groupby("trade_work_type")["payment_amount"].sum()
    max_trade_paid = per_trade.max()

    # Trade cost concentration (Herfindahl index: sum(p_i^2))
    total_paid = per_trade.sum()
    if total_paid > 0:
        shares = per_trade / total_paid  # fraction of total per trade
        trade_cost_concentration = float((shares ** 2).sum())
    else:
        trade_cost_concentration = 0.0

    # Build a one-row DataFrame with the exact columns the model expects
    data = {
        "planned_cost": [planned_cost],
        "planned_duration_days": [planned_duration_days],
        "num_payments": [num_payments],
        "avg_payment_amount": [avg_payment_amount],
        "num_trades": [num_trades],
        "max_trade_paid": [max_trade_paid],
        "trade_cost_concentration": [trade_cost_concentration],
    }

    features_df = pd.DataFrame(data, columns=FEATURE_COLS)
    return features_df


def build_shap_summary(shap_vals, X_row):
    """
    Turn raw SHAP values + feature values into a nice summary dict:
    {
      "increasing": [ {feature, value, impact}, ... ],
      "decreasing": [ ... ]
    }
    """
    rows = []
    for feature, value, shap_val in zip(FEATURE_COLS, X_row.values[0], shap_vals):
        rows.append((feature, float(value), float(shap_val)))

    # Sort by absolute impact
    rows_sorted = sorted(rows, key=lambda x: abs(x[2]), reverse=True)

    increasing = [
        {"feature": f, "value": v, "impact": s}
        for (f, v, s) in rows_sorted
        if s > 0
    ]
    decreasing = [
        {"feature": f, "value": v, "impact": s}
        for (f, v, s) in rows_sorted
        if s < 0
    ]

    # keep top 3 each for readability
    return {
        "increasing": increasing[:3],
        "decreasing": decreasing[:3],
    }


def print_clean_shap(title, prediction, shap_summary):
    """Prints a clean, human-readable explanation using the summary dict."""
    print(f"\n================ {title} ================")
    print(f"Predicted: {prediction:.2f}")

    print("\nTop factors increasing this prediction:")
    if not shap_summary["increasing"]:
        print("  (none)")
    else:
        for item in shap_summary["increasing"]:
            print(
                f"  • {item['feature']} ({item['value']:.2f}) → +{abs(item['impact']):.2f}"
            )

    print("\nTop factors decreasing this prediction:")
    if not shap_summary["decreasing"]:
        print("  (none)")
    else:
        for item in shap_summary["decreasing"]:
            print(
                f"  • {item['feature']} ({item['value']:.2f}) → -{abs(item['impact']):.2f}"
            )


def get_project_id(df_raw, fallback_name: str) -> str:
    """Try to get a project_id from the raw file; fall back to file name."""
    if "project_id" in df_raw.columns:
        return str(df_raw["project_id"].iloc[0])
    # fallback: strip extension from filename
    return os.path.splitext(os.path.basename(fallback_name))[0]


def main():
    # 1. Load models and SHAP explainers saved from train_models.py
    print("Loading models and SHAP explainers...")
    model_weeks = joblib.load("model_weeks_late.pkl")
    model_over = joblib.load("model_overrun_percent.pkl")
    explainer_weeks = joblib.load("explainer_weeks_late.pkl")
    explainer_over = joblib.load("explainer_overrun_percent.pkl")

    # 2. List raw project CSVs
    files = list_raw_files()
    if not files:
        print(f"No CSV files found in folder: {RAW_FOLDER}")
        return None

    # 3. Let you pick one
    chosen_path = choose_file(files)
    filename = os.path.basename(chosen_path)
    print(f"\nYou selected: {filename}")

    # 4. Load that raw CSV
    df_raw = pd.read_csv(chosen_path)

    # 5. Convert raw data -> model features
    X_row = build_features_from_raw(df_raw)

    print("\nFeature values used for this project:")
    print(X_row)

    # 6. Make predictions
    pred_weeks = float(model_weeks.predict(X_row)[0])
    pred_over = float(model_over.predict(X_row)[0])

    print("\n================ PREDICTIONS ================")
    print(f"Predicted weeks late:        {pred_weeks:.2f}")
    print(f"Predicted cost overrun (%):  {pred_over:.2f}")

    # 7. SHAP explanations for this row (raw values)
    shap_vals_w = explainer_weeks.shap_values(X_row)[0]
    shap_vals_o = explainer_over.shap_values(X_row)[0]

    # 8. Build SHAP summaries (for printing AND for API/JSON later)
    shap_summary_weeks = build_shap_summary(shap_vals_w, X_row)
    shap_summary_over = build_shap_summary(shap_vals_o, X_row)

    # 9. Print nice human-readable explanations
    print_clean_shap("Weeks Late Explanation", pred_weeks, shap_summary_weeks)
    print_clean_shap("Cost Overrun Explanation", pred_over, shap_summary_over)

    # 10. Build the risk_summary object (this is what FastAPI will return later)
    project_id = get_project_id(df_raw, filename)

    risk_summary = {
        "project_id": project_id,
        "source_file": filename,
        "predictions": {
            "weeks_late": pred_weeks,
            "cost_overrun_percent": pred_over,
        },
        "drivers": {
            "weeks_late": shap_summary_weeks,
            "cost_overrun_percent": shap_summary_over,
        },
    }

    return risk_summary


if __name__ == "__main__":
    summary = main()
    if summary is not None:
        print("\n\n================ RISK SUMMARY OBJECT ================")
        print(summary)
