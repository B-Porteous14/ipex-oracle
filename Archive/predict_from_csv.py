import joblib
import pandas as pd

# Load models
model_weeks = joblib.load("model_weeks_late.pkl")
model_overrun = joblib.load("model_overrun_percent.pkl")

CSV_FILE = "project_input.csv"  # <-- you will provide this file


def main():
    print(f"Reading project data from {CSV_FILE}...\n")

    # Read the project snapshot
    df = pd.read_csv(CSV_FILE)

    # The model expects exactly these columns:
    feature_cols = [
        "planned_cost",
        "planned_duration_days",
        "num_payments",
        "avg_payment_amount",
        "num_trades",
        "max_trade_paid",
        "trade_cost_concentration",
    ]

    # Extract the row
    features = df[feature_cols]

    # Predict
    pred_weeks = model_weeks.predict(features)[0]
    pred_overrun = model_overrun.predict(features)[0]

    # Output
    print("--- IPEX Oracle Prediction ---")
    print(f"Estimated delay: {pred_weeks:.2f} weeks")
    print(f"Estimated cost overrun: {pred_overrun * 100:.2f}%")
    print("---------------------------------")


if __name__ == "__main__":
    main()
