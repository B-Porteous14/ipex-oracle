import joblib
import pandas as pd

# Load the trained models
model_weeks = joblib.load("model_weeks_late.pkl")
model_overrun = joblib.load("model_overrun_percent.pkl")


def main():
    print("=== IPEX Oracle – Project Risk Prediction ===\n")

    # ---- Get inputs from the user ----
    # For a live project, use what you know so far.
    planned_cost = float(input("Planned total project cost ($): "))
    planned_duration_days = float(input("Planned duration (days): "))

    num_payments = float(input("Number of payments made so far: "))
    avg_payment_amount = float(
        input("Average payment amount so far ($): ")
    )

    num_trades = float(input("Number of trades/subcontract packages: "))
    max_trade_paid = float(
        input("Largest total paid to a single trade so far ($): ")
    )

    trade_cost_concentration = float(
        input("Trade cost concentration (0–1, e.g. 0.3 if one trade is ~30% of spend): ")
    )

    # ---- Build a one-row DataFrame in the same format as training ----
    features = pd.DataFrame([{
        "planned_cost": planned_cost,
        "planned_duration_days": planned_duration_days,
        "num_payments": num_payments,
        "avg_payment_amount": avg_payment_amount,
        "num_trades": num_trades,
        "max_trade_paid": max_trade_paid,
        "trade_cost_concentration": trade_cost_concentration,
    }])

    # ---- Make predictions ----
    pred_weeks_late = model_weeks.predict(features)[0]
    pred_overrun_pct = model_overrun.predict(features)[0]  # this is a decimal (e.g. 0.12 = 12%)

    # ---- Show results ----
    print("\n--- IPEX Oracle Prediction ---")
    print(f"Estimated delay: {pred_weeks_late:.2f} weeks")
    print(f"Estimated cost overrun: {pred_overrun_pct * 100:.2f}%")
    print("---------------------------------")


if __name__ == "__main__":
    main()
