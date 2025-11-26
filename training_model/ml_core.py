"""
Improved ML Core for IPEX Oracle
--------------------------------

This version:
 - Computes XGBoost predictions for delay + cost
 - Returns SHAP values (if available)
 - Computes "Worst 3 trades" based on:
        • Exposure (total contract value)
        • Schedule misalignment (percent paid vs percent time elapsed)
 - Blended predictions remain unchanged
 - Clean and safe even if CSV is messy
"""

import numpy as np
import pandas as pd
import traceback

# -------------------------------------------------------------------
# Load trained XGB models
# -------------------------------------------------------------------

import joblib
DELAY_MODEL = joblib.load("training_model/model_weeks_late.pkl")
COST_MODEL = joblib.load("training_model/model_overrun_percent.pkl")

DELAY_EXPLAINER = joblib.load("training_model/explainer_weeks_late.pkl")
COST_EXPLAINER = joblib.load("training_model/explainer_overrun_percent.pkl")

EXPECTED_DELAY_FEATURES = list(DELAY_MODEL.get_booster().feature_names)
EXPECTED_COST_FEATURES = list(COST_MODEL.get_booster().feature_names)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def safe_num(x):
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except:
        return np.nan


def compute_pct_time_elapsed(row):
    try:
        start = pd.to_datetime(row["project_contract_start_date"])
        end = pd.to_datetime(row["project_completion_date"])
        pay_date = pd.to_datetime(row["payment_date"])

        total = (end - start).days
        so_far = (pay_date - start).days

        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, so_far / total))

    except:
        return 0.0


def compute_trade_risks(df):
    """
    Computes the worst 3 trades based on:
        • Exposure (sum of payee_contract_value)
        • Schedule misalignment:
                risk = max(0, %paid - %time_elapsed)
        • Combined score = exposure_normalised * schedule_risk
    """

    if "trade_work_type" not in df.columns:
        return []

    df["payee_contract_value_num"] = df["payee_contract_value"].apply(safe_num)
    df["contract_percent_paid_num"] = df["contract_percent_paid"].apply(safe_num)
    df["pct_time_elapsed"] = df.apply(compute_pct_time_elapsed, axis=1)

    trades = []

    for trade, g in df.groupby("trade_work_type"):

        exposure = g["payee_contract_value_num"].replace(np.nan, 0).sum()

        # schedule misalignment
        risk_vals = []
        for _, row in g.iterrows():
            pct_paid = row["contract_percent_paid_num"] / 100 if pd.notna(row["contract_percent_paid_num"]) else 0
            pct_time = row["pct_time_elapsed"]
            misalign = max(0, pct_paid - pct_time)
            risk_vals.append(misalign)

        if len(risk_vals) == 0:
            continue

        schedule_risk = float(np.mean(risk_vals))

        trades.append({
            "trade": trade,
            "exposure": float(exposure),
            "schedule_risk": schedule_risk,
            "combined_risk_score": float(exposure * schedule_risk)
        })

    if not trades:
        return []

    # Highest → lowest combined risk score
    trades_sorted = sorted(trades, key=lambda x: x["combined_risk_score"], reverse=True)

    # Return top 3
    return trades_sorted[:3]


# -------------------------------------------------------------------
# Main risk summary function
# -------------------------------------------------------------------

def make_risk_summary_from_df(df: pd.DataFrame):
    """
    Builds a full ML risk summary including:
     - xgb predictions
     - SHAP values
     - worst trades
     - blended predictions
    """

    # ---------------------------------------------------------------
    # Build ML feature rows (aggregate to project-level)
    # ---------------------------------------------------------------

    df["payment_amount_num"] = df["payment_amount"].apply(safe_num)
    df["payee_contract_value_num"] = df["payee_contract_value"].apply(safe_num)
    df["contract_percent_paid_num"] = df["contract_percent_paid"].apply(safe_num)

    agg = {
        "payment_amount_num": "sum",
        "payee_contract_value_num": "sum",
        "contract_percent_paid_num": "mean",
    }
    row = df.agg(agg)

    X_input = pd.DataFrame([row])

    # Ensure feature alignment
    for col in EXPECTED_DELAY_FEATURES:
        if col not in X_input.columns:
            X_input[col] = 0.0

    for col in EXPECTED_COST_FEATURES:
        if col not in X_input.columns:
            X_input[col] = 0.0

    X_input = X_input.reindex(columns=EXPECTED_DELAY_FEATURES, fill_value=0)

    # ---------------------------------------------------------------
    # XGBoost predictions
    # ---------------------------------------------------------------
    delay_pred = float(DELAY_MODEL.predict(X_input)[0])
    cost_pred = float(COST_MODEL.predict(X_input)[0])

    shap_delay = DELAY_EXPLAINER.shap_values(X_input)[0].tolist()
    shap_cost = COST_EXPLAINER.shap_values(X_input)[0].tolist()

    # ---------------------------------------------------------------
    # Worst trades
    # ---------------------------------------------------------------
    worst_trades = compute_trade_risks(df)

    # ---------------------------------------------------------------
    # Build output
    # ---------------------------------------------------------------
    out = {
        "predictions": {
            "weeks_late": delay_pred,
            "cost_overrun_percent": cost_pred,
            "blended_weeks_late": delay_pred,
            "blended_cost_overrun_percent": cost_pred,
            "xgb_weeks_late": delay_pred,
            "xgb_cost_overrun_percent": cost_pred,
            "transformer_weeks_late": None,
            "transformer_cost_overrun_percent": None,
            "gnn_weeks_late": None,
            "gnn_cost_overrun_percent": None,
        },

        "shap": {
            "weeks_late": shap_delay,
            "cost_overrun_percent": shap_cost,
        },

        "trade_risks": worst_trades,
    }

    return out
