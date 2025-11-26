import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# Helper: simple evaluation
# ============================================================

def evaluate_model(model, X_test, y_test, label: str):
    """Evaluate regression model on test set."""
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n----- Evaluation for {label} -----")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² score: {r2:.4f}")
    print("---------------------------------\n")

    return rmse, r2


# ============================================================
# Helper: parse money strings like "$1,234.56" or "-$1,000.00"
# ============================================================

def parse_money(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan

    neg = s.startswith("-")
    s_clean = s.replace("-", "").replace("$", "").replace(",", "")
    try:
        val = float(s_clean)
    except ValueError:
        return np.nan

    return -val if neg else val


# ============================================================
# Build transaction-level training dataset
# ============================================================

def build_transaction_training_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Reads the raw Transaction data.csv and returns a transaction-level
    training dataframe with:
      - many rows per project (each payment / trade snapshot)
      - engineered features
      - targets: weeks_late, cost_overrun_percent (per project, copied to rows)
    """
    print(f"Loading raw transactions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Basic parsing ---
    df["payment_amount_num"] = df["payment_amount"].apply(parse_money)
    df["project_value_num"] = df["project_value"].apply(parse_money)
    df["payee_contract_value_num"] = df["payee_contract_value"].apply(parse_money)

    df["payment_date_parsed"] = pd.to_datetime(
        df["payment_date"], dayfirst=True, errors="coerce"
    )
    df["project_contract_start_parsed"] = pd.to_datetime(
        df["project_contract_start_date"], dayfirst=True, errors="coerce"
    )
    df["project_completion_parsed"] = pd.to_datetime(
        df["project_completion_date"], dayfirst=True, errors="coerce"
    )
    df["payee_contract_start_parsed"] = pd.to_datetime(
        df["payee_contract_start_date"], dayfirst=True, errors="coerce"
    )
    df["payee_contract_completion_parsed"] = pd.to_datetime(
        df["payee_contract_completion_date"], dayfirst=True, errors="coerce"
    )

    # --- Filter to rows with a project_id, payment, and payment date ---
    df = df[df["project_id"].notna()].copy()
    df = df[df["payment_amount_num"].notna()].copy()
    df = df[df["payment_date_parsed"].notna()].copy()

    # --- Completed projects only ---
    completed_mask = df["Status"].str.contains("completed", case=False, na=False)
    completed_projects = (
        df.loc[completed_mask, "project_id"].dropna().unique().tolist()
    )
    print(f"Found {len(completed_projects)} completed projects: {completed_projects}")

    df = df[df["project_id"].isin(completed_projects)].copy()

    # --- Build project-level targets and constants ---
    proj_info = {}
    for pid in completed_projects:
        g = df[df["project_id"] == pid].copy()

        # project value
        pv = g["project_value_num"].dropna()
        if pv.empty:
            continue
        project_value = pv.iloc[0]
        if project_value <= 0:
            continue

        # planned dates
        starts = g["project_contract_start_parsed"].dropna()
        finishes = g["project_completion_parsed"].dropna()
        if starts.empty or finishes.empty:
            continue

        planned_start = starts.iloc[0]
        planned_finish = finishes.iloc[0]
        planned_duration_days = max((planned_finish - planned_start).days, 1)

        # sort by date and compute cumulative *positive* payments
        g = g.sort_values("payment_date_parsed")
        g["payment_cost"] = g["payment_amount_num"].clip(lower=0.0)
        g["cum_pos_paid"] = g["payment_cost"].cumsum()

        # define "practical completion" as first time cumulative paid >= 95% of contract
        threshold = 0.95 * project_value
        pc_rows = g[g["cum_pos_paid"] >= threshold]

        if not pc_rows.empty:
            pc_row = pc_rows.iloc[0]
        else:
            # fallback: use last payment row
            pc_row = g.iloc[-1]

        pc_date = pc_row["payment_date_parsed"]
        total_paid_for_target = float(pc_row["cum_pos_paid"])

        # ---- Target 1: schedule delay in weeks (using pc_date) ----
        weeks_late = max((pc_date - planned_finish).days / 7.0, 0.0)

        # ---- Target 2: cost overrun using EVM-style target at pc_date ----
        days_elapsed = max((pc_date - planned_start).days, 0)
        pct_time_elapsed = days_elapsed / planned_duration_days

        if project_value > 0:
            expected_cost = project_value * min(pct_time_elapsed, 1.0)
            cost_variance = total_paid_for_target - expected_cost
            cost_overrun_percent = (cost_variance / project_value) * 100.0
        else:
            cost_overrun_percent = 0.0

        proj_info[pid] = {
            "project_value_num": project_value,
            "planned_start": planned_start,
            "planned_finish": planned_finish,
            "planned_duration_days": planned_duration_days,
            "weeks_late": weeks_late,
            "cost_overrun_percent": cost_overrun_percent,
        }

    # Keep only rows where we managed to build proj_info
    valid_pids = set(proj_info.keys())
    df = df[df["project_id"].isin(valid_pids)].copy()
    print(f"Using {len(valid_pids)} projects with clean targets.")

    # --- Attach project-level info to each transaction row ---
    df["project_value_num2"] = df["project_id"].map(
        lambda pid: proj_info[pid]["project_value_num"]
    )
    df["project_start"] = df["project_id"].map(
        lambda pid: proj_info[pid]["planned_start"]
    )
    df["project_finish_planned"] = df["project_id"].map(
        lambda pid: proj_info[pid]["planned_finish"]
    )
    df["project_duration_days2"] = df["project_id"].map(
        lambda pid: proj_info[pid]["planned_duration_days"]
    )
    df["target_weeks_late"] = df["project_id"].map(
        lambda pid: proj_info[pid]["weeks_late"]
    )
    df["target_cost_overrun"] = df["project_id"].map(
        lambda pid: proj_info[pid]["cost_overrun_percent"]
    )

    # --- Project time features ---
    df["project_day"] = (
        df["payment_date_parsed"] - df["project_start"]
    ).dt.days
    df["pct_time_elapsed"] = (
        df["project_day"] / df["project_duration_days2"]
    ).clip(lower=0, upper=1)

    # sort for cumulative stuff
    df = df.sort_values(
        ["project_id", "payment_date_parsed", "payee_id", "trade_work_type"]
    )

    # use *positive* payments only for cumulative features
    df["payment_cost"] = df["payment_amount_num"].clip(lower=0.0)

    # --- Project-level cumulative paid ---
    df["cum_project_paid"] = (
        df.groupby("project_id")["payment_cost"]
          .transform(lambda s: s.cumsum())
    )
    df["pct_project_paid"] = df["cum_project_paid"] / df["project_value_num2"]

    # --- Trade-level cumulative + schedule ---
    group_keys = ["project_id", "payee_id", "trade_work_type"]
    df["cum_trade_paid"] = (
        df.groupby(group_keys)["payment_cost"]
          .transform(lambda s: s.cumsum())
    )
    # avoid divide by zero
    df.loc[df["payee_contract_value_num"] == 0, "payee_contract_value_num"] = np.nan
    df["pct_trade_paid"] = df["cum_trade_paid"] / df["payee_contract_value_num"]

    # gap since last trade/project payment
    df["days_since_last_trade_payment"] = (
        df.groupby(group_keys)["payment_date_parsed"]
          .diff()
          .dt.days
          .fillna(0)
    )
    df["days_since_last_project_payment"] = (
        df.groupby("project_id")["payment_date_parsed"]
          .diff()
          .dt.days
          .fillna(0)
    )

    # trade schedule timing
    df["trade_start"] = df["payee_contract_start_parsed"]
    df["trade_finish"] = df["payee_contract_completion_parsed"]
    trade_duration = (df["trade_finish"] - df["trade_start"]).dt.days
    df["trade_duration_days"] = trade_duration.where(trade_duration > 0, np.nan)
    df["trade_day"] = (df["payment_date_parsed"] - df["trade_start"]).dt.days
    df["pct_trade_time_elapsed"] = (
        df["trade_day"] / df["trade_duration_days"]
    ).clip(lower=0, upper=1)
    df.loc[df["trade_duration_days"].isna(), "pct_trade_time_elapsed"] = np.nan

    df["trade_schedule_lag"] = (
        df["pct_trade_time_elapsed"] - df["pct_trade_paid"]
    ).clip(lower=0)

    # trade value share
    df["trade_value_share"] = (
        df["payee_contract_value_num"] / df["project_value_num2"]
    )
    df["is_large_trade"] = (df["trade_value_share"] >= 0.15).astype(float)

    # Drop any rows missing targets
    df = df.dropna(
        subset=["target_weeks_late", "target_cost_overrun"]
    ).copy()

    print(f"Final transaction-level training rows: {len(df)}")

    return df


# ============================================================
# Main training routine
# ============================================================

def main():
    RAW_CSV = "Transaction data_with_synthetic.csv"
  # put this in the same folder as this script

    # 1) Build transaction-level dataset
    data = build_transaction_training_dataframe(RAW_CSV)

    # 2) Define feature columns
    feature_cols = [
        "project_value_num2",
        "project_duration_days2",
        "project_day",
        "pct_time_elapsed",
        "payment_amount_num",
        "cum_project_paid",
        "pct_project_paid",
        "cum_trade_paid",
        "pct_trade_paid",
        "trade_value_share",
        "is_large_trade",
        "days_since_last_trade_payment",
        "days_since_last_project_payment",
        "trade_duration_days",
        "pct_trade_time_elapsed",
        "trade_schedule_lag",
    ]

    # 3) Train/test split by project_id (not random rows)
    all_projects = sorted(data["project_id"].unique().tolist())
    train_projects, test_projects = train_test_split(
        all_projects, test_size=2, random_state=42
    )

    print(f"\nTraining projects ({len(train_projects)}): {train_projects}")
    print(f"Testing  projects ({len(test_projects)}): {test_projects}")

    train_df = data[data["project_id"].isin(train_projects)].copy()
    test_df = data[data["project_id"].isin(test_projects)].copy()

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Testing  rows: {len(test_df)}")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_weeks = train_df["target_weeks_late"]
    y_test_weeks = test_df["target_weeks_late"]

    y_train_cost = train_df["target_cost_overrun"]
    y_test_cost = test_df["target_cost_overrun"]

    # 4) Train models
    print("\nTraining model for WEEKS LATE...")
    model_weeks = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model_weeks.fit(X_train, y_train_weeks)

    print("\nTraining model for COST OVERRUN (EVM-style, cleaned payments)...")
    model_cost = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model_cost.fit(X_train, y_train_cost)

    # 5) Evaluate (transaction-level)
    evaluate_model(model_weeks, X_test, y_test_weeks, "weeks_late (transaction-level)")
    evaluate_model(model_cost, X_test, y_test_cost, "cost_overrun_percent (EVM-style, cleaned)")

    # 6) Save models
    joblib.dump(model_weeks, "model_weeks_late.pkl")
    joblib.dump(model_cost, "model_overrun_percent.pkl")
    print("Saved XGBoost models: model_weeks_late.pkl, model_overrun_percent.pkl")

    # 7) Build and save SHAP explainers
    try:
        import shap

        print("\nFitting SHAP explainers (this can take a little while)...")
        explainer_weeks = shap.TreeExplainer(model_weeks)
        explainer_cost = shap.TreeExplainer(model_cost)

        joblib.dump(explainer_weeks, "explainer_weeks_late.pkl")
        joblib.dump(explainer_cost, "explainer_overrun_percent.pkl")

        print("Saved SHAP explainers: explainer_weeks_late.pkl, explainer_overrun_percent.pkl")
    except ImportError:
        print(
            "\n[WARNING] shap is not installed. "
            "Install it with `pip install shap` if you want fresh SHAP explainers."
        )

    print("\nAll done. Models and explainers are ready for ml_core.py")


if __name__ == "__main__":
    main()
