import pandas as pd
import numpy as np


def clean_currency(series: pd.Series) -> pd.Series:
    """
    Convert strings like "$115,510,211.00" to numeric.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,]", "", regex=True),
        errors="coerce",
    )


def build_training_from_transactions(path_in: str, path_out: str = "training_projects.csv") -> None:
    # 1. Load raw transaction data
    df = pd.read_csv(path_in, on_bad_lines="skip")

    # Normalise status text
    df["Status_clean"] = df["Status"].astype(str).str.strip().str.lower()

    # 2. Find completed projects (any row with status complete/completed)
    completed_projects = (
        df.loc[df["Status_clean"].isin(["complete", "completed"]), "project_id"]
        .dropna()
        .unique()
    )

    print(f"Total rows in file: {len(df)}")
    print(f"Completed project IDs found: {len(completed_projects)}")
    print("Completed project_ids:", completed_projects)

    if len(completed_projects) == 0:
        print("No completed projects found. Exiting.")
        return

    # Limit to all rows belonging to completed projects
    df_completed = df[df["project_id"].isin(completed_projects)].copy()

    # 3. Pre-clean commonly used columns
    df_completed["project_value_num"] = clean_currency(df_completed["project_value"])
    df_completed["payment_amount_num"] = clean_currency(df_completed["payment_amount"])

    df_completed["project_contract_start_date"] = pd.to_datetime(
        df_completed["project_contract_start_date"], dayfirst=True, errors="coerce"
    )
    df_completed["project_completion_date"] = pd.to_datetime(
        df_completed["project_completion_date"], dayfirst=True, errors="coerce"
    )
    df_completed["payment_date"] = pd.to_datetime(
        df_completed["payment_date"], dayfirst=True, errors="coerce"
    )

    rows = []

    # 4. Aggregate per project_id
    for project_id, g in df_completed.groupby("project_id"):
        g = g.copy()

        # Planned dates
        start_dates = g["project_contract_start_date"].dropna()
        finish_dates = g["project_completion_date"].dropna()

        if start_dates.empty or finish_dates.empty:
            print(f"[SKIP] {project_id}: missing planned dates")
            continue

        contract_start = start_dates.iloc[0]
        planned_finish = finish_dates.iloc[0]

        # Actual completion proxy = last payment date
        payment_dates = g["payment_date"].dropna()
        if payment_dates.empty:
            print(f"[SKIP] {project_id}: no payment dates")
            continue
        actual_finish = payment_dates.max()

        planned_duration_days = max((planned_finish - contract_start).days, 1)

        # Planned cost
        planned_costs = g["project_value_num"].dropna()
        if planned_costs.empty:
            print(f"[SKIP] {project_id}: no numeric project_value")
            continue
        planned_cost = float(planned_costs.iloc[0])

        # Actual cost = sum of all payments
        payment_amounts = g["payment_amount_num"].fillna(0.0)
        total_paid = float(payment_amounts.sum())

        # Targets
        days_late = (actual_finish - planned_finish).days
        weeks_late = days_late / 7.0

        if planned_cost > 0:
            cost_overrun_percent = (total_paid - planned_cost) / planned_cost * 100.0
        else:
            cost_overrun_percent = np.nan

        # Basic payment features
        num_payments = int((payment_amounts > 0).sum())
        avg_payment_amount = float(
            payment_amounts[payment_amounts > 0].mean() if num_payments > 0 else 0.0
        )

        # Trade features
        num_trades = int(g["trade_work_type"].dropna().nunique())

        per_trade_paid = (
            g.assign(payment_amount_num=payment_amounts)
            .groupby("trade_work_type")["payment_amount_num"]
            .sum()
        )
        per_trade_paid = per_trade_paid[per_trade_paid > 0]

        if len(per_trade_paid) > 0:
            max_trade_paid = float(per_trade_paid.max())
            trade_shares = per_trade_paid / per_trade_paid.sum()
            trade_cost_concentration = float((trade_shares ** 2).sum())
        else:
            max_trade_paid = 0.0
            trade_cost_concentration = 0.0

        row = {
            "project_id": project_id,
            "planned_cost": planned_cost,
            "planned_duration_days": int(planned_duration_days),
            "num_payments": num_payments,
            "avg_payment_amount": avg_payment_amount,
            "num_trades": num_trades,
            "max_trade_paid": max_trade_paid,
            "trade_cost_concentration": trade_cost_concentration,
            "weeks_late": weeks_late,
            "cost_overrun_percent": cost_overrun_percent,
        }

        rows.append(row)

    training_df = pd.DataFrame(rows)

    # Drop any rows with missing targets
    training_df = training_df.dropna(subset=["weeks_late", "cost_overrun_percent"])

    print("\nBuilt training_projects.csv with shape:", training_df.shape)
    print(training_df.head())

    training_df.to_csv(path_out, index=False)
    print(f"\nSaved training data to {path_out}")


if __name__ == "__main__":
    build_training_from_transactions("Transaction data.csv", "training_projects.csv")
