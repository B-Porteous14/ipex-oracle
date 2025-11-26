import pandas as pd
import numpy as np
from datetime import timedelta


INPUT_CSV = "Transaction data.csv"
OUTPUT_CSV = "Transaction data_with_synthetic.csv"

# How many synthetic copies per completed project
NUM_SYNTH_PER_PROJECT = 5


def parse_money(value):
    """Parse money strings like '$1,234.56' or '-$500.00' to float."""
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


def format_money(x: float):
    """Format a float back to your '$1,234.56' style strings."""
    if pd.isna(x):
        return np.nan
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def to_date(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def from_date(dt: pd.Timestamp):
    if pd.isna(dt):
        return np.nan
    return dt.strftime("%d/%m/%Y")


def make_synthetic_project(group: pd.DataFrame, synth_idx: int) -> pd.DataFrame:
    """
    Take all rows for a single real project and create one synthetic copy.
    We:
      - give it a new project_id
      - scale all money values by a random factor
      - shift dates by a random offset
      - jitter payment dates slightly
    """
    g = group.copy()

    real_pid = str(g["project_id"].dropna().iloc[0])
    new_pid = f"{real_pid}_S{synth_idx}"

    # ---- money scaling factor (log-normal-ish around 1.0) ----
    scale = np.random.normal(1.0, 0.2)
    scale = float(np.clip(scale, 0.6, 1.6))

    # ---- parse money columns ----
    money_cols = [
        "project_value",
        "payee_contract_value",
        "payment_amount",
        "retention_balance",
    ]
    for col in money_cols:
        if col in g.columns:
            g[col + "_num"] = g[col].apply(parse_money)

    # ---- parse dates ----
    date_cols = [
        "project_contract_start_date",
        "project_completion_date",
        "payee_contract_start_date",
        "payee_contract_completion_date",
        "payee_defects_end_date",
        "payment_date",
    ]
    for col in date_cols:
        if col in g.columns:
            g[col + "_dt"] = to_date(g[col])

    # Global project offset in days (-60 to +60)
    global_shift_days = int(np.random.randint(-60, 61))

    # Small jitter for each payment (-7 to +7 days)
    per_payment_jitter = np.random.randint(-7, 8, size=len(g))

    # ---- scale money ----
    for col in money_cols:
        if col in g.columns:
            num_col = col + "_num"
            if num_col in g.columns:
                g[num_col] = g[num_col] * scale
                g[col] = g[num_col].apply(format_money)

    # ---- shift dates ----
    for col in ["project_contract_start_date", "project_completion_date",
                "payee_contract_start_date", "payee_contract_completion_date",
                "payee_defects_end_date"]:
        dt_col = col + "_dt"
        if dt_col in g.columns:
            g[col] = g[dt_col].apply(
                lambda d: from_date(d + timedelta(days=global_shift_days)) if pd.notna(d) else np.nan
            )

    # payment_date: global shift + jitter per row, then re-sort
    if "payment_date_dt" in g.columns:
        shifted_dates = []
        for d, jitter in zip(g["payment_date_dt"], per_payment_jitter):
            if pd.isna(d):
                shifted_dates.append(np.nan)
            else:
                shifted_dates.append(d + timedelta(days=global_shift_days + int(jitter)))
        g["payment_date_dt_synth"] = shifted_dates
        g["payment_date"] = g["payment_date_dt_synth"].apply(from_date)

        # re-sort by synthetic payment_date (to keep time-series logic sensible)
        g = g.sort_values("payment_date_dt_synth").reset_index(drop=True)

    # ---- update project_id and status ----
    g["project_id"] = new_pid

    if "Status" in g.columns:
        # everything in synthetic set is "Completed" (for training)
        g["Status"] = "Completed (synthetic)"

    # Drop helper columns
    drop_cols = [c for c in g.columns if c.endswith("_num") or c.endswith("_dt") or c.endswith("_dt_synth")]
    g = g.drop(columns=drop_cols, errors="ignore")

    return g


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Filter to completed projects (case-insensitive)
    status_col = "Status"
    completed_mask = df[status_col].astype(str).str.contains("completed", case=False, na=False)
    df_completed = df[completed_mask & df["project_id"].notna()].copy()

    completed_projects = sorted(df_completed["project_id"].dropna().unique().tolist())
    print(f"Found {len(completed_projects)} completed projects: {completed_projects}")

    synth_rows = []

    for pid in completed_projects:
        g = df_completed[df_completed["project_id"] == pid].copy()
        # Some files have header / spacer rows with NaNs; drop them gently
        g = g[g["payment_date"].notna() | g["payment_amount"].notna()]

        for i in range(1, NUM_SYNTH_PER_PROJECT + 1):
            synth = make_synthetic_project(g, synth_idx=i)
            synth_rows.append(synth)

    if synth_rows:
        df_synth_all = pd.concat(synth_rows, ignore_index=True)
        print(f"Generated {len(df_synth_all)} synthetic rows "
              f"({len(df_synth_all['project_id'].unique())} synthetic projects).")
    else:
        df_synth_all = pd.DataFrame(columns=df.columns)
        print("No synthetic rows generated – check filters.")

    # Combine real data + synthetic data
    df_out = pd.concat([df, df_synth_all], ignore_index=True)

    print(f"Saving combined real + synthetic data to {OUTPUT_CSV} "
          f"({len(df_out)} rows)...")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
