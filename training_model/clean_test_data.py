import pandas as pd

# Test projects you want to clean
test_projects = ["P0002", "P7945"]

for pid in test_projects:
    path = f"raw_projects/{pid}.csv"
    df = pd.read_csv(path)

    # Parse the completion date once per project
    comp = pd.to_datetime(df["project_completion_date"].dropna().iloc[0], dayfirst=True)

    # Parse payment dates
    df["payment_date_parsed"] = pd.to_datetime(df["payment_date"], dayfirst=True)

    # Filter only payments up to completion date
    df_clean = df[df["payment_date_parsed"] <= comp].copy()

    # Drop helper columns
    df_clean = df_clean.drop(columns=["payment_date_parsed"])

    # Save cleaned version
    df_clean.to_csv(f"raw_projects/{pid}_CLEAN.csv", index=False)
    print(f"Saved cleaned file: raw_projects/{pid}_CLEAN.csv (from {len(df)} → {len(df_clean)} rows)")
