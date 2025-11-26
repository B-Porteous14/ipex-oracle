import pandas as pd

df = pd.read_csv("Transaction data.csv")

test_projects = ["P0002", "P7945"]

for pid in test_projects:
    df_pid = df[df["project_id"] == pid]
    df_pid.to_csv(f"raw_projects/{pid}.csv", index=False)
    print(f"Saved raw_projects/{pid}.csv")
