# test_ml_core.py

import pandas as pd
from ml_core import make_risk_summary_from_df

# 1) Load your big transaction file
df_all = pd.read_csv("Transaction data_with_synthetic.csv")

# 2) Pick a real project_id that you know exists
test_project_id = "P0002"   # <- change to any valid ID from the CSV

df_proj = df_all[df_all["project_id"] == test_project_id].copy()

print(f"Rows for {test_project_id}:", len(df_proj))

# 3) Call the brain
summary = make_risk_summary_from_df(df_proj, source_id=f"{test_project_id}_test")

print("\n=== predictions ===")
for k, v in summary["predictions"].items():
    print(f"{k}: {v}")

print("\n=== risk_buckets ===")
print(summary["risk_buckets"])

print("\n=== transformer availability ===")
print("transformer_weeks_late:", summary["predictions"]["transformer_weeks_late"])
print("transformer_cost_overrun_percent:", summary["predictions"]["transformer_cost_overrun_percent"])

print("\n=== blended vs xgb-only ===")
print("weeks_late (xgb-only):", summary["predictions"]["weeks_late"])
print("blended_weeks_late:", summary["predictions"]["blended_weeks_late"])
print("cost_overrun_percent (xgb-only):", summary["predictions"]["cost_overrun_percent"])
print("blended_cost_overrun_percent:", summary["predictions"]["blended_cost_overrun_percent"])
