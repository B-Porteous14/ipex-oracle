import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# -------------------------------------------------
# 1. LOAD YOUR CSV
# -------------------------------------------------
df = pd.read_csv("data.csv")

# Some exports from your system have a first row like:
# "IPEX data fields, Unnamed:1, Unnamed:2, ..."
# and the REAL column names (project_id, project_value, etc.)
# are actually in the FIRST DATA ROW.
# This block fixes that automatically.
if "project_id" not in df.columns:
    # Use the first row as the header
    df.columns = df.iloc[0]
    df = df[1:]

print("Columns in the file:")
print(df.columns)

# -------------------------------------------------
# 2. CONVERT DATE COLUMNS
# -------------------------------------------------
# Your planned completion column is called project_completion_date
df["project_completion_date"] = pd.to_datetime(df["project_completion_date"])
df["payment_date"] = pd.to_datetime(df["payment_date"])

# -------------------------------------------------
# 3. AGGREGATE TO PROJECT LEVEL
#    Each project_id can have many payment rows.
#    We want ONE ROW per project.
# -------------------------------------------------
projects = df.groupby("project_id").agg({
    "project_completion_date": "first",   # planned finish date from contract
    "payment_date": "max",                # last payment date = actual finish
    "project_value": "first"              # planned cost
}).reset_index()

# Rename last payment date so it's clear
projects.rename(columns={"payment_date": "actual_completion_date"}, inplace=True)

# -------------------------------------------------
# 4. CREATE TARGET: DELAYED (1) OR NOT (0)
# -------------------------------------------------
projects["delayed"] = (
    projects["actual_completion_date"] > projects["project_completion_date"]
).astype(int)

print("\nProject summary (first few rows):")
print(projects[["project_id", "project_completion_date",
                "actual_completion_date", "delayed"]].head())

print("\nNumber of projects:", len(projects))

# -------------------------------------------------
# 5. CHECK IF WE EVEN HAVE ENOUGH DATA TO TRAIN
#    (Your current file might just be a structure example.)
# -------------------------------------------------
if len(projects) < 5 or projects["delayed"].nunique() == 1:
    print("\n👉 You don't have enough real projects yet to train a model.")
    print("   Once you have several completed projects (some delayed, some not),")
    print("   save them into data.csv and run this script again.")
    raise SystemExit

# -------------------------------------------------
# 6. VERY SIMPLE MODEL (XGBoost) USING PROJECT VALUE ONLY
#    Later we will add more features.
# -------------------------------------------------
projects["project_value"] = pd.to_numeric(projects["project_value"], errors="coerce")

X = projects[["project_value"]]  # features (for now just project_value)
y = projects["delayed"]          # target (0 or 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
