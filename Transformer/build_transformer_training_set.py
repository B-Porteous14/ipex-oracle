from pathlib import Path
import pandas as pd

# Base directory = the folder this script lives in
BASE_DIR = Path(__file__).resolve().parent

# Time-series created by build_transformer_timeseries.py
TS_FILE = BASE_DIR / "transformer_timeseries.csv"

# Project-level labels created by train_models.py
LABEL_FILE = BASE_DIR.parent / "training_model" / "training_projects_train.csv"

# Output training set for the transformer model
OUT_FILE = BASE_DIR / "transformer_training_data.csv"


def load_timeseries() -> pd.DataFrame:
    print(f"[transformer] Loading time-series: {TS_FILE}")
    return pd.read_csv(TS_FILE)


def load_labels() -> pd.DataFrame:
    print(f"[transformer] Loading labels: {LABEL_FILE}")
    return pd.read_csv(LABEL_FILE)


def build():
    print("[transformer] Starting build()")

    ts = load_timeseries()
    labels = load_labels()

    # Basic sanity checks
    if "project_id" not in ts.columns:
        raise ValueError("Time-series file must have a 'project_id' column.")
    if "project_id" not in labels.columns:
        raise ValueError("Label file must have a 'project_id' column.")

    # ---- FIX 1: Correct target column list ----
    TARGET_COLUMNS = ["weeks_late", "cost_overrun_percent"]

    # ---- FIX 2: refer to TARGET_COLUMNS correctly ----
    missing_targets = [c for c in TARGET_COLUMNS if c not in labels.columns]
    if missing_targets:
        raise ValueError(f"Label file is missing target columns: {missing_targets}")

    print("[transformer] Time-series shape:", ts.shape)
    print("[transformer] Label table shape:", labels.shape)

    # Only keep project_id + needed targets
    label_small = labels[["project_id"] + TARGET_COLUMNS]

    # Merge so each time step gets the project-level targets
    merged = ts.merge(label_small, on="project_id", how="inner")

    print("[transformer] Merged transformer training shape:", merged.shape)

    merged.to_csv(OUT_FILE, index=False)
    print(f"[transformer] Saved transformer training data to: {OUT_FILE}")


if __name__ == "__main__":
    build()
