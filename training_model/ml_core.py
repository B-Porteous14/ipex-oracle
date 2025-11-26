"""
ml_core.py – Unified risk prediction engine for the IPEX Oracle.

This module:
  • Loads XGBoost delay + cost models
  • Loads SHAP explainers
  • Runs the Transformer model
  • Runs the GNN model
  • Produces final blended predictions
  • Builds clean JSON for the API layer
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------
# Load GNN
# -----------------------------
from training_model.gnn.inference_gnn import predict_risk_gnn

"""Core risk engine: XGBoost + Transformer + GNN.

This version expects the existing Transformer helper API from
training_model.transformer_head: predict_transformer_from_df and
TRANSFORMER_AVAILABLE.
"""

# -----------------------------
# Load Transformer helpers
# -----------------------------
from training_model.transformer_head import (
    predict_transformer_from_df,
    TRANSFORMER_AVAILABLE,
    parse_money,
)

# -----------------------------
# Model paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DELAY = BASE_DIR / "model_weeks_late.pkl"
MODEL_COST = BASE_DIR / "model_overrun_percent.pkl"

EXPLAINER_DELAY = BASE_DIR / "explainer_weeks_late.pkl"
EXPLAINER_COST = BASE_DIR / "explainer_overrun_percent.pkl"


# -----------------------------
# Safe loader for models / explainers
# -----------------------------

def _safe_load_joblib(path: Path, label: str):
    """
    Try to load a joblib artefact. If the file is missing or broken,
    log a warning and return None instead of crashing the server.

    This is important for cloud deployments where .pkl files might not
    yet be present. The rest of the code must handle the None case.
    """
    try:
        if not path.exists():
            print(f"[ml_core] WARNING: {label} not found at {path}. "
                  f"Using placeholder behaviour (zero predictions).")
            return None
        obj = joblib.load(path)
        print(f"[ml_core] Loaded {label} from {path}")
        return obj
    except Exception as e:
        print(f"[ml_core] ERROR loading {label} from {path}: {e}. "
              f"Using placeholder behaviour (zero predictions).")
        return None


# -----------------------------
# Load XGBoost + SHAP explainers
# -----------------------------
print("[ml_core] Loading XGBoost models and SHAP explainers…")

# Models and explainers were originally saved with joblib, so we must
# load them the same way here. If they are missing in the container,
# the safe loader will return None and downstream code will fall back
# to dummy predictions instead of crashing.

model_weeks = _safe_load_joblib(MODEL_DELAY, "MODEL_DELAY (weeks_late)")
model_cost = _safe_load_joblib(MODEL_COST, "MODEL_COST (overrun_percent)")

expl_weeks = _safe_load_joblib(EXPLAINER_DELAY, "EXPLAINER_DELAY (weeks_late)")
expl_cost = _safe_load_joblib(EXPLAINER_COST, "EXPLAINER_COST (overrun_percent)")


# ============================================================================
# FEATURE ENGINEERING – MUST MATCH YOUR TRAINING DATA
# ============================================================================

# Original training feature columns for the XGBoost models. The models and
# SHAP explainers expect these exact names in this order.
FEATURE_COLS = [
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


def build_xgb_features_for_project(df_project: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the XGBoost training features for a single project.

    This ports the core feature engineering logic from
    train_models.build_transaction_training_dataframe, restricted to one
    project_id. Returns a DataFrame with columns FEATURE_COLS.
    """

    if df_project.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    df = df_project.copy()

    # Ensure project_id exists as string
    if "project_id" not in df.columns:
        df["project_id"] = "UNKNOWN"
    df["project_id"] = df["project_id"].astype(str)

    # --- Basic parsing ---
    if "payment_amount_num" not in df.columns:
        df["payment_amount_num"] = df.get("payment_amount", np.nan).apply(parse_money)

    if "project_value_num" not in df.columns:
        df["project_value_num"] = df.get("project_value", np.nan).apply(parse_money)

    if "payee_contract_value_num" not in df.columns:
        df["payee_contract_value_num"] = df.get("payee_contract_value", np.nan).apply(parse_money)

    df["payment_date_parsed"] = pd.to_datetime(
        df.get("payment_date"), dayfirst=True, errors="coerce"
    )
    df["project_contract_start_parsed"] = pd.to_datetime(
        df.get("project_contract_start_date"), dayfirst=True, errors="coerce"
    )
    df["project_completion_parsed"] = pd.to_datetime(
        df.get("project_completion_date"), dayfirst=True, errors="coerce"
    )
    df["payee_contract_start_parsed"] = pd.to_datetime(
        df.get("payee_contract_start_date"), dayfirst=True, errors="coerce"
    )
    df["payee_contract_completion_parsed"] = pd.to_datetime(
        df.get("payee_contract_completion_date"), dayfirst=True, errors="coerce"
    )

    # Filter to rows with payment and date
    df = df[df["payment_amount_num"].notna()].copy()
    df = df[df["payment_date_parsed"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    # --- Project-level info (single project) ---
    pid = df["project_id"].iloc[0]

    pv = df["project_value_num"].dropna()
    if pv.empty:
        return pd.DataFrame(columns=FEATURE_COLS)
    project_value = pv.iloc[0]
    if project_value <= 0:
        return pd.DataFrame(columns=FEATURE_COLS)

    starts = df["project_contract_start_parsed"].dropna()
    finishes = df["project_completion_parsed"].dropna()
    if starts.empty or finishes.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    planned_start = starts.iloc[0]
    planned_finish = finishes.iloc[0]
    planned_duration_days = max((planned_finish - planned_start).days, 1)

    # Sort and compute cumulative positive payments
    df = df.sort_values("payment_date_parsed")
    df["payment_cost"] = df["payment_amount_num"].clip(lower=0.0)

    # Define "practical completion" as first time cumulative paid >= 95% of contract
    df["cum_pos_paid"] = df["payment_cost"].cumsum()
    threshold = 0.95 * project_value
    pc_rows = df[df["cum_pos_paid"] >= threshold]
    if not pc_rows.empty:
        pc_row = pc_rows.iloc[0]
    else:
        pc_row = df.iloc[-1]

    pc_date = pc_row["payment_date_parsed"]

    # --- Attach project-level info to each transaction row ---
    df["project_value_num2"] = project_value
    df["project_start"] = planned_start
    df["project_finish_planned"] = planned_finish
    df["project_duration_days2"] = planned_duration_days

    # Project time features
    df["project_day"] = (df["payment_date_parsed"] - df["project_start"]).dt.days
    df["pct_time_elapsed"] = (
        df["project_day"] / df["project_duration_days2"]
    ).clip(lower=0, upper=1)

    # Project-level cumulative paid (positive payments only)
    df["cum_project_paid"] = df["payment_cost"].cumsum()
    df["pct_project_paid"] = df["cum_project_paid"] / df["project_value_num2"]

    # Trade-level cumulative + schedule
    group_keys = ["payee_id", "trade_work_type"]
    for k in group_keys:
        if k not in df.columns:
            df[k] = "UNKNOWN"

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
        df["payment_date_parsed"].diff().dt.days.fillna(0)
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

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLS].fillna(0.0)


# ============================================================================
# PREDICTION HELPERS
# ============================================================================


def _safe_float(x: object, default: float = 0.0) -> float:
    """Convert to float, treating None / invalid as a default."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def predict_xgb(df_features: pd.DataFrame):
    """Predict using XGBoost for weeks_late and cost_overrun_percent.

    df_features must already contain FEATURE_COLS.
    """

    # If we have no features or no models loaded (e.g. missing .pkl files
    # in the cloud container), fall back to zeros instead of crashing.
    if df_features.empty or model_weeks is None or model_cost is None:
        print("[ml_core] predict_xgb: models not available or no features; "
              "returning zeros.")
        return 0.0, 0.0

    df_input = df_features[FEATURE_COLS]

    pred_weeks = model_weeks.predict(df_input)
    pred_cost = model_cost.predict(df_input)

    # Aggregate row-level predictions to a single project value
    weeks = _safe_float(np.median(pred_weeks)) if len(pred_weeks) else 0.0
    cost = _safe_float(np.median(pred_cost)) if len(pred_cost) else 0.0

    return weeks, cost


def explain_xgb(df_features: pd.DataFrame):
    """SHAP values for LLM explanations (using original FEATURE_COLS)."""

    if df_features.empty or expl_weeks is None or expl_cost is None:
        print("[ml_core] explain_xgb: explainers not available or no features; "
              "returning zero SHAP values.")
        return np.zeros(len(FEATURE_COLS)), np.zeros(len(FEATURE_COLS))

    df_input = df_features[FEATURE_COLS]

    shap_weeks = expl_weeks.shap_values(df_input)
    shap_cost = expl_cost.shap_values(df_input)

    # Use the SHAP values from the last row as representative
    return shap_weeks[-1], shap_cost[-1]


# ============================================================================
# TRADE RISK SCORING – TOP 3 TRADES
# ============================================================================


def calculate_simple_trade_risks(df_project: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute a simple risk score for each trade and return top 3.

    This is a lightweight version of the earlier trade_risks logic:
      - value_share: contract value vs project value
      - payment_progress: pct_contract_paid
      - schedule_lag: behind vs pct_project_paid

    Returns a list of dicts sorted by descending risk_score.
    """

    required_cols = {
        "project_value",
        "payee_id",
        "trade_work_type",
        "payee_contract_value",
        "payment_amount",
        "payment_date",
    }
    if not required_cols.issubset(df_project.columns):
        return []

    df = df_project.copy()

    # Parse numeric values
    df["payment_amount_num"] = df["payment_amount"].apply(parse_money)
    df["project_value_num"] = df["project_value"].apply(parse_money)
    df["payee_contract_value_num"] = df["payee_contract_value"].apply(parse_money)

    df["payment_date_parsed"] = pd.to_datetime(
        df["payment_date"], dayfirst=True, errors="coerce"
    )

    # Filter to usable rows
    df = df[df["payment_amount_num"].notna()].copy()
    df = df[df["payment_date_parsed"].notna()].copy()
    if df.empty:
        return []

    pv = df["project_value_num"].dropna()
    if pv.empty:
        return []
    project_value = float(pv.iloc[0])
    if project_value <= 0:
        return []

    df["payment_cost"] = df["payment_amount_num"].clip(lower=0.0)

    trade_rows: List[Dict[str, Any]] = []

    grouped = df.groupby(["payee_id", "trade_work_type"])
    for (payee_id, trade), g in grouped:
        try:
            contract_vals = g["payee_contract_value_num"].dropna()
            if contract_vals.empty:
                continue
            contract_value = float(contract_vals.iloc[0])
            if contract_value <= 0:
                continue

            total_paid = float(g["payment_cost"].sum())
            pct_contract_paid = total_paid / contract_value if contract_value > 0 else 0.0

            value_share = contract_value / project_value

            # Rough schedule lag proxy: compare pct_contract_paid vs overall pct_project_paid
            df_proj = df.copy()
            df_proj_sorted = df_proj.sort_values("payment_date_parsed")
            df_proj_sorted["cum_project_paid"] = df_proj_sorted["payment_cost"].cumsum()
            df_proj_sorted["pct_project_paid"] = (
                df_proj_sorted["cum_project_paid"] / project_value
            )
            latest = df_proj_sorted.iloc[-1]
            pct_project_paid = float(latest["pct_project_paid"])

            schedule_lag = max(pct_project_paid - pct_contract_paid, 0.0)

            # Simple composite risk score
            risk_score = 0.6 * schedule_lag + 0.4 * value_share

            trade_rows.append(
                {
                    "payee_id": str(payee_id),
                    "trade": str(trade),
                    "risk_score": float(risk_score),
                    "metrics": {
                        "contract_value": float(contract_value),
                        "total_paid": float(total_paid),
                        "pct_contract_paid": float(pct_contract_paid),
                        "value_share_of_project": float(value_share),
                        "pct_project_paid_overall": float(pct_project_paid),
                        "schedule_lag_proxy": float(schedule_lag),
                    },
                }
            )
        except Exception:
            continue

    trade_rows_sorted = sorted(trade_rows, key=lambda r: r["risk_score"], reverse=True)
    return trade_rows_sorted[:3]


# ============================================================================
# MAIN PROJECT SUMMARY BUILDER
# ============================================================================

def make_risk_summary_from_df(df_project: pd.DataFrame):
    """
    Takes one project's raw transactions dataframe.
    Runs all prediction engines.
    Returns a clean JSON-ready dict.
    """

    # Build full XGBoost feature matrix for this project
    df_xgb = build_xgb_features_for_project(df_project)

    # If we cannot build any features, return a minimal zeroed summary
    if df_xgb.empty:
        project_id = str(df_project.get("project_id", ["unknown"])[0])
        return {
            "project_id": project_id,
            "predictions": {
                "weeks_late": 0.0,
                "cost_overrun_percent": 0.0,
                "xgb_weeks_late": 0.0,
                "xgb_cost_overrun_percent": 0.0,
                "transformer_weeks_late": None,
                "transformer_cost_overrun_percent": None,
                "gnn_weeks_late": None,
                "gnn_cost_overrun_percent": None,
            },
        }

    # XGBoost
    xgb_weeks, xgb_cost = predict_xgb(df_xgb)

    # Transformer – use existing helper if available
    tr_weeks, tr_cost = None, None
    if TRANSFORMER_AVAILABLE:
        try:
            project_id = str(df_project["project_id"].iloc[0]) if "project_id" in df_project.columns else "unknown"
            tr_preds = predict_transformer_from_df(df_project, project_id)
            if tr_preds is not None:
                tr_weeks = _safe_float(tr_preds.get("weeks_late", 0.0))
                tr_cost = _safe_float(tr_preds.get("cost_overrun_percent", 0.0))
                print(f"[ml_core] Transformer used for project {project_id}: weeks={tr_weeks:.2f}, cost={tr_cost:.2f}")
            else:
                print(f"[ml_core] Transformer skipped for project {project_id} (insufficient or invalid time-series data)")
        except Exception as e:
            print(f"[ml_core] Transformer failed: {e}")

    # GNN is currently disabled for production blending. We keep the fields
    # in the output schema but do not use them in the blended values.
    gnn_weeks, gnn_cost = None, None

    # BLENDING: XGBoost + Transformer only.
    # If Transformer is available for this project, use a simple 70/30 fusion.
    if tr_weeks is not None and tr_cost is not None:
        blended_weeks = 0.7 * _safe_float(xgb_weeks) + 0.3 * _safe_float(tr_weeks)
        blended_cost = 0.7 * _safe_float(xgb_cost) + 0.3 * _safe_float(tr_cost)
    else:
        blended_weeks = _safe_float(xgb_weeks)
        blended_cost = _safe_float(xgb_cost)

    # SHAP
    shap_weeks, shap_cost = explain_xgb(df_xgb)

    # Trade risks (top 3 trades by simple risk score)
    trade_risks = calculate_simple_trade_risks(df_project)

    # JSON OUTPUT PACKAGE
    return {
        "project_id": df_project.get("project_id", ["unknown"])[0],
        "predictions": {
            "weeks_late": _safe_float(blended_weeks),
            "cost_overrun_percent": _safe_float(blended_cost),

            # explicit blended fields for clarity
            "blended_weeks_late": _safe_float(blended_weeks),
            "blended_cost_overrun_percent": _safe_float(blended_cost),

            # raw model outputs
            "xgb_weeks_late": _safe_float(xgb_weeks),
            "xgb_cost_overrun_percent": _safe_float(xgb_cost),

            "transformer_weeks_late": None if tr_weeks is None else _safe_float(tr_weeks),
            "transformer_cost_overrun_percent": None if tr_cost is None else _safe_float(tr_cost),

            "gnn_weeks_late": None if gnn_weeks is None else _safe_float(gnn_weeks),
            "gnn_cost_overrun_percent": None if gnn_cost is None else _safe_float(gnn_cost),
        },
        "shap": {
            "weeks_late": shap_weeks.tolist(),
            "cost_overrun_percent": shap_cost.tolist(),
        },
        "trade_risks": trade_risks,
    }


# ============================================================================
# BATCH HANDLER
# ============================================================================

def batch_predict(df_all: pd.DataFrame):
    """
    Runs all projects through the prediction engine.
    Returns a list of JSON dictionaries.
    """
    out = []
    for pid, group in df_all.groupby("project_id"):
        out.append(make_risk_summary_from_df(group))
    return out
