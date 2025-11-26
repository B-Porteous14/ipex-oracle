"""
Full ML core for IPEX Oracle.

- Uses XGBoost models for weeks-late and cost-overrun predictions.
- Uses SHAP explainers for feature attributions (when available).
- Is robust to:
    * missing feature columns
    * non-numeric values in feature columns
    * missing model / explainer files in the deployed environment

Public API (used by FastAPI app):

    make_risk_summary_from_df(df_project) -> dict
    batch_predict(df_all) -> list[dict]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import logging

import joblib
import numpy as np
import pandas as pd

# Optional dependencies (SHAP / torch etc.) – imported lazily where needed
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

logger = logging.getLogger("ml_core")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Paths to model artefacts
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DELAY_PATH = BASE_DIR / "model_weeks_late.pkl"
MODEL_COST_PATH = BASE_DIR / "model_overrun_percent.pkl"

EXPLAINER_DELAY_PATH = BASE_DIR / "explainer_weeks_late.pkl"
EXPLAINER_COST_PATH = BASE_DIR / "explainer_overrun_percent.pkl"

# (Transformer artefacts – wired later if needed)
TRANSFORMER_MODEL_PATH = BASE_DIR / "transformer_model.pt"
TRANSFORMER_NORM_STATS_PATH = BASE_DIR / "transformer_norm_stats.npz"


# ---------------------------------------------------------------------------
# Feature lists used when the XGBoost models were trained
# (must match training_projects_train.csv / training code)
# ---------------------------------------------------------------------------

XGB_DELAY_FEATURES: List[str] = [
    "project_value_num2",
    "project_duration_days2",
    "project_day",
    "pct_time_elapsed",
    "payment_amount_num",
    "cum_project_paid",
    "pct_project_paid",
    "trade_value_share",
    "is_large_trade",
    "days_since_last_trade_payment",
    "trade_duration_days",
    "pct_trade_time_elapsed",
    "pct_trade_paid",
    "trade_schedule_lag",
    "planned_cost",
    "planned_duration_days",
    "num_payments",
    "avg_payment_amount",
    "num_trades",
    "max_trade_paid",
    "trade_cost_concentration",
]

# For now cost model uses the same feature set; if you trained with a
# slightly different set, edit this list accordingly.
XGB_COST_FEATURES: List[str] = list(XGB_DELAY_FEATURES)


# ---------------------------------------------------------------------------
# Load models and explainers (with safety)
# ---------------------------------------------------------------------------

def _safe_load(path: Path, description: str) -> Any:
    if not path.exists():
        logger.warning("[ml_core] %s file missing: %s", description, path)
        return None
    try:
        obj = joblib.load(path)
        logger.info("[ml_core] Loaded %s from %s", description, path.name)
        return obj
    except Exception as exc:  # pragma: no cover
        logger.exception("[ml_core] Failed to load %s from %s: %s", description, path, exc)
        return None


model_weeks = _safe_load(MODEL_DELAY_PATH, "XGB weeks-late model")
model_cost = _safe_load(MODEL_COST_PATH, "XGB cost-overrun model")

explainer_weeks = _safe_load(EXPLAINER_DELAY_PATH, "SHAP weeks-late explainer") if shap else None
explainer_cost = _safe_load(EXPLAINER_COST_PATH, "SHAP cost-overrun explainer") if shap else None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ensure_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Coerce the specified columns to numeric, with errors -> NaN -> 0.0.
    This avoids "Could not convert string '...$104.70-...'" errors.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Only fillna on columns that actually exist
    existing = [c for c in cols if c in df.columns]
    if existing:
        df[existing] = df[existing].fillna(0.0)
    return df


def _build_feature_matrix(
    df_project: pd.DataFrame, feature_names: List[str], label: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a DataFrame X with *exactly* the columns in feature_names:
    - any missing columns are added and filled with 0.0
    - all columns are coerced to numeric
    - column order is fixed

    Returns (X, missing_features).
    """
    df = df_project.copy()

    # Coerce any existing feature columns to numeric
    df = _ensure_numeric_columns(df, feature_names)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        logger.warning(
            "[ml_core] %s model: input missing %d feature(s): %s – filling with 0.0",
            label,
            len(missing),
            ", ".join(missing),
        )
        for col in missing:
            df[col] = 0.0

    # Ensure correct order and only the expected columns
    X = df.reindex(columns=feature_names)
    return X, missing


def _predict_with_xgb(
    model: Any,
    explainer: Any,
    df_project: pd.DataFrame,
    feature_names: List[str],
    label: str,
) -> Tuple[float, List[float]]:
    """
    Run an XGBoost model + SHAP (if available) on a single-project DataFrame.

    Returns (prediction, shap_values_list).
    If anything fails, returns (0.0, []) and logs a warning.
    """
    if model is None:
        logger.warning("[ml_core] %s model not loaded – returning 0.0", label)
        return 0.0, []

    try:
        X, _ = _build_feature_matrix(df_project, feature_names, label)
        # XGBoost returns np.ndarray; we expect a single-row prediction
        y_pred = float(model.predict(X)[0])

        shap_vals: List[float] = []
        if explainer is not None and shap is not None:
            try:
                # Some explainers return (n_samples, n_features)
                sv = explainer.shap_values(X)
                # Ensure we handle both list-of-arrays and single array forms
                if isinstance(sv, list):
                    sv_arr = np.array(sv[0])
                else:
                    sv_arr = np.array(sv)
                shap_vals = sv_arr[0].astype(float).tolist()
            except Exception as exc:  # pragma: no cover
                logger.warning("[ml_core] %s SHAP failed: %s", label, exc)

        return y_pred, shap_vals

    except Exception as exc:  # pragma: no cover
        logger.warning("[ml_core] %s model predict failed: %s", label, exc)
        return 0.0, []


def _build_explanation(
    weeks_late: float,
    cost_overrun_pct: float,
) -> Tuple[str, List[str]]:
    """
    Very simple rules-based explanation and actions based on the blended
    XGBoost predictions. You can tweak the thresholds later.
    """
    explanation_lines: List[str] = []
    actions: List[str] = []

    # Schedule risk
    if weeks_late <= 1:
        explanation_lines.append(
            "The project is forecast to be close to the planned completion date."
        )
    elif weeks_late <= 4:
        explanation_lines.append(
            "The project shows a moderate schedule risk with a small forecast delay."
        )
        actions.append("Investigate trades with recent schedule slippage and confirm recovery plans.")
    else:
        explanation_lines.append(
            "The project shows a high schedule risk with a significant forecast delay."
        )
        actions.append("Escalate schedule risk and consider re-sequencing or adding resources.")

    # Cost risk
    if cost_overrun_pct <= 2:
        explanation_lines.append(
            "Cost performance is forecast to be close to the original budget."
        )
    elif cost_overrun_pct <= 10:
        explanation_lines.append(
            "The project is forecast to have a moderate cost overrun."
        )
        actions.append("Review contingency, variations and at-risk trades for additional cost controls.")
    else:
        explanation_lines.append(
            "The project is forecast to have a significant cost overrun."
        )
        actions.append("Escalate cost risk and review major contracts, scope and claims in detail.")

    if not actions:
        actions.append(
            "Continue to monitor progress and costs, but no major risk intervention is recommended at this stage."
        )

    explanation_text = " ".join(explanation_lines)
    return explanation_text, actions


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def make_risk_summary_from_df(df_project: pd.DataFrame) -> Dict[str, Any]:
    """
    Takes one project's feature row(s) and returns a JSON-ready dictionary
    with predictions, SHAP values, and simple explanation text.

    df_project may contain multiple rows, but we assume it is already
    aggregated to a single project; if not, we use the first row.
    """
    if df_project.empty:
        raise ValueError("df_project is empty")

    # If multiple rows, take the first aggregated row
    df_project = df_project.iloc[[0]].copy()

    # Try to find a project_id if present
    project_id = None
    for col in ("project_id", "Project_ID", "project_code"):
        if col in df_project.columns:
            project_id = str(df_project[col].iloc[0])
            break
    if project_id is None:
        project_id = "unknown"

    # --- XGBoost predictions ---
    weeks_xgb, shap_weeks = _predict_with_xgb(
        model_weeks,
        explainer_weeks,
        df_project,
        XGB_DELAY_FEATURES,
        label="delay",
    )

    cost_xgb, shap_cost = _predict_with_xgb(
        model_cost,
        explainer_cost,
        df_project,
        XGB_COST_FEATURES,
        label="cost",
    )

    # For now we don't have transformer / GNN models wired in this file;
    # keep them as None so the API schema remains stable.
    transformer_weeks = None
    transformer_cost = None
    gnn_weeks = None
    gnn_cost = None

    # Blended metrics – for now just use XGB; you can later combine multiple models
    blended_weeks = weeks_xgb
    blended_cost = cost_xgb

    explanation_text, actions = _build_explanation(
        weeks_late=blended_weeks,
        cost_overrun_pct=blended_cost,
    )

    summary: Dict[str, Any] = {
        "project_id": project_id,
        "predictions": {
            "weeks_late": float(blended_weeks),
            "cost_overrun_percent": float(blended_cost),
            "blended_weeks_late": float(blended_weeks),
            "blended_cost_overrun_percent": float(blended_cost),
            "xgb_weeks_late": float(weeks_xgb),
            "xgb_cost_overrun_percent": float(cost_xgb),
            "transformer_weeks_late": transformer_weeks,
            "transformer_cost_overrun_percent": transformer_cost,
            "gnn_weeks_late": gnn_weeks,
            "gnn_cost_overrun_percent": gnn_cost,
        },
        "shap": {
            "weeks_late": shap_weeks,
            "cost_overrun_percent": shap_cost,
        },
        "trade_risks": [],  # future: per-trade risk outputs
        "explanation_text": explanation_text,
        "recommended_actions": actions,
        "llm_used": False,
        "llm_error": "OPENAI_API_KEY environment variable is not set.",
        "source_id": df_project.get("source_id", pd.Series([None])).iloc[0] or f"{project_id}.csv",
    }

    return summary


def batch_predict(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Batch version: group by project_id (if present) and run
    make_risk_summary_from_df for each project.
    """
    outputs: List[Dict[str, Any]] = []

    if "project_id" not in df_all.columns:
        # Treat entire frame as a single project
        outputs.append(make_risk_summary_from_df(df_all))
        return outputs

    for pid, group in df_all.groupby("project_id"):
        try:
            outputs.append(make_risk_summary_from_df(group))
        except Exception as exc:  # pragma: no cover
            logger.warning("[ml_core] batch_predict failed for project %s: %s", pid, exc)

    return outputs
