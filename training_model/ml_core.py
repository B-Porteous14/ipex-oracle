"""
ML core for IPEX Oracle.

This version will:

- Try to load the real XGBoost models and SHAP explainers:
    * model_weeks_late.pkl
    * model_overrun_percent.pkl
    * explainer_weeks_late.pkl
    * explainer_overrun_percent.pkl
- Build a simple numeric feature vector from the project transactions and
  feed it into the models.
- Fall back to the old "dummy" behaviour (0 predictions, empty SHAP)
  if the models / explainers are missing or anything errors.

NOTE:
For truly accurate predictions, you should eventually update
`_build_feature_vector` so that it *exactly* matches the feature
engineering used when you trained the XGBoost models (e.g. whatever
you did in `train_delay_model.py`). Right now it just uses a small set
of generic stats and pads the rest with zeros.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import shap  # noqa: F401
except ImportError:  # pragma: no cover
    shap = None  # type: ignore


LOGGER = logging.getLogger("ipex_ml_core")
LOGGER.setLevel(logging.INFO)


# -------------------------------------------------------------------
# Paths to model artefacts
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_DELAY_PATH = BASE_DIR / "model_weeks_late.pkl"
MODEL_COST_PATH = BASE_DIR / "model_overrun_percent.pkl"
EXPL_DELAY_PATH = BASE_DIR / "explainer_weeks_late.pkl"
EXPL_COST_PATH = BASE_DIR / "explainer_overrun_percent.pkl"

# Transformer / GNN are not wired in yet – placeholders only.
# TRANSFORMER_MODEL_PATH = BASE_DIR / "transformer_model.pt"
# TRANSFORMER_STATS_PATH = BASE_DIR / "transformer_norm_stats.npz"


MODEL_DELAY = None
MODEL_COST = None
EXPLAINER_DELAY = None
EXPLAINER_COST = None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------


def _load_models() -> None:
    """Load XGBoost models and SHAP explainers if available."""
    global MODEL_DELAY, MODEL_COST, EXPLAINER_DELAY, EXPLAINER_COST

    if joblib is None:
        LOGGER.warning("joblib is not installed – running in dummy mode.")
        return

    # Delay model
    if MODEL_DELAY is None and MODEL_DELAY_PATH.exists():
        try:
            MODEL_DELAY = joblib.load(MODEL_DELAY_PATH)
            LOGGER.info("Loaded delay model from %s", MODEL_DELAY_PATH)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load delay model: %s", exc)

    # Cost model
    if MODEL_COST is None and MODEL_COST_PATH.exists():
        try:
            MODEL_COST = joblib.load(MODEL_COST_PATH)
            LOGGER.info("Loaded cost model from %s", MODEL_COST_PATH)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to load cost model: %s", exc)

    # SHAP explainers
    if shap is not None:
        if EXPLAINER_DELAY is None and EXPL_DELAY_PATH.exists():
            try:
                EXPLAINER_DELAY = joblib.load(EXPL_DELAY_PATH)
                LOGGER.info("Loaded delay SHAP explainer from %s", EXPL_DELAY_PATH)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Failed to load delay explainer: %s", exc)

        if EXPLAINER_COST is None and EXPL_COST_PATH.exists():
            try:
                EXPLAINER_COST = joblib.load(EXPL_COST_PATH)
                LOGGER.info("Loaded cost SHAP explainer from %s", EXPL_COST_PATH)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Failed to load cost explainer: %s", exc)
    else:
        LOGGER.warning("shap is not installed – SHAP outputs will be empty.")


# Load at import time
_load_models()


# -------------------------------------------------------------------
# Feature engineering helper
# -------------------------------------------------------------------


def _build_feature_vector(df_project: pd.DataFrame, model: Any) -> np.ndarray:
    """
    Build a numeric feature vector for a single project.

    IMPORTANT:
    - This is a generic placeholder.
    - It creates an array of length `model.n_features_in_` and fills the
      first few slots with simple stats from the project, leaving the
      rest as zeros.
    - For *proper* predictions, change this to mirror the feature
      engineering you used when training the model.

    Stats we currently use (if the columns exist):
    - num_rows: number of transaction rows
    - total_payment_amount: sum of 'payment_amount'
    - mean_payment_amount: mean of 'payment_amount'
    - max_contract_percent_paid: max of 'contract_percent_paid'
    - project_value: first value of 'project_value'
    """
    try:
        n_features = int(getattr(model, "n_features_in_", 0))
    except Exception:
        n_features = 0

    if n_features <= 0:
        # Fallback: single zero feature
        return np.zeros((1, 1), dtype=float)

    x = np.zeros((1, n_features), dtype=float)

    # Basic stats from the project
    num_rows = float(len(df_project))

    total_payment = (
        float(df_project["payment_amount"].fillna(0).sum())
        if "payment_amount" in df_project.columns
        else 0.0
    )

    mean_payment = (
        float(df_project["payment_amount"].fillna(0).mean())
        if "payment_amount" in df_project.columns
        else 0.0
    )

    max_percent_paid = (
        float(df_project["contract_percent_paid"].fillna(0).max())
        if "contract_percent_paid" in df_project.columns
        else 0.0
    )

    project_value = (
        float(df_project["project_value"].iloc[0])
        if "project_value" in df_project.columns and len(df_project) > 0
        else 0.0
    )

    stats = [
        num_rows,
        total_payment,
        mean_payment,
        max_percent_paid,
        project_value,
    ]

    for i, val in enumerate(stats):
        if i < n_features:
            x[0, i] = val

    return x


# -------------------------------------------------------------------
# Core API
# -------------------------------------------------------------------


def make_risk_summary_from_df(df_project: pd.DataFrame) -> Dict[str, Any]:
    """
    Takes one project's raw transactions dataframe and returns a JSON-ready
    dictionary with predictions and (optional) SHAP values.

    This is the function FastAPI calls in `training_model/main.py`.
    """

    # Project ID (best effort)
    if "project_id" in df_project.columns and len(df_project["project_id"]) > 0:
        project_id = str(df_project["project_id"].iloc[0])
    else:
        project_id = "unknown"

    # ------------------------------------------------------------------
    # Default / fallback predictions (dummy mode)
    # ------------------------------------------------------------------
    weeks_late_xgb: float = 0.0
    cost_overrun_xgb: float = 0.0
    shap_weeks: List[float] = []
    shap_cost: List[float] = []

    # ------------------------------------------------------------------
    # Try real XGBoost models
    # ------------------------------------------------------------------
    try:
        if MODEL_DELAY is not None and MODEL_COST is not None:
            features_delay = _build_feature_vector(df_project, MODEL_DELAY)
            features_cost = _build_feature_vector(df_project, MODEL_COST)

            # XGBoost predictions
            weeks_pred = float(MODEL_DELAY.predict(features_delay)[0])
            cost_pred = float(MODEL_COST.predict(features_cost)[0])

            weeks_late_xgb = weeks_pred
            cost_overrun_xgb = cost_pred

            # SHAP values (if explainers available)
            if EXPLAINER_DELAY is not None:
                try:
                    sv = EXPLAINER_DELAY.shap_values(features_delay)
                    shap_weeks = np.asarray(sv)[0].tolist()
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Failed to compute SHAP weeks_late: %s", exc)

            if EXPLAINER_COST is not None:
                try:
                    sv = EXPLAINER_COST.shap_values(features_cost)
                    shap_cost = np.asarray(sv)[0].tolist()
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Failed to compute SHAP cost_overrun: %s", exc)

        else:
            LOGGER.info("Models not loaded – using dummy predictions.")

    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Error during model prediction; falling back to dummy: %s", exc)
        weeks_late_xgb = 0.0
        cost_overrun_xgb = 0.0
        shap_weeks = []
        shap_cost = []

    # ------------------------------------------------------------------
    # Blended predictions (for now just XGB – transformer / GNN later)
    # ------------------------------------------------------------------
    blended_weeks = weeks_late_xgb
    blended_cost = cost_overrun_xgb

    # ------------------------------------------------------------------
    # Text explanations (simple rules for now)
    # ------------------------------------------------------------------
    if blended_weeks > 4 or blended_cost > 10:
        explanation_text = (
            "The project is forecast to be at elevated risk of delay and/or "
            "cost overrun based on current payment patterns."
        )
        recommended_actions = [
            "Review cash flow and payment schedule with key subcontractors.",
            "Increase monitoring of high-value trades and critical path tasks.",
        ]
    else:
        explanation_text = (
            "The project is forecast to be close to the planned completion date. "
            "Cost performance is forecast to be close to the original budget. "
            "No specific subcontractor is currently flagged as high risk based "
            "on payment data."
        )
        recommended_actions = [
            "Continue to monitor progress and costs, but no major risk "
            "intervention is recommended at this stage."
        ]

    # LLM flags – still off by default in this backend
    llm_used = False
    llm_error = ""
    if "OPENAI_API_KEY" not in os.environ:
        llm_error = "OPENAI_API_KEY environment variable is not set."

    # ------------------------------------------------------------------
    # Final JSON-style result
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "project_id": project_id,
        "predictions": {
            # Blended headline numbers
            "weeks_late": blended_weeks,
            "cost_overrun_percent": blended_cost,
            "blended_weeks_late": blended_weeks,
            "blended_cost_overrun_percent": blended_cost,
            # XGBoost base models
            "xgb_weeks_late": weeks_late_xgb,
            "xgb_cost_overrun_percent": cost_overrun_xgb,
            # Placeholders for future models
            "transformer_weeks_late": None,
            "transformer_cost_overrun_percent": None,
            "gnn_weeks_late": None,
            "gnn_cost_overrun_percent": None,
        },
        "shap": {
            "weeks_late": shap_weeks,
            "cost_overrun_percent": shap_cost,
        },
        "trade_risks": [],
        "explanation_text": explanation_text,
        "recommended_actions": recommended_actions,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "source_id": f"{project_id}.csv",
    }

    return result


def batch_predict(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Batch version: group by project_id and run make_risk_summary_from_df
    for each project. This mirrors the interface expected by the API.
    """
    outputs: List[Dict[str, Any]] = []

    if "project_id" not in df_all.columns:
        # No project_id column – treat entire frame as one project
        outputs.append(make_risk_summary_from_df(df_all))
        return outputs

    for pid, group in df_all.groupby("project_id"):
        outputs.append(make_risk_summary_from_df(group))

    return outputs
