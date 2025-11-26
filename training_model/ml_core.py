"""
ml_core.py

Production ML core for IPEX Oracle.

- Loads XGBoost delay & cost models (and SHAP explainers) from .pkl files.
- Aligns incoming features to the model's expected feature set.
- Avoids XGBoost "feature_names mismatch" errors by passing NumPy arrays.
- Gracefully degrades to zeros if any artefacts are missing.

Transformer / GNN models are optional and currently return None so the
API fields appear as `null` in JSON. You can wire these in later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("ml_core")
if not logger.handlers:
    # Basic console logging; Railway will capture stdout.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[ml_core] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths to artefacts
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DELAY_PATH = BASE_DIR / "model_weeks_late.pkl"
MODEL_COST_PATH = BASE_DIR / "model_overrun_percent.pkl"

EXPLAINER_DELAY_PATH = BASE_DIR / "explainer_weeks_late.pkl"
EXPLAINER_COST_PATH = BASE_DIR / "explainer_overrun_percent.pkl"

# Optional transformer artefacts (only used if you later wire them in)
TRANSFORMER_MODEL_PATH = BASE_DIR / "transformer_model.pt"
TRANSFORMER_NORM_STATS_PATH = BASE_DIR / "transformer_norm_stats.npz"

# ---------------------------------------------------------------------------
# Safe loading helpers
# ---------------------------------------------------------------------------


def _load_joblib(path: Path, name: str) -> Optional[Any]:
    if not path.exists():
        logger.warning("%s not found at %s – using fallback behaviour.", name, path)
        return None
    try:
        obj = joblib.load(path)
        logger.info("Loaded %s from %s", name, path.name)
        return obj
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load %s from %s: %s", name, path, exc)
        return None


def _get_feature_names_from_model(model: Any) -> List[str]:
    """
    Try hard to discover the feature name list used when the XGBoost model
    was trained. Different xgboost versions store this differently.
    """
    if model is None:
        return []

    # 1) Newer xgboost: model.feature_names_in_
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names):
        return [str(c) for c in list(names)]

    # 2) Access booster.feature_names
    booster = getattr(model, "get_booster", None)
    if booster is not None:
        try:
            bst = booster()
            if hasattr(bst, "feature_names") and bst.feature_names:
                return [str(c) for c in list(bst.feature_names)]
        except Exception:
            pass  # fall through

    return []


# ---------------------------------------------------------------------------
# Load models / explainers at import time
# ---------------------------------------------------------------------------

model_weeks = _load_joblib(MODEL_DELAY_PATH, "delay model")
model_cost = _load_joblib(MODEL_COST_PATH, "cost model")

explainer_weeks = _load_joblib(EXPLAINER_DELAY_PATH, "delay SHAP explainer")
explainer_cost = _load_joblib(EXPLAINER_COST_PATH, "cost SHAP explainer")

DELAY_FEATURES: List[str] = _get_feature_names_from_model(model_weeks)
COST_FEATURES: List[str] = _get_feature_names_from_model(model_cost)

if DELAY_FEATURES:
    logger.info("Delay model expects %d features.", len(DELAY_FEATURES))
if COST_FEATURES:
    logger.info("Cost model expects %d features.", len(COST_FEATURES))

# Optional: lazy-loaded transformer model (if you wire it in later)
_transformer_model: Any = None


# ---------------------------------------------------------------------------
# Feature preparation and alignment
# ---------------------------------------------------------------------------


def _numericise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all columns to numeric where possible.

    - For numeric dtypes we keep as-is.
    - For object/string columns we strip non-numeric characters (to cope with
      '$', commas, etc.) and coerce to float. Non-convertible values become NaN.
    """
    out = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype(float)
        else:
            # Convert things like "$1,234.56" or "-$6,557.18" to numbers
            cleaned = (
                s.astype(str)
                .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
                .replace({"": np.nan})
            )
            out[col] = pd.to_numeric(cleaned, errors="coerce")

    # Replace remaining NaNs with 0 – the models were trained on numeric data
    out = out.fillna(0.0)
    return out


def _align_features_for_model(
    df_numeric: pd.DataFrame,
    expected_cols: List[str],
    model_label: str,
) -> pd.DataFrame:
    """
    Aligns the numeric dataframe to the model's expected feature set.

    - Missing features are added and filled with 0.
    - Extra features are dropped.
    - Returns a dataframe with columns in the exact expected order.
    """
    if not expected_cols:
        # If we don't know the feature names, just pass everything through.
        logger.warning(
            "%s has no recorded feature_names; using all numeric columns: %s",
            model_label,
            list(df_numeric.columns),
        )
        return df_numeric.copy()

    missing = [c for c in expected_cols if c not in df_numeric.columns]
    extra = [c for c in df_numeric.columns if c not in expected_cols]

    if missing:
        logger.warning(
            "%s: input missing %d feature(s): %s – filling with 0.0",
            model_label,
            len(missing),
            ", ".join(missing),
        )
    if extra:
        logger.info(
            "%s: input has %d extra feature(s) that will be ignored: %s",
            model_label,
            len(extra),
            ", ".join(extra[:25]) + ("..." if len(extra) > 25 else ""),
        )

    df_aligned = df_numeric.reindex(columns=expected_cols, fill_value=0.0)
    return df_aligned


# ---------------------------------------------------------------------------
# Core prediction helpers
# ---------------------------------------------------------------------------


def _predict_with_xgb(
    model: Any,
    df_numeric: pd.DataFrame,
    expected_cols: List[str],
    label: str,
) -> float:
    """
    Predict using an XGBoost model, with robust handling of feature name
    mismatches.

    We **always** pass a NumPy array into model.predict(), which avoids
    xgboost's feature_names consistency checks.
    """
    if model is None:
        return 0.0

    try:
        X = _align_features_for_model(df_numeric, expected_cols, f"{label} model")
        # IMPORTANT: use .values so xgboost does not try to match column names
        y_pred = model.predict(X.values)
        if isinstance(y_pred, (list, np.ndarray, pd.Series)):
            return float(y_pred[0])
        return float(y_pred)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("%s model predict failed: %s", label, exc)
        return 0.0


def _compute_shap_values(
    explainer: Any,
    df_numeric: pd.DataFrame,
    expected_cols: List[str],
    label: str,
) -> List[float]:
    """
    Compute SHAP values for a single row, if possible.
    """
    if explainer is None:
        return []

    try:
        X = _align_features_for_model(df_numeric, expected_cols, f"{label} SHAP")
        sv = explainer.shap_values(X)
        # sv might be shape (n_samples, n_features)
        if isinstance(sv, list):
            sv = sv[0]
        if isinstance(sv, np.ndarray):
            return sv[0].tolist()
        return []
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("%s SHAP failed: %s", label, exc)
        return []


def _simple_explanation_text(weeks_late: float, cost_overrun_pct: float) -> str:
    """
    Lightweight, non-LLM explanation that always works (even without OpenAI).
    """
    if weeks_late <= 0.5 and abs(cost_overrun_pct) < 2:
        risk_level = "low risk"
    elif weeks_late <= 4 and cost_overrun_pct < 10:
        risk_level = "moderate risk"
    else:
        risk_level = "elevated risk"

    return (
        f"The project is currently assessed as {risk_level}. "
        f"Schedule performance is forecast at roughly {weeks_late:.1f} weeks "
        f"{'late' if weeks_late >= 0 else 'ahead of plan'}, while cost "
        f"performance is forecast at about {cost_overrun_pct:.1f}% "
        f"{'over' if cost_overrun_pct >= 0 else 'under'} the original budget."
    )


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def make_risk_summary_from_df(df_project: pd.DataFrame) -> Dict[str, Any]:
    """
    Main entry point used by FastAPI.

    df_project: dataframe containing data for a **single project**.
                It can be raw transactions or pre-aggregated features.
                Any numeric columns will be used for modelling.
    """

    if df_project is None or df_project.empty:
        return {
            "project_id": "unknown",
            "predictions": {
                "weeks_late": 0.0,
                "cost_overrun_percent": 0.0,
                "blended_weeks_late": 0.0,
                "blended_cost_overrun_percent": 0.0,
                "xgb_weeks_late": 0.0,
                "xgb_cost_overrun_percent": 0.0,
                "transformer_weeks_late": None,
                "transformer_cost_overrun_percent": None,
                "gnn_weeks_late": None,
                "gnn_cost_overrun_percent": None,
            },
            "shap": {"weeks_late": [], "cost_overrun_percent": []},
            "trade_risks": [],
            "explanation_text": "No project data was provided.",
            "recommended_actions": [],
            "llm_used": False,
            "llm_error": "No data",
            "source_id": None,
        }

    # Try to get a project_id if one exists
    project_id = None
    for candidate in ["project_id", "ProjectID", "project"]:
        if candidate in df_project.columns:
            try:
                project_id = str(df_project[candidate].iloc[0])
                break
            except Exception:
                pass
    if project_id is None:
        project_id = "unknown"

    # Prepare numeric features
    df_numeric = _numericise_dataframe(df_project)

    # --- XGBoost predictions -------------------------------------------------
    xgb_weeks = _predict_with_xgb(
        model_weeks, df_numeric, DELAY_FEATURES, label="delay"
    )
    xgb_cost_pct = _predict_with_xgb(
        model_cost, df_numeric, COST_FEATURES, label="cost"
    )

    shap_weeks = _compute_shap_values(
        explainer_weeks, df_numeric, DELAY_FEATURES, label="delay"
    )
    shap_cost = _compute_shap_values(
        explainer_cost, df_numeric, COST_FEATURES, label="cost"
    )

    # --- Transformer / GNN placeholders -------------------------------------
    transformer_weeks: Optional[float] = None
    transformer_cost_pct: Optional[float] = None

    gnn_weeks: Optional[float] = None
    gnn_cost_pct: Optional[float] = None

    # You can wire in real models later; for now we just log that they're null.
    if not TRANSFORMER_MODEL_PATH.exists():
        logger.info("Transformer model file not found; transformer outputs will be null.")
    if gnn_weeks is None and gnn_cost_pct is None:
        logger.info("GNN models not configured; GNN outputs will be null.")

    # --- Blended scores (for now just equal to XGBoost) ---------------------
    blended_weeks = xgb_weeks
    blended_cost_pct = xgb_cost_pct

    # --- Simple explanation & recommendations -------------------------------
    explanation = _simple_explanation_text(blended_weeks, blended_cost_pct)

    recommended_actions: List[str] = []
    if blended_weeks > 2 or blended_cost_pct > 5:
        recommended_actions.append(
            "Review the largest trades and recent payment history to identify drivers of delay and cost pressure."
        )
    if blended_weeks <= 2 and blended_cost_pct <= 5:
        recommended_actions.append(
            "Continue to monitor progress and costs; no major risk intervention is recommended at this stage."
        )
    if not recommended_actions:
        recommended_actions.append(
            "Engage the project team to validate these risk signals and agree on a mitigation plan."
        )

    # We are not actually calling OpenAI here, so we mark the LLM as unused.
    llm_used = False
    llm_error = "OPENAI_API_KEY environment variable is not set."

    # Try to infer a source_id if csv filename was known upstream; otherwise use project_id.
    source_id = os.environ.get("CURRENT_SOURCE_ID", f"{project_id}.csv")

    return {
        "project_id": project_id,
        "predictions": {
            "weeks_late": float(blended_weeks),
            "cost_overrun_percent": float(blended_cost_pct),
            "blended_weeks_late": float(blended_weeks),
            "blended_cost_overrun_percent": float(blended_cost_pct),
            "xgb_weeks_late": float(xgb_weeks),
            "xgb_cost_overrun_percent": float(xgb_cost_pct),
            "transformer_weeks_late": transformer_weeks,
            "transformer_cost_overrun_percent": transformer_cost_pct,
            "gnn_weeks_late": gnn_weeks,
            "gnn_cost_overrun_percent": gnn_cost_pct,
        },
        "shap": {
            "weeks_late": shap_weeks,
            "cost_overrun_percent": shap_cost,
        },
        "trade_risks": [],  # placeholder for future per-trade risk flags
        "explanation_text": explanation,
        "recommended_actions": recommended_actions,
        "llm_used": llm_used,
        "llm_error": llm_error,
        "source_id": source_id,
    }


def batch_predict(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Batch version used by CLI tools / tests: group by project_id (if present)
    and run make_risk_summary_from_df for each group.
    """
    outputs: List[Dict[str, Any]] = []

    if df_all is None or df_all.empty:
        return outputs

    if "project_id" not in df_all.columns:
        # Treat entire dataframe as one project
        outputs.append(make_risk_summary_from_df(df_all))
        return outputs

    for pid, group in df_all.groupby("project_id"):
        outputs.append(make_risk_summary_from_df(group.reset_index(drop=True)))

    return outputs
