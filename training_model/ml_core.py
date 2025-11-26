from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional deps – code still imports if some are missing.
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Try relative import first (package) then flat (module)
try:
    from . import transformer_head  # type: ignore
except Exception:  # pragma: no cover
    try:
        import transformer_head  # type: ignore
    except Exception:
        transformer_head = None  # type: ignore


# ---------------------------------------------------------------------------
# Paths to model artefacts
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DELAY_PATH = BASE_DIR / "model_weeks_late.pkl"
MODEL_COST_PATH = BASE_DIR / "model_overrun_percent.pkl"
EXPLAINER_DELAY_PATH = BASE_DIR / "explainer_weeks_late.pkl"
EXPLAINER_COST_PATH = BASE_DIR / "explainer_overrun_percent.pkl"

TRANSFORMER_MODEL_PATH = BASE_DIR / "transformer_model.pt"
TRANSFORMER_NORM_STATS_PATH = BASE_DIR / "transformer_norm_stats.npz"


# ---------------------------------------------------------------------------
# Global model handles (lazy-loaded once)
# ---------------------------------------------------------------------------

_models_loaded = False

_model_weeks: Optional[Any] = None
_model_cost: Optional[Any] = None

_explainer_weeks: Optional[Any] = None
_explainer_cost: Optional[Any] = None

_transformer_model: Optional[Any] = None
_transformer_norm_stats: Optional[Any] = None


def _load_models_once() -> None:
    """
    Load XGBoost models, SHAP explainers and Transformer once.
    Safe to call multiple times – it only loads on first call.
    """
    global _models_loaded
    global _model_weeks, _model_cost
    global _explainer_weeks, _explainer_cost
    global _transformer_model, _transformer_norm_stats

    if _models_loaded:
        return

    print("[ml_core] Loading models…")

    # --- XGBoost models -----------------------------------------------------
    if joblib is not None and MODEL_DELAY_PATH.exists():
        try:
            _model_weeks = joblib.load(MODEL_DELAY_PATH)
            print(f"[ml_core] Loaded weeks-late model from {MODEL_DELAY_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading weeks model: {exc}")

    if joblib is not None and MODEL_COST_PATH.exists():
        try:
            _model_cost = joblib.load(MODEL_COST_PATH)
            print(f"[ml_core] Loaded cost-overrun model from {MODEL_COST_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading cost model: {exc}")

    # --- SHAP explainers ----------------------------------------------------
    if joblib is not None and EXPLAINER_DELAY_PATH.exists():
        try:
            _explainer_weeks = joblib.load(EXPLAINER_DELAY_PATH)
            print(f"[ml_core] Loaded weeks explainer from {EXPLAINER_DELAY_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading weeks explainer: {exc}")

    if joblib is not None and EXPLAINER_COST_PATH.exists():
        try:
            _explainer_cost = joblib.load(EXPLAINER_COST_PATH)
            print(f"[ml_core] Loaded cost explainer from {EXPLAINER_COST_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading cost explainer: {exc}")

    # --- Transformer model + normalisation stats ---------------------------
    if torch is not None and TRANSFORMER_MODEL_PATH.exists():
        try:
            _transformer_model = torch.load(TRANSFORMER_MODEL_PATH, map_location="cpu")
            if hasattr(_transformer_model, "eval"):
                _transformer_model.eval()
            print(f"[ml_core] Loaded transformer model from {TRANSFORMER_MODEL_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading transformer model: {exc}")
            _transformer_model = None

    if TRANSFORMER_NORM_STATS_PATH.exists():
        try:
            _transformer_norm_stats = np.load(TRANSFORMER_NORM_STATS_PATH)
            print(f"[ml_core] Loaded transformer norm stats from {TRANSFORMER_NORM_STATS_PATH.name}")
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING loading transformer norm stats: {exc}")
            _transformer_norm_stats = None

    _models_loaded = True


# ---------------------------------------------------------------------------
# Feature engineering for XGBoost (based on training_projects_train.csv)
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS = [
    "planned_cost",
    "planned_duration_days",
    "num_payments",
    "avg_payment_amount",
    "num_trades",
    "max_trade_paid",
    "trade_cost_concentration",
]


def _safe_get_first(df: pd.DataFrame, cols: List[str]) -> float:
    """Return first non-null value from the first existing column; else 0."""
    for col in cols:
        if col in df.columns and len(df[col]) > 0:
            val = df[col].iloc[0]
            try:
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return 0.0


def _build_features_from_raw(df_project: pd.DataFrame) -> pd.DataFrame:
    """
    Build the tabular features the XGBoost models expect.

    Matches the training columns in training_projects_train.csv:
      planned_cost
      planned_duration_days
      num_payments
      avg_payment_amount
      num_trades
      max_trade_paid
      trade_cost_concentration
    """
    df = df_project.copy()

    # planned_cost
    planned_cost = _safe_get_first(
        df,
        ["planned_cost", "project_value", "contract_sum", "project_contract_value"],
    )

    # planned_duration_days (or derive from dates)
    if "planned_duration_days" in df.columns and len(df) > 0:
        try:
            planned_duration_days = float(df["planned_duration_days"].iloc[0])
        except Exception:
            planned_duration_days = 0.0
    else:
        planned_duration_days = 0.0
        start_val = _safe_get_first(
            df,
            ["project_contract_start_date", "contract_start_date", "start_date"],
        )
        end_val = _safe_get_first(
            df,
            ["project_completion_date", "planned_completion_date", "end_date"],
        )
        try:
            if start_val and end_val:
                start_dt = pd.to_datetime(start_val)
                end_dt = pd.to_datetime(end_val)
                planned_duration_days = float((end_dt - start_dt).days)
        except Exception:
            planned_duration_days = 0.0

    # num_payments
    num_payments = float(len(df)) if len(df) > 0 else 0.0

    # payment amount stats
    payment_col = None
    for col in ["payment_amount", "amount", "payment_value"]:
        if col in df.columns:
            payment_col = col
            break

    if payment_col is not None and len(df) > 0:
        payments = df[payment_col].fillna(0.0)
        avg_payment_amount = float(payments.mean())
        total_paid = float(payments.sum())
    else:
        avg_payment_amount = 0.0
        total_paid = 0.0

    # trade metrics
    trade_col = None
    for col in ["trade_id", "trade_work_type", "trade_name"]:
        if col in df.columns:
            trade_col = col
            break

    if trade_col is not None and payment_col is not None:
        grouped = df.groupby(trade_col)[payment_col].sum()
        num_trades = float(grouped.shape[0])
        if grouped.shape[0] > 0:
            max_trade_paid = float(grouped.max())
            trade_cost_concentration = float(max_trade_paid / total_paid) if total_paid > 0 else 0.0
        else:
            num_trades = 0.0
            max_trade_paid = 0.0
            trade_cost_concentration = 0.0
    else:
        num_trades = 0.0
        max_trade_paid = 0.0
        trade_cost_concentration = 0.0

    row = {
        "planned_cost": planned_cost,
        "planned_duration_days": planned_duration_days,
        "num_payments": num_payments,
        "avg_payment_amount": avg_payment_amount,
        "num_trades": num_trades,
        "max_trade_paid": max_trade_paid,
        "trade_cost_concentration": trade_cost_concentration,
    }

    X = pd.DataFrame([row], columns=_FEATURE_COLUMNS)
    return X


# ---------------------------------------------------------------------------
# XGBoost + SHAP execution
# ---------------------------------------------------------------------------

def _run_xgboost_models(
    df_project: pd.DataFrame,
) -> Tuple[float, float, Dict[str, List[Dict[str, float]]]]:
    """
    Run XGBoost weeks-late and cost-overrun models.
    Returns:
        weeks_pred, cost_pred, shap_info
    """
    _load_models_once()
    X = _build_features_from_raw(df_project)

    weeks_pred = 0.0
    cost_pred = 0.0

    shap_info: Dict[str, List[Dict[str, float]]] = {
        "weeks_late": [],
        "cost_overrun_percent": [],
    }

    # weeks model
    if _model_weeks is not None:
        try:
            weeks_pred = float(_model_weeks.predict(X)[0])
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING: weeks model predict failed: {exc}")
            weeks_pred = 0.0

        if _explainer_weeks is not None and shap is not None:
            try:
                sv = _explainer_weeks(X)  # TreeExplainer / Explanation / array
                if hasattr(sv, "values"):
                    vals = np.asarray(sv.values)[0]
                else:
                    vals = np.asarray(sv)[0]
                for col, val, s in zip(X.columns, X.iloc[0].tolist(), vals.tolist()):
                    shap_info["weeks_late"].append(
                        {"feature": col, "value": float(val), "shap": float(s)}
                    )
            except Exception as exc:  # pragma: no cover
                print(f"[ml_core] WARNING: weeks SHAP failed: {exc}")

    # cost model
    if _model_cost is not None:
        try:
            cost_pred = float(_model_cost.predict(X)[0])
        except Exception as exc:  # pragma: no cover
            print(f"[ml_core] WARNING: cost model predict failed: {exc}")
            cost_pred = 0.0

        if _explainer_cost is not None and shap is not None:
            try:
                sv = _explainer_cost(X)
                if hasattr(sv, "values"):
                    vals = np.asarray(sv.values)[0]
                else:
                    vals = np.asarray(sv)[0]
                for col, val, s in zip(X.columns, X.iloc[0].tolist(), vals.tolist()):
                    shap_info["cost_overrun_percent"].append(
                        {"feature": col, "value": float(val), "shap": float(s)}
                    )
            except Exception as exc:  # pragma: no cover
                print(f"[ml_core] WARNING: cost SHAP failed: {exc}")

    return weeks_pred, cost_pred, shap_info


# ---------------------------------------------------------------------------
# Transformer execution (via transformer_head.predict_project)
# ---------------------------------------------------------------------------

def _run_transformer(
    df_project: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run your Transformer on a single project.

    Delegates to transformer_head.predict_project if available.
    Returns (weeks_late, cost_overrun_percent) or (None, None).
    """
    _load_models_once()

    if transformer_head is None or _transformer_model is None:
        return None, None

    try:
        if hasattr(transformer_head, "predict_project"):
            weeks, cost = transformer_head.predict_project(
                df_project=df_project,
                model=_transformer_model,
                norm_stats=_transformer_norm_stats,
            )
            if weeks is not None:
                weeks = float(weeks)
            if cost is not None:
                cost = float(cost)
            return weeks, cost
    except Exception as exc:  # pragma: no cover
        print(f"[ml_core] WARNING: transformer prediction failed: {exc}")

    return None, None


# ---------------------------------------------------------------------------
# Public API used by FastAPI
# ---------------------------------------------------------------------------

def make_risk_summary_from_df(df_project: pd.DataFrame) -> Dict[str, Any]:
    """
    Main entry point: one project's raw dataframe in, JSON-style dict out.
    """

    # project_id
    if "project_id" in df_project.columns and len(df_project["project_id"]) > 0:
        project_id = str(df_project["project_id"].iloc[0])
    else:
        project_id = "unknown"

    # source id (for trace/debug)
    attrs = getattr(df_project, "attrs", {}) or {}
    source_id = str(attrs.get("source_id", f"{project_id}.csv"))

    # ---- XGB + SHAP --------------------------------------------------------
    xgb_weeks, xgb_cost, shap_info = _run_xgboost_models(df_project)

    # ---- Transformer -------------------------------------------------------
    transformer_weeks, transformer_cost = _run_transformer(df_project)

    # ---- Blended = same as before (just XGB for now) ----------------------
    blended_weeks = xgb_weeks
    blended_cost = xgb_cost

    # ---- Simple explanation logic (same as before style) -------------------
    if blended_weeks > 8 or blended_cost > 10:
        explanation_text = (
            "The project is forecast to finish significantly late and/or over budget "
            "based on payment and contract data."
        )
        recommended_actions = [
            "Review the critical trades driving delay and cost.",
            "Re-baseline the schedule and agree a recovery plan with key subcontractors.",
        ]
    elif blended_weeks > 2 or blended_cost > 5:
        explanation_text = (
            "The project shows moderate schedule or cost risk. "
            "Performance should be closely monitored."
        )
        recommended_actions = [
            "Increase monitoring of cashflow and site progress.",
            "Review high-value trades for potential slippage.",
        ]
    else:
        explanation_text = (
            "The project is forecast to be close to the planned completion date "
            "and near the original budget based on current data."
        )
        recommended_actions = [
            "Continue to monitor progress and costs, "
            "but no major risk intervention is recommended at this stage."
        ]

    return {
        "project_id": project_id,
        "predictions": {
            # headline
            "weeks_late": blended_weeks,
            "cost_overrun_percent": blended_cost,
            "blended_weeks_late": blended_weeks,
            "blended_cost_overrun_percent": blended_cost,
            # XGB base models
            "xgb_weeks_late": xgb_weeks,
            "xgb_cost_overrun_percent": xgb_cost,
            # Transformer
            "transformer_weeks_late": transformer_weeks,
            "transformer_cost_overrun_percent": transformer_cost,
            # GNN reserved
            "gnn_weeks_late": None,
            "gnn_cost_overrun_percent": None,
        },
        "shap": shap_info,
        "trade_risks": [],
        "explanation_text": explanation_text,
        "recommended_actions": recommended_actions,
        "llm_used": False,
        "llm_error": "OPENAI_API_KEY environment variable is not set.",
        "source_id": source_id,
    }


def batch_predict(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Batch version: group by project_id and call make_risk_summary_from_df.
    """
    outputs: List[Dict[str, Any]] = []

    if "project_id" not in df_all.columns:
        outputs.append(make_risk_summary_from_df(df_all))
        return outputs

    for _, group in df_all.groupby("project_id"):
        outputs.append(make_risk_summary_from_df(group))

    return outputs
