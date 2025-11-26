"""
transformer_head.py

Inference glue between a raw per-project payments DataFrame and the
trained Transformer saved in:

    training_model/transformer_model.pt
    training_model/transformer_norm_stats.npz

Exposes:

    - TRANSFORMER_AVAILABLE: bool
    - predict_transformer_from_df(df_project, project_id) -> Optional[Dict[str, float]]

This version is very verbose in its logging so that we can see what is
happening in Railway logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Torch is optional: Railway has it, but local tools might not.
try:
    import torch  # type: ignore
    from torch import nn  # noqa: F401

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

# ---------------------------------------------------------------------
# Globals – populated by _load_transformer()
# ---------------------------------------------------------------------

TRANSFORMER_AVAILABLE: bool = False

_MODEL: Optional[Any] = None
_NORM_STATS: Optional[Dict[str, Any]] = None
_FEATURE_COLS: Optional[np.ndarray] = None
_TARGET_COLS: Optional[np.ndarray] = None


def _log(msg: str) -> None:
    # Simple print-based logging so it always appears in Railway logs
    print(f"[transformer_head] {msg}", flush=True)


def parse_money(value) -> float:
    """
    Parse money-like strings into floats.

    Examples accepted:
        "$1,234.50", "-$6,557.18", "  1234  ", 1234.0, None

    Returns np.nan for unparseable values.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan

    neg = s.startswith("-")
    s_clean = s.replace("-", "").replace("$", "").replace(",", "")
    try:
        val = float(s_clean)
    except ValueError:
        return np.nan

    return -val if neg else val


# ---------------------------------------------------------------------
# Loading artefacts
# ---------------------------------------------------------------------

def _load_transformer() -> None:
    """
    Load the Transformer model and normalisation stats.

    Called at import time and on first prediction.
    """
    global TRANSFORMER_AVAILABLE, _MODEL, _NORM_STATS, _FEATURE_COLS, _TARGET_COLS

    if TRANSFORMER_AVAILABLE:
        return

    if not _TORCH_OK:
        _log("torch not available – Transformer disabled.")
        TRANSFORMER_AVAILABLE = False
        return

    base = Path(__file__).resolve().parent
    model_path = base / "transformer_model.pt"
    stats_path = base / "transformer_norm_stats.npz"

    if not model_path.exists() or not stats_path.exists():
        _log(
            f"Missing artefacts at {model_path} / {stats_path} – "
            "Transformer predictions will be disabled."
        )
        TRANSFORMER_AVAILABLE = False
        return

    try:
        _log(f"Loading Transformer model from {model_path} ...")
        _MODEL = torch.load(model_path, map_location="cpu")
        if hasattr(_MODEL, "eval"):
            _MODEL.eval()

        _log(f"Loading normalisation stats from {stats_path} ...")
        stats = np.load(stats_path, allow_pickle=True)
        _NORM_STATS = {
            "mean": stats["mean"],
            "std": stats["std"],
            "target_mean": stats["target_mean"],
            "target_std": stats["target_std"],
        }
        _FEATURE_COLS = stats["feature_cols"]
        _TARGET_COLS = stats["target_cols"]

        _log(
            f"Loaded Transformer with {len(_FEATURE_COLS)} features: "
            f"{list(_FEATURE_COLS)} and targets {list(_TARGET_COLS)}"
        )
        TRANSFORMER_AVAILABLE = True
    except Exception as exc:
        _log(f"ERROR loading Transformer: {exc}")
        TRANSFORMER_AVAILABLE = False
        _MODEL = None
        _NORM_STATS = None
        _FEATURE_COLS = None
        _TARGET_COLS = None


# Try to load at import so startup logs show status
_load_transformer()


# ---------------------------------------------------------------------
# Timeseries feature building
# ---------------------------------------------------------------------

def _build_timeseries_for_project(df_project: pd.DataFrame, project_id: str) -> Optional[np.ndarray]:
    """
    Build a [T, F] timeseries array for a single project, in the same
    feature order as used for training (from transformer_norm_stats.npz).

    Expected features (from stats):
        ['t',
         'days_since_first_payment',
         'payment_amount',
         'cum_project_paid',
         'pct_project_paid',
         'project_value']
    """

    if _FEATURE_COLS is None:
        _log("No feature_cols in norm stats; cannot build timeseries.")
        return None

    if df_project is None or df_project.empty:
        _log(f"Project {project_id}: empty df_project passed to Transformer.")
        return None

    df = df_project.copy()

    # Parse money columns
    if "payment_amount_num" not in df.columns:
        df["payment_amount_num"] = df.get("payment_amount", np.nan).apply(parse_money)
    if "project_value_num" not in df.columns:
        df["project_value_num"] = df.get("project_value", np.nan).apply(parse_money)

    # Parse dates
    df["payment_date_parsed"] = pd.to_datetime(
        df.get("payment_date"), dayfirst=True, errors="coerce"
    )

    # Drop rows without payment date or amount
    df = df[~df["payment_date_parsed"].isna()].copy()
    df = df[~df["payment_amount_num"].isna()].copy()

    if df.empty:
        _log(f"Project {project_id}: no valid payment rows for Transformer.")
        return None

    # Sort by payment date
    df = df.sort_values("payment_date_parsed").reset_index(drop=True)

    # Compute base quantities
    first_date = df["payment_date_parsed"].iloc[0]
    df["t"] = np.arange(len(df), dtype=float)
    df["days_since_first_payment"] = (
        df["payment_date_parsed"] - first_date
    ).dt.days.astype(float)

    project_value = df["project_value_num"].iloc[0]
    if not np.isfinite(project_value) or project_value <= 0:
        pv_fallback = parse_money(df.get("project_value", [np.nan])[0])
        project_value = (
            pv_fallback if np.isfinite(pv_fallback) and pv_fallback > 0 else np.nan
        )

    df["project_value"] = project_value

    df["cum_project_paid"] = df["payment_amount_num"].cumsum()
    if np.isfinite(project_value) and project_value > 0:
        df["pct_project_paid"] = df["cum_project_paid"] / project_value
    else:
        df["pct_project_paid"] = 0.0

    # Build matrix in the exact feature order
    features = []
    for col in _FEATURE_COLS:
        col = str(col)
        if col not in df.columns:
            _log(
                f"Project {project_id}: missing feature column '{col}', "
                "filling with 0."
            )
            features.append(np.zeros(len(df), dtype=float))
        else:
            series = df[col].astype(float).fillna(0.0).to_numpy()
            features.append(series)

    x = np.stack(features, axis=0).T  # [T, F]
    _log(f"Project {project_id}: built timeseries with shape {x.shape}.")
    return x


def _normalise_features(x: np.ndarray) -> np.ndarray:
    """
    Apply per-feature normalisation using norm_stats["mean"] and ["std"].

    Expects x: [T, F]
    """
    if _NORM_STATS is None:
        return x

    mean = _NORM_STATS.get("mean")
    std = _NORM_STATS.get("std")

    if mean is None or std is None:
        return x

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    if mean.shape[0] != x.shape[1] or std.shape[0] != x.shape[1]:
        _log(
            f"Warning: mean/std shapes {mean.shape}/{std.shape} do not match "
            f"features {x.shape}; skipping normalisation."
        )
        return x

    std_safe = np.where(std == 0, 1.0, std)
    x_norm = (x - mean) / std_safe
    return x_norm


# ---------------------------------------------------------------------
# Public API: prediction
# ---------------------------------------------------------------------

def predict_transformer_from_df(
    df_project: pd.DataFrame, project_id: str = "unknown"
) -> Optional[Dict[str, float]]:
    """
    High-level helper used by ml_core.

    Returns:
        {"weeks_late": float, "cost_overrun_percent": float}
    or None if Transformer not available / fails.
    """
    _load_transformer()

    if not TRANSFORMER_AVAILABLE or _MODEL is None:
        _log(f"Project {project_id}: Transformer not available in predict()")
        return None

    try:
        x = _build_timeseries_for_project(df_project, project_id)
        if x is None or x.size == 0:
            _log(f"Project {project_id}: no usable timeseries for Transformer.")
            return None

        x = _normalise_features(x)
        x_batch = np.expand_dims(x, axis=0)  # [1, T, F]

        x_tensor = torch.from_numpy(x_batch).float()

        with torch.no_grad():
            out = _MODEL(x_tensor)

        out_np = out.detach().cpu().numpy()

        if out_np.ndim == 2 and out_np.shape[0] == 1:
            vec = out_np[0]
        elif out_np.ndim == 3:
            vec = out_np[0, -1, :]
        else:
            vec = out_np.reshape(-1)

        if vec.size >= 2:
            weeks_pred = float(vec[0])
            cost_pred = float(vec[1])
        elif vec.size == 1:
            weeks_pred = float(vec[0])
            cost_pred = 0.0
        else:
            _log(f"Project {project_id}: Transformer output empty.")
            return None

        # Un-normalise targets if they were trained in z-space
        if (
            _NORM_STATS is not None
            and "target_mean" in _NORM_STATS
            and "target_std" in _NORM_STATS
        ):
            t_mean = np.asarray(_NORM_STATS["target_mean"], dtype=float)
            t_std = np.asarray(_NORM_STATS["target_std"], dtype=float)
            if t_mean.shape[0] >= 2 and t_std.shape[0] >= 2:
                weeks_pred = weeks_pred * t_std[0] + t_mean[0]
                cost_pred = cost_pred * t_std[1] + t_mean[1]

        _log(
            f"Project {project_id}: Transformer weeks={weeks_pred:.2f}, "
            f"cost={cost_pred:.2f}"
        )

        return {
            "weeks_late": float(weeks_pred),
            "cost_overrun_percent": float(cost_pred),
        }

    except Exception as exc:
        _log(
            f"ERROR during Transformer inference for project {project_id}: {exc}"
        )
        return None
