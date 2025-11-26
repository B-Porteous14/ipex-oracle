"""
transformer_head.py

Inference glue between a raw per-project payments DataFrame and the
trained Transformer saved in:

    training_model/transformer_model.pt
    training_model/transformer_norm_stats.npz

This module exposes:

    - TRANSFORMER_AVAILABLE: bool
    - parse_money(value) -> float | numpy.nan
    - predict_transformer_from_df(df_project, project_id) -> Optional[Dict[str, float]]

ml_core imports these and will:

    * Skip Transformer if TRANSFORMER_AVAILABLE is False
    * Log and continue if predict_transformer_from_df() returns None / raises
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Torch is optional: Railway has it, but local tools might not.
try:
    import torch
    from torch import nn  # noqa: F401  # for type hints / saved models

    _TORCH_OK = True
except Exception:  # pragma: no cover
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


def parse_money(value) -> float:
    """
    Parse money-like strings into floats.

    Mirrors the helper in train_models.py so behaviour is consistent.
    Examples of accepted values:
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

    This is called lazily the first time you run prediction, but we also
    attempt to call it at import time so failures appear clearly in logs.
    """
    global TRANSFORMER_AVAILABLE, _MODEL, _NORM_STATS, _FEATURE_COLS, _TARGET_COLS

    if TRANSFORMER_AVAILABLE:
        return

    if not _TORCH_OK:
        print("[transformer_head] torch not available – Transformer disabled.")
        TRANSFORMER_AVAILABLE = False
        return

    base = Path(__file__).resolve().parent
    model_path = base / "transformer_model.pt"
    stats_path = base / "transformer_norm_stats.npz"

    if not model_path.exists() or not stats_path.exists():
        print(
            f"[transformer_head] Missing Transformer artefacts at {model_path} / {stats_path}; "
            "Transformer predictions will be disabled."
        )
        TRANSFORMER_AVAILABLE = False
        return

    try:
        print(f"[transformer_head] Loading Transformer model from {model_path.name}...")
        _MODEL = torch.load(model_path, map_location="cpu")
        if hasattr(_MODEL, "eval"):
            _MODEL.eval()

        print(f"[transformer_head] Loading Transformer normalisation stats from {stats_path.name}...")
        stats = np.load(stats_path, allow_pickle=True)
        _NORM_STATS = {
            "mean": stats["mean"],
            "std": stats["std"],
            "target_mean": stats["target_mean"],
            "target_std": stats["target_std"],
        }
        _FEATURE_COLS = stats["feature_cols"]
        _TARGET_COLS = stats["target_cols"]

        print(
            f"[transformer_head] Loaded Transformer with {len(_FEATURE_COLS)} features: "
            f"{list(_FEATURE_COLS)} and targets {list(_TARGET_COLS)}"
        )
        TRANSFORMER_AVAILABLE = True
    except Exception as exc:  # pragma: no cover
        print(f"[transformer_head] ERROR loading Transformer: {exc}")
        TRANSFORMER_AVAILABLE = False
        _MODEL = None
        _NORM_STATS = None
        _FEATURE_COLS = None
        _TARGET_COLS = None


# Try to load artefacts on import so we get early feedback
_load_transformer()


# ---------------------------------------------------------------------
# Timeseries feature building
# ---------------------------------------------------------------------

def _build_timeseries_for_project(df_project: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Build a [T, F] timeseries array for a single project, in the same
    feature order as used for training (from transformer_norm_stats.npz).

    Features from stats file:
        ['t',
         'days_since_first_payment',
         'payment_amount',
         'cum_project_paid',
         'pct_project_paid',
         'project_value']
    """

    if _FEATURE_COLS is None:
        print("[transformer_head] No feature_cols in norm stats; cannot build timeseries.")
        return None

    # Work on a copy
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
        print("[transformer_head] No valid payment rows for Transformer.")
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
        # Try a fallback from original column
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
            # If a feature is completely missing, fill with zeros
            print(
                f"[transformer_head] Warning: missing feature column '{col}', "
                "filling with 0."
            )
            features.append(np.zeros(len(df), dtype=float))
        else:
            series = df[col].astype(float).fillna(0.0).to_numpy()
            features.append(series)

    # Shape: [F, T] -> [T, F]
    x = np.stack(features, axis=0).T
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

    # Make sure shapes line up: [F] broadcast over [T, F]
    if mean.shape[0] != x.shape[1]:
        print(
            "[transformer_head] Warning: mean shape does not match features; "
            "skipping normalisation."
        )
        return x
    if std.shape[0] != x.shape[1]:
        print(
            "[transformer_head] Warning: std shape does not match features; "
            "skipping normalisation."
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

    Takes a per-project payments dataframe, builds the time-series features,
    runs the Transformer, and returns a dict:

        {
            "weeks_late": float,
            "cost_overrun_percent": float,
        }

    Returns None if Transformer is not available or if inference fails.
    """
    if not TRANSFORMER_AVAILABLE or _MODEL is None:
        return None

    try:
        x = _build_timeseries_for_project(df_project)
        if x is None or x.size == 0:
            print(
                f"[transformer_head] Project {project_id}: "
                "no usable timeseries for Transformer."
            )
            return None

        x = _normalise_features(x)  # [T, F]
        # Add batch dimension: [1, T, F]
        x_batch = np.expand_dims(x, axis=0)

        x_tensor = torch.from_numpy(x_batch).float()

        with torch.no_grad():
            out = _MODEL(x_tensor)

        # Try to interpret output
        out_np = out.detach().cpu().numpy()

        # Common patterns:
        #   [1, 2]        -> 2 targets
        #   [1, T, 2]     -> sequence of 2-dim outputs, take last step
        if out_np.ndim == 2 and out_np.shape[0] == 1:
            vec = out_np[0]
        elif out_np.ndim == 3:
            # Take last timestep
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
            print(
                f"[transformer_head] Project {project_id}: "
                "Transformer output empty."
            )
            return None

        # Optionally un-normalise targets if they were trained in z-space
        if (
            _NORM_STATS is not None
            and "target_mean" in _NORM_STATS
            and "target_std" in _NORM_STATS
        ):
            t_mean = np.asarray(_NORM_STATS["target_mean"], dtype=float)
            t_std = np.asarray(_NORM_STATS["target_std"], dtype=float)
            # Expect shape [2]
            if t_mean.shape[0] == 2 and t_std.shape[0] == 2:
                weeks_pred = weeks_pred * t_std[0] + t_mean[0]
                cost_pred = cost_pred * t_std[1] + t_mean[1]

        print(
            f"[transformer_head] Project {project_id}: "
            f"Transformer weeks={weeks_pred:.2f}, cost={cost_pred:.2f}"
        )

        return {
            "weeks_late": float(weeks_pred),
            "cost_overrun_percent": float(cost_pred),
        }

    except Exception as exc:  # pragma: no cover
        print(
            f"[transformer_head] ERROR during Transformer inference for project "
            f"{project_id}: {exc}"
        )
        return None
