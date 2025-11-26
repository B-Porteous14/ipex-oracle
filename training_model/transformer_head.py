"""
Safe transformer_head.py

Right now, transformer_model.pt appears to be a state_dict (an OrderedDict of
weights) rather than a full nn.Module. Without the original model class
definition, we cannot run real Transformer inference.

This file:
  - Tries to load the checkpoint.
  - If it looks like a state_dict, logs a clear message and disables Transformer.
  - Keeps the predict_transformer_from_df() API so ml_core stays happy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    _TORCH_OK = False

TRANSFORMER_AVAILABLE: bool = False

_MODEL: Optional[Any] = None
_NORM_STATS: Optional[Dict[str, Any]] = None
_FEATURE_COLS: Optional[np.ndarray] = None
_TARGET_COLS: Optional[np.ndarray] = None


def _log(msg: str) -> None:
    print(f"[transformer_head] {msg}", flush=True)


def parse_money(value) -> float:
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


def _load_transformer() -> None:
    """
    Try to load the Transformer checkpoint.

    If we detect it is only a state_dict (OrderedDict), we log and disable
    Transformer predictions until the real architecture is wired.
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
        _log(f"Loading Transformer checkpoint from {model_path} ...")
        obj = torch.load(model_path, map_location="cpu")

        # If this is a state_dict (most common case), we cannot call it directly.
        if isinstance(obj, dict):
            from collections import OrderedDict

            if isinstance(obj, OrderedDict) or all(
                isinstance(k, str) for k in obj.keys()
            ):
                _log(
                    "Checkpoint looks like a state_dict (OrderedDict of weights), "
                    "not a full nn.Module. Without the original Transformer class "
                    "definition we cannot run inference. Disabling Transformer for now."
                )
                TRANSFORMER_AVAILABLE = False
                _MODEL = None
                return

        # Otherwise, we assume it's a full nn.Module
        _MODEL = obj
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
            f"Loaded Transformer norm stats with {len(_FEATURE_COLS)} features "
            f"and {len(_TARGET_COLS)} targets."
        )
        TRANSFORMER_AVAILABLE = True
    except Exception as exc:
        _log(f"ERROR loading Transformer: {exc}")
        TRANSFORMER_AVAILABLE = False
        _MODEL = None
        _NORM_STATS = None
        _FEATURE_COLS = None
        _TARGET_COLS = None


# Try at import
_load_transformer()


def predict_transformer_from_df(
    df_project: pd.DataFrame, project_id: str = "unknown"
) -> Optional[Dict[str, float]]:
    """
    Placeholder inference function used by ml_core.

    Currently returns None because the Transformer checkpoint is a state_dict
    and we don't have the model architecture wired in this repo yet.
    """
    _load_transformer()

    if not TRANSFORMER_AVAILABLE or _MODEL is None:
        _log(
            f"Project {project_id}: Transformer not available (state_dict only) – "
            "returning None."
        )
        return None

    # If we ever wire the true architecture, the real inference code would go here.
    return None
