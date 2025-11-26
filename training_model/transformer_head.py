"""
transformer_head.py

Glue code between your raw project dataframe and the trained Transformer
saved in `transformer_model.pt` + `transformer_norm_stats.npz`.

This module exposes a single function:

    predict_project(df_project, model, norm_stats) -> (weeks_late, cost_overrun%)

It is called from ml_core._run_transformer().
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Columns we treat as *non-feature* meta columns and drop from the sequence.
_META_COLUMNS = {
    "project_id",
    "Project_ID",
    "project",
    "payment_id",
    "invoice_id",
    "trade_id",
    "trade_work_type",
    "trade_name",
    "supplier_name",
    "company_name",
}

_DATE_COLUMNS = {
    "payment_date",
    "date",
    "transaction_date",
    "invoice_date",
}


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by a date-like column if present, otherwise return as-is."""
    for col in df.columns:
        if col in _DATE_COLUMNS:
            try:
                df_sorted = df.copy()
                df_sorted[col] = pd.to_datetime(df_sorted[col], errors="coerce")
                return df_sorted.sort_values(col).reset_index(drop=True)
            except Exception:
                continue
    return df.reset_index(drop=True)


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose numeric feature columns from the dataframe and drop known meta columns.

    This is intentionally generic so it still works if you add/remove columns,
    as long as the numeric feature set and order match what the Transformer
    was trained on.
    """
    df_num = df.select_dtypes(include=["number"]).copy()

    # Drop known meta columns if they happen to be numeric.
    for col in list(df_num.columns):
        if col in _META_COLUMNS:
            df_num.drop(columns=[col], inplace=True)

    return df_num


def _apply_norm_stats(
    x: np.ndarray,
    norm_stats: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Apply per-feature normalisation if norm_stats contains 'mean' and 'std' arrays.

    Expects:
        x: [T, F]
        norm_stats["mean"]: [F] or broadcastable
        norm_stats["std"]:  [F] or broadcastable
    """
    if norm_stats is None:
        return x

    mean = norm_stats.get("mean")
    std = norm_stats.get("std")

    if mean is None or std is None:
        return x

    try:
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        # Avoid divide-by-zero
        std_arr = np.where(std_arr == 0, 1.0, std_arr)

        # Broadcast if shapes don’t match perfectly
        return (x - mean_arr) / std_arr
    except Exception as exc:  # pragma: no cover
        print(f"[transformer_head] WARNING: failed to apply norm stats: {exc}")
        return x


# ---------------------------------------------------------------------------
# Main entry point used by ml_core
# ---------------------------------------------------------------------------

def predict_project(
    df_project: pd.DataFrame,
    model: Any,
    norm_stats: Optional[Any] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run the Transformer for a single project.

    Args
    ----
    df_project:
        Raw transactions dataframe for ONE project (multiple rows over time).

    model:
        The loaded Transformer model (from transformer_model.pt), already moved
        to CPU and put in eval() mode by ml_core.

    norm_stats:
        Optional np.load(...) object from transformer_norm_stats.npz containing
        'mean' and 'std' arrays, or None.

    Returns
    -------
    (weeks_late, cost_overrun_percent) as floats, or (None, None) if something fails.
    """
    if torch is None or model is None:
        return None, None

    # 1) Sort by time so the sequence is in chronological order
    df_sorted = _sort_by_time(df_project)

    # 2) Build feature matrix [T, F]
    df_feats = _select_feature_columns(df_sorted)

    if df_feats.shape[0] == 0 or df_feats.shape[1] == 0:
        print("[transformer_head] No numeric feature columns available for Transformer.")
        return None, None

    x_np = df_feats.to_numpy(dtype=np.float32)  # [T, F]

    # 3) Apply normalisation if stats are provided
    if norm_stats is not None:
        try:
            # np.load returns an NpzFile; we convert to dict-like.
            if hasattr(norm_stats, "files"):
                stats_dict = {k: norm_stats[k] for k in norm_stats.files}
            else:
                stats_dict = dict(norm_stats)
        except Exception:
            stats_dict = None
    else:
        stats_dict = None

    x_np = _apply_norm_stats(x_np, stats_dict)  # [T, F]

    # 4) Add batch dimension -> [1, T, F]
    x_np = np.expand_dims(x_np, axis=0)
    x_tensor = torch.from_numpy(x_np)  # shape [1, T, F]

    # 5) Run the model
    try:
        model.eval()
        with torch.no_grad():
            output = model(x_tensor)

        # Common patterns:
        #   - output shape [1, 2] -> (weeks, cost)
        #   - output shape [1]    -> weeks only
        if isinstance(output, (list, tuple)):
            output = output[0]

        if isinstance(output, torch.Tensor):
            out_np = output.detach().cpu().numpy()
        else:
            out_np = np.asarray(output)

        out_np = out_np.reshape(-1)

        if out_np.size >= 2:
            weeks_pred = float(out_np[0])
            cost_pred = float(out_np[1])
        elif out_np.size == 1:
            weeks_pred = float(out_np[0])
            cost_pred = 0.0
        else:
            print("[transformer_head] Transformer output is empty.")
            return None, None

        return weeks_pred, cost_pred

    except Exception as exc:  # pragma: no cover
        print(f"[transformer_head] ERROR during Transformer inference: {exc}")
        return None, None
