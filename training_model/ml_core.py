"""
Minimal ml_core stub for IPEX Oracle.

This version does NOT load any model files. It only provides a simple
make_risk_summary_from_df function so that the FastAPI API can start
successfully in the cloud (Railway) even when model artefacts are missing.

Once deployment is stable, you can replace this with the full ML logic.
"""

from typing import Any, Dict, List
import pandas as pd


def make_risk_summary_from_df(df_project: pd.DataFrame) -> Dict[str, Any]:
    """
    Takes one project's raw transactions dataframe and returns a
    JSON-ready dictionary with dummy predictions.

    This is a temporary stub to keep the API online in environments
    where model files (.pkl, .pt, etc.) are not available.
    """

    # Try to extract a project_id if present; otherwise 'unknown'
    if "project_id" in df_project.columns and len(df_project["project_id"]) > 0:
        project_id = str(df_project["project_id"].iloc[0])
    else:
        project_id = "unknown"

    return {
        "project_id": project_id,
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
        "shap": {
            "weeks_late": [],
            "cost_overrun_percent": [],
        },
        "trade_risks": [],
    }


def batch_predict(df_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Batch version: group by project_id and run make_risk_summary_from_df
    for each project. This mirrors the expected interface used by the API.
    """
    outputs: List[Dict[str, Any]] = []
    if "project_id" not in df_all.columns:
        # no project_id column – treat entire frame as one project
        outputs.append(make_risk_summary_from_df(df_all))
        return outputs

    for pid, group in df_all.groupby("project_id"):
        outputs.append(make_risk_summary_from_df(group))

    return outputs
