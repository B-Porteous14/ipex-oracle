# main.py
#
# FastAPI backend for the IPEX Oracle
# Combines XGBoost + Transformer + GNN + LLM explanation layer
#
# Final output contains:
#  - schedule + cost predictions
#  - trade-risk analysis
#  - blended predictions (xgboost + transformer + gnn)
#  - explanation_text + recommended_actions (LLM)
#

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd

# Core ML Engine (XGBoost + Transformer + GNN)
from training_model.ml_core import make_risk_summary_from_df

# Oracle / LLM Layer
from training_model.llm_layer import generate_oracle_explanation


# ---------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------

app = FastAPI(title="IPEX Oracle API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Helper: Load CSV from raw_projects
# ---------------------------------------------------------------

RAW_PROJECTS_DIR = Path("training_model/raw_projects")

def load_project_csv_as_df(filename: str) -> pd.DataFrame:
    csv_path = RAW_PROJECTS_DIR / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Project CSV not found: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")


# ---------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "IPEX Oracle API running"}


"""FastAPI HTTP API for the IPEX Oracle.

Exposes both the older-style endpoints (/predict/from_csv, /predict/from_db)
and the newer REST-style /project/{filename} endpoint. All routes ultimately
call make_risk_summary_from_df from ml_core and then decorate the result
with LLM explanations.
"""


# ---------------------------------------------------------------
# MAIN ENDPOINT — Predict Risk for a Project CSV (older style)
# ---------------------------------------------------------------

@app.get("/predict/from_csv")
def predict_from_csv(
    filename: str,
    with_story: bool = True,
    audience: str = "builder",
):
    """Predict risk for a project by reading a raw CSV from raw_projects/.

    This mirrors the previous API while using the new ml_core under the hood.
    """
    # 1) Load CSV → DataFrame
    try:
        df = load_project_csv_as_df(filename)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"File not found in raw_projects/: {filename}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load CSV: {e}")

    # 2) Core risk summary
    try:
        risk_summary = make_risk_summary_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML core error: {e}")

    # 3) Optional LLM story
    if with_story:
        try:
            oracle_view = generate_oracle_explanation(risk_summary, audience=audience)
        except Exception as e:
            oracle_view = {
                "explanation_text": "LLM processing failed.",
                "recommended_actions": [],
                "llm_used": False,
                "llm_error": str(e),
            }

        risk_summary["explanation_text"] = oracle_view.get("explanation_text", "")
        risk_summary["recommended_actions"] = oracle_view.get("recommended_actions", [])
        risk_summary["llm_used"] = oracle_view.get("llm_used", False)
        risk_summary["llm_error"] = oracle_view.get("llm_error")

    # Attach source_id for backwards compatibility
    risk_summary.setdefault("source_id", filename)
    return risk_summary


# ---------------------------------------------------------------
# DB-BACKED ENDPOINT (placeholder, as before)
# ---------------------------------------------------------------

@app.get("/predict/from_db")
def predict_from_db(
    project_id: int,
    with_story: bool = True,  # kept for backwards compatibility
    audience: str = "builder",
):
    """Placeholder for a future DB-backed endpoint.

    In production this would query the DB for payment rows, build a
    DataFrame and call make_risk_summary_from_df(df).
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "DB-backed prediction endpoint is not implemented in this "
            "prototype. Use /predict/from_csv or /project/{filename} for now."
        ),
    )


# ---------------------------------------------------------------
# NEWER STYLE ENDPOINT — Predict Risk for a Project CSV
# ---------------------------------------------------------------

@app.get("/project/{filename}")
def analyse_project(filename: str, audience: str = "builder"):

    # 1) Load CSV → DataFrame
    try:
        df = load_project_csv_as_df(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"{filename} not found in raw_projects/")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2) ML Core: Build risk summary (XGBoost + Transformer + GNN)
    try:
        risk_summary = make_risk_summary_from_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML core error: {e}")

    # 3) LLM Explanation Layer
    try:
        llm_output = generate_oracle_explanation(risk_summary, audience=audience)
    except Exception as e:
        llm_output = {
            "explanation_text": "LLM processing failed.",
            "recommended_actions": [],
            "llm_used": False,
            "llm_error": str(e),
        }

    # 4) Merge ML + LLM outputs
    final_output = {
        "project_id": filename.replace(".csv", ""),
        "source_id": filename,
        "predictions": risk_summary["predictions"],
        # risk_buckets / drivers / shap_drivers / trade_risks may not
        # exist in the simplified ml_core; guard with .get.
        "risk_buckets": risk_summary.get("risk_buckets", {}),
        "drivers": risk_summary.get("drivers", {}),
        "shap_drivers": risk_summary.get("shap_drivers", {}),
        "trade_risks": risk_summary.get("trade_risks", []),
        "explanation_text": llm_output["explanation_text"],
        "recommended_actions": llm_output["recommended_actions"],
        "llm_used": llm_output["llm_used"],
        "llm_error": llm_output["llm_error"],
    }

    return final_output


# ---------------------------------------------------------------
# BATCH ENDPOINT — Optional (future expansion)
# ---------------------------------------------------------------

@app.post("/batch")
def batch_projects(files: list[str]):
    results = []
    for f in files:
        try:
            out = analyse_project(f)
            results.append(out)
        except Exception as e:
            results.append({"file": f, "error": str(e)})
    return results
