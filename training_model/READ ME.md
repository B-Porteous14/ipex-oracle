# IPEX Oracle – Construction Brain Core (v0.1)

This project is a prototype "Oracle" for construction projects.  
It predicts:

- Weeks early/late a project will finish.
- Percentage cost overrun/underrun.
- Top 3 highest-risk subcontractors (payee_id + trade) based on simple analytics.

It exposes a FastAPI service that can be called by the IPEX platform or tested via Swagger docs.

---

## Project Structure

- `main.py`  
  FastAPI app (API endpoints).

- `ml_core.py`  
  "Brain" of the Oracle:
  - builds features from raw payment data
  - loads XGBoost models and SHAP explainers
  - returns predictions + SHAP drivers
  - calculates simple `trade_risks` (top 3 risky subcontractors).

- `train_models.py`  
  Training script. Trains two XGBoost models from `training_projects.csv` and saves:
  - `model_weeks_late.pkl`
  - `model_overrun_percent.pkl`
  - `explainer_weeks_late.pkl`
  - `explainer_overrun_percent.pkl`

- `training_projects.csv`  
  One row per completed project. Used for model training.

- `model_weeks_late.pkl`, `model_overrun_percent.pkl`  
  Trained XGBoost regression models.

- `explainer_weeks_late.pkl`, `explainer_overrun_percent.pkl`  
  SHAP explainers for model explainability.

- `raw_projects/`  
  Folder containing raw CSV files for unfinished projects.  
  Each file is a payment history with columns including:
  - `project_value`
  - `project_contract_start_date`
  - `project_completion_date`
  - `trade_work_type`
  - `payee_id`
  - `payee_contract_value`
  - `payment_amount`
  - `payment_date`

---

## Install Dependencies

From this folder:

```bash
pip install -r requirements.txt