# llm_layer.py
#
# "Oracle voice" layer for the IPEX Oracle.
#
# The rest of your system only calls:
#   generate_oracle_explanation(risk_summary: dict, audience: str = "builder") -> dict
#
# It returns:
#   {
#       "explanation_text": str,
#       "recommended_actions": List[str],
#       "llm_used": bool,
#       "llm_error": Optional[str]
#   }
#
# audience can be: "subcontractor", "builder", "developer" (case-insensitive)
#
# Optional SHAP-style drivers (hook for ml_core):
#   risk_summary["shap_drivers"] = {
#       "weeks_late": [ "High proportion of late payments", ... ],
#       "cost_overrun_percent": [ {"label": "...", "impact": "..."} , ...]
#   }

from typing import Dict, Any, List, Optional
import os


def _format_number(x: float, decimals: int = 1) -> str:
    try:
        return f"{x:.{decimals}f}"
    except Exception:
        return str(x)


def _normalise_audience(audience: Optional[str]) -> str:
    """
    Normalise audience string into one of:
      - 'subcontractor'
      - 'builder'
      - 'developer'
    Default = 'builder'
    """
    if not audience:
        return "builder"
    a = audience.strip().lower()
    if a in {"subbie", "subby", "subcontractor", "sub"}:
        return "subcontractor"
    if a in {"dev", "developer", "fund", "investor"}:
        return "developer"
    # default
    return "builder"


# -------------------------------------------------------------------
#  RULE-BASED ORACLE (fallback)
# -------------------------------------------------------------------

def _generate_rule_based_explanation(risk_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple, hard-coded explanation logic.
    Used when the LLM isn't available or fails.
    """

    # Get headline values with fallbacks to predictions if not available
    headline = risk_summary.get("headline", {})
    preds = risk_summary.get("predictions", {})
    weeks_late = headline.get("weeks_late", preds.get("weeks_late", 0.0))
    overrun_pct = headline.get("cost_overrun_percent", preds.get("cost_overrun_percent", 0.0))

    trade_risks: List[Dict[str, Any]] = risk_summary.get("trade_risks", [])

    # --- Build simple risk messages ---
    # Schedule message
    if weeks_late > 4:
        sched_msg = (
            f"The project is forecast to finish about "
            f"{_format_number(weeks_late)} weeks late."
        )
    elif weeks_late > 1:
        sched_msg = (
            f"The project is forecast to be slightly delayed by about "
            f"{_format_number(weeks_late)} weeks."
        )
    elif weeks_late < -4:
        sched_msg = (
            f"The project is forecast to finish early by about "
            f"{_format_number(abs(weeks_late))} weeks."
        )
    elif weeks_late < -1:
        sched_msg = (
            f"The project is forecast to be slightly ahead of schedule by "
            f"{_format_number(abs(weeks_late))} weeks."
        )
    else:
        sched_msg = (
            "The project is forecast to be close to the planned completion date."
        )

    # Cost message
    if overrun_pct > 10:
        cost_msg = (
            f"Cost risk is high, with a predicted overrun of around "
            f"{_format_number(overrun_pct)}%."
        )
    elif overrun_pct > 3:
        cost_msg = (
            f"Cost risk is moderate, with a predicted overrun of around "
            f"{_format_number(overrun_pct)}%."
        )
    elif overrun_pct < -5:
        cost_msg = (
            f"The project is forecast to underrun the budget by around "
            f"{_format_number(abs(overrun_pct))}%."
        )
    else:
        cost_msg = (
            "Cost performance is forecast to be close to the original budget."
        )

    # Trade risk message
    if trade_risks:
        top_trade = trade_risks[0]
        trade_name = top_trade.get("trade", "a key trade")
        reason = top_trade.get("reason", "elevated risk indicators")
        trade_msg = (
            f"The highest risk subcontractor is in {trade_name}, "
            f"flagged as {reason}."
        )
    else:
        trade_msg = (
            "No specific subcontractor is currently flagged as high risk "
            "based on payment data."
        )

    explanation_text = " ".join([sched_msg, cost_msg, trade_msg])

    # --- Build recommended actions ---
    actions: List[str] = []

    # Schedule actions
    if weeks_late > 2:
        actions.append(
            "Review the construction program and identify critical path "
            "activities that are at risk."
        )
        actions.append(
            "Engage with high-risk trades to confirm resourcing and "
            "sequencing for upcoming works."
        )

    # Cost actions
    if overrun_pct > 5:
        actions.append(
            "Perform a cost review focusing on major trades and any "
            "recent variations."
        )
        actions.append(
            "Check whether contingency allowances are still adequate given "
            "the predicted overrun."
        )

    # Trade-specific actions
    if trade_risks:
        top = trade_risks[0]
        trade_name = top.get("trade", "the highest-risk trade")
        actions.append(
            f"Hold a short risk review meeting with the {trade_name} "
            "subcontractor to confirm scope, progress and upcoming claims."
        )
        if len(trade_risks) > 1:
            actions.append(
                "Monitor the next two highest-risk subcontractors for any "
                "signs of slowing progress or large late claims."
            )

    # Fallback if we somehow didn't add anything
    if not actions:
        actions.append(
            "Continue to monitor progress and costs, but no major risk "
            "intervention is recommended at this stage."
        )

    return {
        "explanation_text": explanation_text,
        "recommended_actions": actions,
    }


# -------------------------------------------------------------------
#  LLM-BASED ORACLE (OpenAI, plain text, tone + SHAP aware)
# -------------------------------------------------------------------

def _extract_driver_lines(risk_summary: Dict[str, Any]) -> List[str]:
    """
    Pulls optional SHAP-like driver summaries out of risk_summary.
    Expected (but optional) structure:

        risk_summary["shap_drivers"] = {
            "weeks_late": [ "High proportion of late payments", ... ],
            "cost_overrun_percent": [
                {"label": "Large variations", "impact": "increases cost"},
                ...
            ]
        }

    Returns a list of text lines to add to the prompt.
    """
    lines: List[str] = []

    shap_drivers = risk_summary.get("shap_drivers") or {}
    if not isinstance(shap_drivers, dict):
        return lines

    # Weeks late drivers
    wl_drivers = shap_drivers.get("weeks_late") or []
    if wl_drivers:
        lines.append("")
        lines.append("Key drivers for predicted delay (weeks late):")
        for d in wl_drivers[:5]:
            if isinstance(d, str):
                lines.append(f"- {d}")
            elif isinstance(d, dict):
                label = d.get("label") or d.get("feature") or "Unnamed driver"
                impact = d.get("impact") or d.get("effect") or ""
                if impact:
                    lines.append(f"- {label} ({impact})")
                else:
                    lines.append(f"- {label}")

    # Cost overrun drivers
    co_drivers = shap_drivers.get("cost_overrun_percent") or []
    if co_drivers:
        lines.append("")
        lines.append("Key drivers for predicted cost overrun percent:")
        for d in co_drivers[:5]:
            if isinstance(d, str):
                lines.append(f"- {d}")
            elif isinstance(d, dict):
                label = d.get("label") or d.get("feature") or "Unnamed driver"
                impact = d.get("impact") or d.get("effect") or ""
                if impact:
                    lines.append(f"- {label} ({impact})")
                else:
                    lines.append(f"- {label}")

    return lines


def _build_llm_prompt(risk_summary: Dict[str, Any], audience: str) -> str:
    """
    Turn the structured risk_summary dict into a plain-text prompt
    for the LLM. Keep it simple and audience-specific.
    """

    # Get headline values with fallbacks to predictions if not available
    headline = risk_summary.get("headline", {})
    preds = risk_summary.get("predictions", {})
    weeks_late = float(headline.get("weeks_late", preds.get("weeks_late", 0.0)))
    overrun_pct = float(headline.get("cost_overrun_percent", preds.get("cost_overrun_percent", 0.0)))

    trade_risks: List[Dict[str, Any]] = risk_summary.get("trade_risks", [])

    # Audience-specific guidance
    if audience == "subcontractor":
        audience_desc = (
            "You are speaking to a subcontractor working on the job. "
            "They care about program, cashflow, and clear expectations. "
            "Use straightforward site language, keep it practical."
        )
    elif audience == "developer":
        audience_desc = (
            "You are speaking to a property developer / client. "
            "They care about overall project risk, delay exposure, and budget impact. "
            "Focus on high-level risk, commercial impact, and mitigation."
        )
    else:  # builder (default)
        audience_desc = (
            "You are speaking to a builder / head contractor project team. "
            "They care about program, subcontractor performance, and cost control. "
            "Use practical construction language and be specific about which trades "
            "and behaviours are driving risk."
        )

    lines: List[str] = []
    lines.append("You are the IPEX Oracle, an assistant for construction project risk.")
    lines.append(audience_desc)
    lines.append("")
    lines.append("Here is the model output for a project:")
    lines.append(f"- Predicted weeks late (negative = early, positive = late): {weeks_late}")
    lines.append(f"- Predicted cost overrun percent (negative = underrun, positive = overrun): {overrun_pct}")

    if trade_risks:
        lines.append("")
        lines.append("Top subcontractor trade risks (from payment behaviour):")
        for i, tr in enumerate(trade_risks[:5], start=1):
            trade_name = tr.get("trade", "Unknown trade")
            reason = tr.get("reason", "elevated risk indicators")
            risk_score = tr.get("risk_score", None)
            if risk_score is not None:
                lines.append(f"  {i}. {trade_name} (risk_score={risk_score}): {reason}")
            else:
                lines.append(f"  {i}. {trade_name}: {reason}")
    else:
        lines.append("")
        lines.append("No specific subcontractor is currently flagged as high risk.")

    # Optional SHAP drivers (if ml_core provided them)
    lines.extend(_extract_driver_lines(risk_summary))

    lines.append("")
    lines.append(
        "Write a short, clear explanation in one or two paragraphs for this audience."
    )
    lines.append(
        "Explain briefly WHY the project is in this position, using the key drivers above "
        "where helpful (for both delay and cost)."
    )
    lines.append("Avoid data science jargon. Make it sound like an experienced PM.")
    lines.append("")
    lines.append("Then on new lines, write 3–6 bullet-point recommended actions,")
    lines.append("each starting with '- '. Example:")
    lines.append("")
    lines.append("Explanation paragraph here.")
    lines.append("")
    lines.append("Recommended actions:")
    lines.append("- Action 1")
    lines.append("- Action 2")
    lines.append("- Action 3")

    return "\n".join(lines)


def _parse_llm_output(raw_text: str) -> Dict[str, Any]:
    """
    Simple parser:
    - If it finds a line 'Recommended actions:' it splits there.
    - Lines starting with '- ' after that become recommended_actions.
    - Everything before is explanation_text.
    - We then clean both explanation and actions to strip newlines / double spaces.
    """
    text = raw_text.strip()
    if not text:
        return {
            "explanation_text": "",
            "recommended_actions": [],
        }

    lines = text.splitlines()

    # Try to find "Recommended actions:" line (case-insensitive)
    split_idx = None
    for i, line in enumerate(lines):
        if "recommended actions" in line.lower():
            split_idx = i
            break

    if split_idx is None:
        # No obvious split; treat all as explanation
        explanation_text = " ".join(text.split())
        return {
            "explanation_text": explanation_text,
            "recommended_actions": [],
        }

    expl_lines = lines[:split_idx]
    action_lines = lines[split_idx + 1 :]

    explanation_text = "\n".join(expl_lines).strip()

    actions: List[str] = []
    for line in action_lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            actions.append(stripped[2:].strip())
        elif stripped.startswith("• "):
            actions.append(stripped[2:].strip())

    # Clean up newlines / extra spaces
    explanation_text = " ".join(explanation_text.split())
    actions = [" ".join(a.split()) for a in actions if a]

    # Keep at most 6 actions
    actions = actions[:6]

    return {
        "explanation_text": explanation_text,
        "recommended_actions": actions,
    }


def _generate_llm_explanation(
    risk_summary: Dict[str, Any],
    audience: str,
) -> Dict[str, Any]:
    """
    Call a real LLM via OpenAI and interpret its plain-text output.
    """

    # Import here so we get a clear error if the library is missing
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai library is not installed.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Use a solid default model name
    model_name = os.getenv("ORACLE_LLM_MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key)

    prompt = _build_llm_prompt(risk_summary, audience=audience)

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )

    # response.output_text is a helper to get the whole text
    raw_text = response.output_text  # type: ignore[attr-defined]

    parsed = _parse_llm_output(raw_text)

    return {
        "explanation_text": parsed["explanation_text"],
        "recommended_actions": parsed["recommended_actions"],
    }


# -------------------------------------------------------------------
#  PUBLIC ENTRY POINT
# -------------------------------------------------------------------

def generate_oracle_explanation(
    risk_summary: Dict[str, Any],
    audience: str = "builder",
) -> Dict[str, Any]:
    """
    Main entry point used by the rest of the app.

    - Tries to use the real LLM.
    - If anything fails (no key, network error, parse error, etc.),
      falls back to the rule-based explanation.
    - Always returns:
        llm_used = True/False
        llm_error = error message when fallback is used (for debugging)
    """
    audience_norm = _normalise_audience(audience)

    try:
        output = _generate_llm_explanation(risk_summary, audience=audience_norm)
        output["llm_used"] = True
        output["llm_error"] = None
        return output
    except Exception as e:
        error_msg = str(e)
        fallback = _generate_rule_based_explanation(risk_summary)
        fallback["llm_used"] = False
        fallback["llm_error"] = error_msg
        return fallback
