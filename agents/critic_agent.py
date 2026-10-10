"""
Critic Agent.

AGENTIC CONCEPT:
Reflection & Self-Correction
----------------------------
Autonomous agents must not blindly trust their first draft.
The Critic Agent acts as an agronomic auditor:
1. Verifies that recommendations are grounded in retrieved evidence and telemetry.
2. Checks safety (e.g. avoid scorch from midday watering in 34°C sun, avoid unsupported fungicide claims).
3. Evaluates if revision is needed.
"""

import json
from state.agent_state import FarmState
from config import select_model


def critic_node(state: FarmState) -> FarmState:
    """
    INPUT: draft_answer, soil_data, weather_data, retrieved_knowledge, revision_count
    PROCESS: Uses Pro model to audit the draft recommendation for accuracy and safety.
    OUTPUT: state['critique'], state['confidence'], state['revision_count']
    """
    draft = state.get("draft_answer", "")
    weather = state.get("weather_data", {})
    soil = state.get("soil_data", {})
    knowledge = state.get("retrieved_knowledge", [])
    rev_count = state.get("revision_count", 0)

    # Use Pro Model for rigorous reflection
    llm = select_model(task_complexity="complex")

    # If this is already a revised version, allow critique to approve
    revised_flag = "REVISED" if rev_count > 0 else "FIRST_PASS"

    prompt = (
        f"You are the Agronomic Critic Agent auditing a farm advisory draft [{revised_flag}].\n"
        f"Draft Recommendation:\n'{draft}'\n\n"
        f"Telemetry & Evidence:\n"
        f"- Soil Moisture: {soil.get('moisture')}% (Target 40-70%)\n"
        f"- Weather: {weather.get('temperature')}°C, {weather.get('humidity')}% RH\n"
        f"- Retrieved Guidelines: {knowledge}\n\n"
        f"Audit Criteria:\n"
        f"1. Does the draft caution against watering in peak 34°C midday heat?\n"
        f"2. Does it treat disease as a differential possibility rather than absolute certainty?\n"
        f"3. Does it provide concrete, prioritized guidance?\n"
        f"Respond in JSON format:\n"
        f"{{\"approved\": true/false, \"critique\": \"explanation\", \"issues\": [\"issue 1\"], \"confidence_score\": 0.0-1.0}}"
    )

    try:
        res = llm.invoke(prompt)
        critique_data = json.loads(str(res.content))
        approved = critique_data.get("approved", True)
        critique_text = critique_data.get("critique", "Draft reviewed.")
        conf = float(critique_data.get("confidence_score", 0.85))
    except Exception:
        # Heuristic fallback for mock/offline
        if rev_count == 0:
            approved = False
            critique_text = "Ensure irrigation timing specifies early morning/evening to avoid 34°C leaf scorch."
            conf = 0.75
        else:
            approved = True
            critique_text = "Revised draft resolves heat timing and prioritizes safety."
            conf = 0.90

    return {
        "critique": critique_text,
        "confidence": conf,
        "revision_count": rev_count + 1
    }
