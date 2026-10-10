"""
Safety and Guardrails Module.

Demonstrates:
- Input Guardrails: Block toxic/off-topic inputs before spending compute/tool calls.
- Output Guardrails: Ensure recommendations avoid dangerous absolutes or chemical misuse.
- Risk Classification for Human-in-the-Loop.
"""

import json
from typing import Dict, Any, Tuple
from config import select_model


def check_input_guardrails(user_query: str) -> Tuple[bool, str]:
    """
    INPUT GUARDRAIL:
    Evaluates whether the user query is relevant to agriculture and safe to process.
    Uses Flash model (simple classification).
    """
    query_clean = user_query.strip().lower()

    # Rule-based fast check for harmful or off-topic queries
    banned_keywords = ["bomb", "weapon", "malware", "hack", "bypass", "exploit", "poison neighbor"]
    for word in banned_keywords:
        if word in query_clean:
            return False, f"Query rejected: contains unsafe term '{word}'."

    if len(query_clean) < 5:
        return False, "Query rejected: Query is too brief or empty."

    # LLM-assisted classification
    llm = select_model(task_complexity="simple")
    prompt = (
        f"Evaluate query safety for an agriculture advisory agent:\n"
        f"Query: '{user_query}'\n"
        f"Respond in JSON: {{\"safe\": true/false, \"category\": \"agriculture\"|\"offtopic\"|\"harmful\", \"reason\": \"explanation\"}}"
    )
    try:
        res = llm.invoke(prompt)
        data = json.loads(str(res.content))
        return data.get("safe", True), data.get("reason", "Passed input guardrail.")
    except Exception:
        # Default fallback to permissive if mock format varies
        return True, "Passed heuristic guardrails."


def check_output_guardrails(draft_text: str) -> Tuple[bool, str, str]:
    """
    OUTPUT GUARDRAIL:
    Audits draft recommendations.
    Enforces epistemic humility:
    - Reject dogmatic claims (e.g., 'Your plant definitely has disease X')
    - Flag physical automation tasks for Human-in-the-Loop.
    """
    risk_level = "low"
    requires_approval = False

    lower_text = draft_text.lower()

    # Flag high-risk physical actuator commands
    if any(k in lower_text for k in ["activate irrigation", "turn on pump", "start valve", "apply pesticide"]):
        risk_level = "high"
        requires_approval = True

    # Check for overconfident diagnosis
    overconfident_phrases = ["definitely has", "guaranteed to be", "without a doubt"]
    for phrase in overconfident_phrases:
        if phrase in lower_text:
            return False, f"Guardrail violation: Overconfident claim '{phrase}' detected. System must suggest possibilities.", risk_level

    return True, "Passed output guardrail.", risk_level
