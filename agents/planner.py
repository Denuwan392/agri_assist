"""
Planner Agent.

AGENTIC CONCEPT:
Planning & Goal Setting & Exploration
-------------------------------------
Instead of immediately attempting to answer a raw question, an autonomous agent:
1. Translates ambiguous prompts into an explicit, testable 'Goal'.
2. Decomposes the goal into ordered execution steps ('Plan').
3. Identifies missing pieces of information ('Exploration / Discovery').
"""

import json
from state.agent_state import FarmState
from config import select_model


def planner_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query
    PROCESS: Uses Pro Model to analyze the query, define a concrete objective,
             and generate an action plan.
    OUTPUT: state['goal'], state['plan'], state['missing_info']
    """
    query = state.get("user_query", "")

    # Planning requires higher reasoning depth -> Resource-Aware selection: 'complex'
    llm = select_model(task_complexity="complex")

    prompt = (
        f"You are the Lead Planning Agent for GreenLeaf Farm.\n"
        f"The farmer asked: '{query}'\n"
        f"Create a structured plan in valid JSON with keys:\n"
        f"1. 'goal': (string) High-level objective for the farm advisor\n"
        f"2. 'plan': (list of strings) Step-by-step diagnostic and advisory actions\n"
        f"3. 'missing_info': (list of strings) What sensors/data points might be needed\n"
    )

    try:
        response = llm.invoke(prompt)
        parsed = json.loads(str(response.content))
        goal = parsed.get("goal", "Diagnose crop stress and determine safe irrigation requirement")
        plan = parsed.get("plan", [
            "Check soil moisture and nutrient levels",
            "Evaluate ambient weather conditions",
            "Examine crop symptoms against known agricultural conditions",
            "Retrieve verified agronomic guidelines via RAG",
            "Synthesize safe prioritized action plan"
        ])
        missing_info = parsed.get("missing_info", ["Irrigation system status", "Recent rainfall"])
    except Exception:
        goal = "Diagnose tomato leaf yellowing under current environmental conditions"
        plan = [
            "1. Check soil telemetry (moisture, pH)",
            "2. Fetch ambient temperature and heat forecast",
            "3. Analyze foliar symptoms (yellow leaves)",
            "4. Retrieve RAG literature on heat stress and chlorosis",
            "5. Formulate prioritized action plan"
        ]
        missing_info = ["Recent fertilizer application schedule"]

    return {
        "goal": goal,
        "plan": plan,
        "missing_info": missing_info
    }
