"""
Router Agent.

AGENTIC CONCEPT:
Routing & Dynamic Dispatch
--------------------------
Routing evaluates the context or query to decide which downstream
specialists or tools are required, avoiding unnecessary latency and costs.

Demonstrates:
- Fast Heuristic check (Rule-based)
- LLM-based intelligent categorization (Flash Model)
"""

import json
from state.agent_state import FarmState
from config import select_model


def router_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query, goal, plan
    PROCESS: Determines whether the question requires:
             - 'soil' only
             - 'weather' only
             - 'crop' only
             - 'all_specialists' (soil + weather + crop)
    OUTPUT: state['route']
    """
    query = state.get("user_query", "").lower()

    # Fast heuristic check
    has_soil = any(w in query for w in ["soil", "moisture", "ph", "ground"])
    has_weather = any(w in query for w in ["weather", "temperature", "rain", "forecast", "heat"])
    has_crop = any(w in query for w in ["leaves", "yellow", "disease", "tomato", "fruit", "pest"])

    # If it touches multiple domains, route to all specialists
    if sum([has_soil, has_weather, has_crop]) >= 2:
        route = "all_specialists"
    elif has_soil:
        route = "soil"
    elif has_weather:
        route = "weather"
    elif has_crop:
        route = "crop"
    else:
        # LLM Router fallback (Resource-Aware: simple task -> Flash)
        llm = select_model(task_complexity="simple")
        prompt = (
            f"Classify this agricultural query into one route:\n"
            f"Query: '{query}'\n"
            f"Allowed routes: 'soil', 'weather', 'crop', 'all_specialists'.\n"
            f"Respond in JSON: {{\"route\": \"<chosen_route>\", \"reasoning\": \"<brief>\"}}"
        )
        try:
            res = llm.invoke(prompt)
            data = json.loads(str(res.content))
            route = data.get("route", "all_specialists")
        except Exception:
            route = "all_specialists"

    return {"route": route}
