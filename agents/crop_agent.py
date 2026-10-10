"""
Crop Pathology Specialist Agent.

AGENTIC CONCEPT:
Multi-Agent Specialization & Botanical Diagnosis
------------------------------------------------
This agent specializes in foliar symptoms, chlorosis patterns, and plant pathology.
It forms differential hypotheses (e.g. nitrogen deficiency vs heat chlorosis vs early blight).
"""

from state.agent_state import FarmState
from config import select_model


def crop_agent_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query, goal
    PROCESS: Analyzes foliar symptoms described in query or observed on tomatoes.
    OUTPUT: state['specialist_results']['crop'], state['a2a_messages']
    """
    user_query = state.get("user_query", "")

    # 1. Reasoning
    llm = select_model(task_complexity="complex")
    prompt = (
        f"You are the Crop Pathology Specialist Agent for GreenLeaf Farm.\n"
        f"Analyze the crop symptoms mentioned in the query: '{user_query}'\n"
        f"Context: Crop is Solanum lycopersicum (Tomato).\n"
        f"Investigate: Causes of yellow leaves (chlorosis), fungal risks (e.g. Early Blight),\n"
        f"and interactions with drought or nitrogen deficiency.\n"
        f"State diagnostic hypotheses clearly."
    )
    response = llm.invoke(prompt)
    analysis_text = str(response.content)

    # 2. Agent-to-Agent (A2A) Message
    a2a_msg = {
        "sender": "crop_agent",
        "receiver": "final_agent",
        "task": "pathology_differential",
        "payload": {
            "symptom": "yellow leaves",
            "primary_hypothesis": "Drought-induced nutrient immobility / nitrogen deficiency",
            "secondary_hypothesis": "Early blight fungal infection (check for target spots)"
        }
    }

    spec_results = dict(state.get("specialist_results", {}))
    spec_results["crop"] = analysis_text

    a2a_list = list(state.get("a2a_messages", []))
    a2a_list.append(a2a_msg)

    return {
        "specialist_results": spec_results,
        "a2a_messages": a2a_list
    }
