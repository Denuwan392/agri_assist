"""
Soil Specialist Agent.

AGENTIC CONCEPT:
Multi-Agent Specialization & Tool Calling & ReAct Action Loop & A2A Messages
----------------------------------------------------------------------------
This agent is an expert in pedology, soil moisture, and pH.
It actively executes tools, interprets sensor values against agronomic thresholds,
and formats its findings as a structured Agent-to-Agent (A2A) message.
"""

from state.agent_state import FarmState
from tools.farm_database import get_soil_data
from config import select_model


def soil_agent_node(state: FarmState) -> FarmState:
    """
    INPUT: farm_id, user_query
    PROCESS:
        1. Tool Call: Invokes `get_soil_data(farm_id)`
        2. ReAct Observation: Interprets telemetry
        3. LLM Reasoning: Analyzes deficits
        4. A2A Communication: Crafts structured message for Final Agent
    OUTPUT: state['soil_data'], state['specialist_results']['soil'], state['a2a_messages']
    """
    farm_id = state.get("farm_id", "farm_01")

    # 1. ReAct Step: Action
    react_event = {"action": "get_soil_data", "input": {"farm_id": farm_id}}
    soil_data = get_soil_data(farm_id)
    react_event["observation"] = soil_data

    # 2. Reasoning with Specialist Persona
    llm = select_model(task_complexity="complex")
    prompt = (
        f"You are the Soil Specialist Agent for GreenLeaf Farm.\n"
        f"Analyze the following soil data:\n"
        f"Moisture: {soil_data.get('moisture')}%\n"
        f"pH: {soil_data.get('ph')}\n"
        f"Nitrogen: {soil_data.get('nitrogen_ppm')} ppm\n"
        f"Target Tomato Moisture Range: 40% - 70%\n"
        f"Provide a concise technical assessment of soil conditions and irrigation urgency."
    )
    response = llm.invoke(prompt)
    analysis_text = str(response.content)

    # 3. Agent-to-Agent (A2A) Structured Message
    a2a_msg = {
        "sender": "soil_agent",
        "receiver": "final_agent",
        "task": "soil_telemetry_analysis",
        "payload": {
            "moisture": soil_data.get("moisture"),
            "status": "critical_deficit" if soil_data.get("moisture", 50) < 30 else "optimal",
            "recommendation": "Urgent drip irrigation required"
        }
    }

    # Retrieve existing state containers
    spec_results = dict(state.get("specialist_results", {}))
    spec_results["soil"] = analysis_text

    a2a_list = list(state.get("a2a_messages", []))
    a2a_list.append(a2a_msg)

    react_trace = list(state.get("react_trace", []))
    react_trace.append(react_event)

    return {
        "soil_data": soil_data,
        "specialist_results": spec_results,
        "a2a_messages": a2a_list,
        "react_trace": react_trace
    }
