"""
Weather Specialist Agent.

AGENTIC CONCEPT:
Multi-Agent Specialization & Tool Calling with Exception Handling & Recovery
----------------------------------------------------------------------------
This agent analyzes microclimate, ambient heat, and evapotranspiration risks.
Demonstrates:
- Resilient Tool Calling (Catches 503 API timeout and falls back to cached telemetry)
- ReAct trace recording
- A2A message formatting
"""

from state.agent_state import FarmState
from tools.farm_database import get_weather_data
from config import select_model, SIMULATE_ERRORS


def weather_agent_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query
    PROCESS:
        1. Tool Call: Invokes `get_weather_data()` with error simulation & fallback
        2. ReAct Observation: Evaluates ambient temperature & humidity
        3. LLM Reasoning: Assesses heat stress impact on tomatoes
        4. A2A Communication: Crafts structured message for Final Agent
    OUTPUT: state['weather_data'], state['specialist_results']['weather'], state['tool_errors']
    """
    location = "GreenLeaf Farm"
    tool_errors = list(state.get("tool_errors", []))

    # 1. ReAct Step: Action
    react_event = {"action": "get_weather_data", "input": {"location": location}}

    # Call weather tool with simulated failure to illustrate resilience
    weather_data = get_weather_data(location=location, simulate_failure=SIMULATE_ERRORS)

    if "_recovered_from_error" in weather_data:
        tool_errors.append(f"Weather API timeout handled: used {weather_data['_source']}")

    react_event["observation"] = {
        "temperature": weather_data.get("temperature"),
        "humidity": weather_data.get("humidity"),
        "source": weather_data.get("_source")
    }

    # 2. Reasoning
    llm = select_model(task_complexity="complex")
    prompt = (
        f"You are the Weather Specialist Agent for GreenLeaf Farm.\n"
        f"Analyze the following weather data:\n"
        f"Temperature: {weather_data.get('temperature')}°C\n"
        f"Humidity: {weather_data.get('humidity')}%\n"
        f"Rain Probability: {weather_data.get('rain_probability')}%\n"
        f"Evaluate atmospheric evapotranspiration and heat stress risks for tomato crops."
    )
    response = llm.invoke(prompt)
    analysis_text = str(response.content)

    # 3. Agent-to-Agent (A2A) Message
    a2a_msg = {
        "sender": "weather_agent",
        "receiver": "final_agent",
        "task": "weather_stress_analysis",
        "payload": {
            "temperature": weather_data.get("temperature"),
            "heat_stress": weather_data.get("temperature", 20) > 32,
            "recommendation": "Elevated water demand due to 34°C ambient heat"
        }
    }

    spec_results = dict(state.get("specialist_results", {}))
    spec_results["weather"] = analysis_text

    a2a_list = list(state.get("a2a_messages", []))
    a2a_list.append(a2a_msg)

    react_trace = list(state.get("react_trace", []))
    react_trace.append(react_event)

    return {
        "weather_data": weather_data,
        "specialist_results": spec_results,
        "a2a_messages": a2a_list,
        "tool_errors": tool_errors,
        "react_trace": react_trace
    }
