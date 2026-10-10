"""
LangGraph Workflow Definition for GreenLeaf Farm Advisor.

Demonstrates:
- StateGraph compilation
- Conditional Edges (Routing and Reflection loops)
- Parallel execution branches (Soil, Weather, Crop agents running concurrently)
- Clean state synchronization
"""

from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, START, END

from state.agent_state import FarmState
from agents.planner import planner_node
from agents.router import router_node
from agents.soil_agent import soil_agent_node
from agents.weather_agent import weather_agent_node
from agents.crop_agent import crop_agent_node
from agents.critic_agent import critic_node
from agents.final_agent import final_agent_node
from rag.retriever import rag_node
from safety.guardrails import check_input_guardrails
from safety.monitoring import logger


# Node 1: Guardrail Node
def guardrail_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query
    PROCESS: Input validation and security checks
    OUTPUT: state['risk_level']
    """
    query = state.get("user_query", "")
    is_safe, reason = check_input_guardrails(query)
    if not is_safe:
        logger.log("guardrail", f"REJECTED: {reason}", status="FAILED")
        return {
            "final_answer": f"Request cannot be processed by Farm Advisor: {reason}",
            "risk_level": "rejected"
        }
    logger.log("guardrail", "Input passed safety validation", status="SUCCESS")
    return {"risk_level": "low"}


# Conditional Edge: Router dispatcher
def route_decision(state: FarmState) -> Literal["soil_agent", "weather_agent", "crop_agent", "parallel_specialists"]:
    """
    Determines next hop based on router decision.
    """
    route = state.get("route", "all_specialists")
    if route == "soil":
        return "soil_agent"
    elif route == "weather":
        return "weather_agent"
    elif route == "crop":
        return "crop_agent"
    return "parallel_specialists"


# Parallel Fan-Out Node: Dispatches parallel agents
def parallel_dispatcher(state: FarmState) -> FarmState:
    logger.log("parallel_fan_out", "Dispatched Soil, Weather, and Crop specialists in parallel")
    return {}


# Conditional Edge: Reflection Self-Correction Loop
def critic_evaluation(state: FarmState) -> Literal["final_agent", "end"]:
    """
    If the critic found deficiencies and revision_count < 2, loop back to final_agent.
    Otherwise terminate to output.
    """
    conf = state.get("confidence", 0.8)
    rev_count = state.get("revision_count", 0)

    if conf < 0.80 and rev_count < 2:
        logger.log("critic_loop", f"Confidence {conf:.2f} < 0.80 -> Routing back to Final Agent for Revision")
        return "final_agent"

    logger.log("critic_loop", f"Approved (Confidence {conf:.2f}) -> Concluding advisory")
    return "end"


def build_farm_graph():
    """
    Constructs and compiles the complete LangGraph StateGraph.
    """
    builder = StateGraph(FarmState)

    # 1. Register Nodes
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("planner", planner_node)
    builder.add_node("router", router_node)

    # Parallel Specialist Nodes
    builder.add_node("soil_agent", soil_agent_node)
    builder.add_node("weather_agent", weather_agent_node)
    builder.add_node("crop_agent", crop_agent_node)

    # Knowledge & Synthesis Nodes
    builder.add_node("rag", rag_node)
    builder.add_node("final_agent", final_agent_node)
    builder.add_node("critic", critic_node)

    # 2. Add Edges
    builder.add_edge(START, "guardrail")

    # Guardrail Check
    def check_guardrail_status(state: FarmState) -> Literal["planner", "end"]:
        if state.get("risk_level") == "rejected":
            return "end"
        return "planner"

    builder.add_conditional_edges("guardrail", check_guardrail_status, {
        "planner": "planner",
        "end": END
    })

    builder.add_edge("planner", "router")

    # Routing Decision: Can route to single specialist or parallel all
    def dynamic_router_target(state: FarmState) -> List[str]:
        route = state.get("route", "all_specialists")
        if route == "soil":
            return ["soil_agent"]
        elif route == "weather":
            return ["weather_agent"]
        elif route == "crop":
            return ["crop_agent"]
        # Parallel Execution: fan out to all 3 specialist agents simultaneously!
        return ["soil_agent", "weather_agent", "crop_agent"]

    builder.add_conditional_edges("router", dynamic_router_target, [
        "soil_agent", "weather_agent", "crop_agent"
    ])

    # Fan-In: All parallel specialist branches converge into the RAG node
    builder.add_edge("soil_agent", "rag")
    builder.add_edge("weather_agent", "rag")
    builder.add_edge("crop_agent", "rag")

    # RAG -> Final Synthesis Agent
    builder.add_edge("rag", "final_agent")

    # Final Agent -> Critic (Reflection)
    builder.add_edge("final_agent", "critic")

    # Critic -> Conditional Self-Correction Loop or Finish
    builder.add_conditional_edges("critic", critic_evaluation, {
        "final_agent": "final_agent",
        "end": END
    })

    return builder.compile()
