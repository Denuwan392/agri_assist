"""
Central State Definition for GreenLeaf Farm Advisor.

AGENTIC CONCEPT:
Shared State (Blackboard Pattern) & Reducers
--------------------------------------------
In LangGraph, all agents, tools, and guardrails read from and write to a single,
strongly-typed state dictionary. This replaces messy global variables or ad-hoc
arguments, enabling:
1. Complete traceability of the agent's reasoning trajectory.
2. Parallel execution merging: when parallel branches run concurrently, keys with
   `Annotated[..., reducer]` automatically combine writes without collision!
3. Checkpointing and time-travel debugging.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to merge dictionaries from concurrent nodes."""
    res = dict(a or {})
    if b:
        res.update(b)
    return res


def append_lists(a: List[Any], b: List[Any]) -> List[Any]:
    """Reducer to concatenate lists from concurrent nodes."""
    return list(a or []) + list(b or [])


class FarmState(TypedDict, total=False):
    # 1. Inputs & Conversation
    user_query: str
    farm_id: str

    # 2. Planning & Goal Setting
    goal: str
    plan: List[str]
    missing_info: List[str]  # Exploration / Discovery: tracks info gaps

    # 3. Routing & Execution Decisions
    route: str

    # 4. Environment & Data Interaction (Tools / External)
    farm_data: Dict[str, Any]
    weather_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    tool_errors: Annotated[List[str], append_lists]      # Exception handling and recovery logs
    react_trace: Annotated[List[Dict[str, Any]], append_lists]  # ReAct-style Action-Observation log

    # 5. RAG (External Agricultural Knowledge)
    retrieved_knowledge: List[str]

    # 6. Specialist Results (Multi-Agent & A2A Messages)
    # Using merge_dicts reducer allows concurrent writes from soil, weather, crop agents!
    specialist_results: Annotated[Dict[str, str], merge_dicts]
    a2a_messages: Annotated[List[Dict[str, Any]], append_lists]

    # 7. Memory Context
    short_term_context: List[str]
    long_term_context: List[Dict[str, Any]]

    # 8. Synthesis, Reflection, Self-Correction
    draft_answer: str
    critique: str
    revision_count: int
    confidence: float

    # 9. Human-in-the-Loop & Safety
    risk_level: str
    requires_human_approval: bool
    human_approved: bool

    # 10. Final Output
    final_answer: str
