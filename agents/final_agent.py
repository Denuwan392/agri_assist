"""
Final Synthesis Agent.

AGENTIC CONCEPTS:
- Multi-Agent Synthesis & Prompt Chaining
- Prioritization (Ranking urgency: water > heat > disease)
- Long-Term Memory Integration
- Human-in-the-Loop Risk Assessment
"""

from state.agent_state import FarmState
from config import select_model
from safety.guardrails import check_output_guardrails
from memory.long_term import query_long_term_memory


def final_agent_node(state: FarmState) -> FarmState:
    """
    INPUT: specialist_results, retrieved_knowledge, long_term_context, critique
    PROCESS:
        1. Prompt Chaining: Takes outputs of Soil, Weather, Crop agents + RAG documents
        2. Injects Long-Term Memory from previous seasons
        3. Prioritizes actions by urgency (moisture deficit > thermal stress > disease)
        4. Applies Output Guardrails & flags Human-in-the-Loop requirement
    OUTPUT: state['draft_answer'], state['final_answer'], state['risk_level'], state['requires_human_approval']
    """
    spec_results = state.get("specialist_results", {})
    knowledge = state.get("retrieved_knowledge", [])
    critique = state.get("critique", "")
    rev_count = state.get("revision_count", 0)
    farm_id = state.get("farm_id", "farm_01")

    # Retrieve past interventions from SQLite Long-Term Memory
    past_history = query_long_term_memory(farm_id=farm_id, limit=1)

    llm = select_model(task_complexity="complex")

    # Prompt Chaining & Synthesis
    if critique and rev_count > 0:
        # Self-correction revision step
        prompt = (
            f"You are the Lead Farm Advisor revising an advisory based on peer critique.\n"
            f"Previous Critique:\n'{critique}'\n\n"
            f"Revise the draft to address all issues while prioritizing actionable farm steps."
        )
    else:
        prompt = (
            f"You are the Lead Farm Advisor synthesizing specialist reports for GreenLeaf Farm.\n"
            f"Specialist Findings:\n"
            f"- Soil Specialist: {spec_results.get('soil', 'N/A')}\n"
            f"- Weather Specialist: {spec_results.get('weather', 'N/A')}\n"
            f"- Crop Specialist: {spec_results.get('crop', 'N/A')}\n\n"
            f"Retrieved Agricultural Knowledge:\n"
            f"{chr(10).join(f'- {k}' for k in knowledge)}\n\n"
            f"Long-Term Memory (Historical Farm Intervention):\n"
            f"{past_history}\n\n"
            f"Synthesize a prioritized recommendation ranking urgent physical needs first.\n"
            f"Include instruction for irrigation system activation."
        )

    res = llm.invoke(prompt)
    draft_text = str(res.content)

    # Output Guardrail Audit
    is_safe, msg, risk_level = check_output_guardrails(draft_text)
    requires_approval = (risk_level == "high")

    return {
        "draft_answer": draft_text,
        "final_answer": draft_text,
        "risk_level": risk_level,
        "requires_human_approval": requires_approval,
        "long_term_context": past_history
    }
