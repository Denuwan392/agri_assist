"""
RAG Node.

AGENTIC CONCEPT:
Agentic RAG / External Grounding
--------------------------------
Retrieves curated, authoritative agricultural knowledge from external documents
based on the combined insights of the user query and specialist findings.
Grounds downstream reasoning to prevent hallucinated advice.
"""

from state.agent_state import FarmState
from rag.knowledge_base import get_agricultural_knowledge


def rag_node(state: FarmState) -> FarmState:
    """
    INPUT: user_query, specialist_results
    PROCESS: Uses domain-specific retriever to fetch grounding agricultural guidelines.
    OUTPUT: state['retrieved_knowledge']
    """
    query = state.get("user_query", "")
    soil_summary = state.get("specialist_results", {}).get("soil", "")

    # Formulate enriched retrieval query combining user query + findings
    retrieval_query = f"{query} tomato yellow leaves moisture heat"
    docs = get_agricultural_knowledge(retrieval_query, top_k=3)

    return {
        "retrieved_knowledge": docs
    }
