"""
Short-Term Memory Manager using LangGraph Checkpointer / Thread state.

Demonstrates:
- Ephemeral in-session memory across conversational turns
- Context injection into current prompt
"""

from typing import List, Dict, Any

class InMemorySessionStore:
    """
    Simulates short-term working memory store for conversational threads.
    In production LangGraph, this is provided by MemorySaver or SqliteSaver checkpointers.
    """
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def format_history_as_context(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous conversational context."
        return "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in history])


_session_store = InMemorySessionStore()

def get_short_term_store() -> InMemorySessionStore:
    return _session_store
