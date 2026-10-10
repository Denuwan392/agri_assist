"""
Agricultural Knowledge Base and Simple Keyword/Vector Retriever.

Demonstrates:
- Knowledge Retrieval / Agentic RAG
- External knowledge grounding vs internal model weights
"""

import os
import re
from typing import List, Dict, Any

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agriculture_knowledge.txt")


class AgriculturalRetriever:
    """
    Lightweight, dependency-free text retriever using BM25-style keyword overlap
    and semantic token matching. Can be swapped with Chroma/FAISS.
    """

    def __init__(self, filepath: str = KB_PATH):
        self.documents: List[str] = []
        self._load_documents(filepath)

    def _load_documents(self, filepath: str):
        if not os.path.exists(filepath):
            self.documents = [
                "Tomato yellow leaves can result from nitrogen deficiency, underwatering, or blight.",
                "High temperatures above 32°C dramatically increase plant water demand.",
                "Soil moisture below 30% causes acute tomato root stress."
            ]
            return

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.documents = [line.strip() for line in lines if line.strip()]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves the top_k most relevant agricultural passages for a query.
        """
        # Tokenize query
        query_words = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in self.documents:
            doc_words = set(re.findall(r"\w+", doc.lower()))
            overlap = len(query_words.intersection(doc_words))
            # Boost matches for critical terms
            if "yellow" in query_words and "yellow" in doc_words:
                overlap += 2
            if "moisture" in query_words and "moisture" in doc_words:
                overlap += 2
            if "heat" in query_words or "temperature" in query_words:
                if "temperature" in doc_words or "heat" in doc_words:
                    overlap += 2
            scored_docs.append((overlap, doc))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]


# Singleton instance
_retriever_instance = AgriculturalRetriever()

def get_agricultural_knowledge(query: str, top_k: int = 3) -> List[str]:
    """Helper function to invoke RAG retrieval"""
    return _retriever_instance.retrieve(query, top_k=top_k)
