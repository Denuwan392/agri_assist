"""
Execution Logger and Monitoring module.

Demonstrates:
- Observability and Monitoring for Agentic AI workflows
- Tracing execution time, models invoked, nodes traversed, and errors
"""

import time
from typing import List, Dict, Any


class AgentRunLogger:
    """Tracks and formats telemetry across the agentic graph run."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log(self, node: str, message: str, status: str = "SUCCESS", model: str = "N/A", extra: Any = None):
        elapsed = round(time.time() - self.start_time, 3)
        event = {
            "elapsed_sec": elapsed,
            "node": node,
            "message": message,
            "status": status,
            "model": model,
            "extra": extra
        }
        self.events.append(event)
        prefix = f"[{node.upper()}]"
        print(f"  {prefix:<16} {message}")

    def summary(self) -> str:
        lines = ["\n================ RUN MONITORING SUMMARY ================"]
        for e in self.events:
            lines.append(f"T+{e['elapsed_sec']:05.2f}s | {e['node']:<14} | {e['status']:<7} | {e['message']}")
        lines.append("========================================================")
        return "\n".join(lines)


# Singleton
logger = AgentRunLogger()
