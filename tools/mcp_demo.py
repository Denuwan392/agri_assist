"""
Model Context Protocol (MCP) Conceptual Demonstration.

Demonstrates:
- What MCP is vs standard Tool Calling
- A standalone, educational Mock MCP Server exposing tools and resources
"""

from typing import Dict, Any, List


class MockMCPServer:
    """
    Conceptual demonstration of a Model Context Protocol (MCP) server.

    WHAT IS MCP?
    - Tool Calling is an LLM feature: The model outputs a JSON function signature.
    - MCP is a standard open protocol (client-server architecture) that allows external
      data sources, devices, and tool providers to expose tools, resources, and prompts
      consistently across different AI clients without custom glue code.
    """

    def __init__(self, name: str = "greenleaf-farm-mcp-server"):
        self.name = name
        self.tools = {
            "get_soil_telemetry": {
                "description": "Fetch real-time moisture, pH, and nitrogen data for a farm.",
                "parameters": {"farm_id": "string"}
            },
            "get_weather_telemetry": {
                "description": "Fetch live ambient temperature and rain probability.",
                "parameters": {"location": "string"}
            }
        }

    def list_tools(self) -> Dict[str, Any]:
        """MCP endpoint: tools/list"""
        return self.tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP endpoint: tools/call"""
        if tool_name == "get_soil_telemetry":
            return {"content": [{"type": "text", "text": "Moisture: 28%, pH: 6.4, Nitrogen: 32ppm"}]}
        elif tool_name == "get_weather_telemetry":
            return {"content": [{"type": "text", "text": "Temp: 34°C, Humidity: 68%, Rain: 10%"}]}
        raise ValueError(f"Unknown MCP tool: {tool_name}")


def demonstrate_mcp_concept():
    server = MockMCPServer()
    print("  [MCP SERVER] Discovered server:", server.name)
    print("  [MCP TOOLS AVAILABLE]:", list(server.list_tools().keys()))
    res = server.call_tool("get_soil_telemetry", {"farm_id": "farm_01"})
    print("  [MCP TOOL CALL RESPONSE]:", res["content"][0]["text"])
