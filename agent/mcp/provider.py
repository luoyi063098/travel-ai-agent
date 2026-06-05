from __future__ import annotations

import json
from typing import Any

from models.schemas import MCPCallToolResponse, ToolInfo
from agent.mcp.weather import WeatherTool
from agent.mcp.distance import DistanceTool


class MCPProvider:
    """MCP Protocol provider - manages tool registration and invocation."""

    def __init__(self):
        self._tools: dict[str, Any] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        self.register("weather", WeatherTool())
        self.register("distance", DistanceTool())

    def register(self, name: str, tool: Any):
        self._tools[name] = tool

    def list_tools(self) -> list[ToolInfo]:
        tools = []
        for name, tool in self._tools.items():
            tools.append(
                ToolInfo(
                    name=name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict) -> MCPCallToolResponse:
        if name not in self._tools:
            return MCPCallToolResponse(
                content=[{"type": "text", "text": f"Tool '{name}' not found"}],
                is_error=True,
            )
        try:
            result = await self._tools[name].execute(**arguments)
            return MCPCallToolResponse(
                content=[{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            )
        except Exception as e:
            return MCPCallToolResponse(
                content=[{"type": "text", "text": f"Tool error: {e}"}],
                is_error=True,
            )

    def get_tools_description(self) -> str:
        """Generate tool descriptions for LLM prompt."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
            lines.append(f"  Parameters: {tool.parameters}")
        return "\n".join(lines)


mcp_provider = MCPProvider()
