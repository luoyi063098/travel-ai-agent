"""
MCP（Model Context Protocol）提供商模块。
负责工具的注册、列表查询、调用执行和描述生成。
"""

from __future__ import annotations                      # 启用类型注解的延迟求值

import json                                              # JSON 序列化，用于将工具执行结果转为字符串
from typing import Any                                   # 任意类型，用于工具对象类型标注

from models.schemas import MCPCallToolResponse, ToolInfo  # MCP 协议数据模型：调用响应和工具信息
from agent.mcp.weather import WeatherTool                 # 天气查询工具
from agent.mcp.distance import DistanceTool               # 城市间距离计算工具


class MCPProvider:
    """
    MCP 协议提供商 —— 负责工具的生命周期管理。

    职责包括：
    - 注册内置工具
    - 暴露可用工具列表
    - 根据名称和参数执行具体工具
    - 生成供 LLM 使用的工具描述文本
    """

    def __init__(self):
        """初始化时创建一个空的工具字典，并注册所有内置工具。"""
        self._tools: dict[str, Any] = {}                  # 工具名称到工具实例的映射字典
        self._register_builtin_tools()                    # 注册内置的天气和距离工具

    def _register_builtin_tools(self):
        """注册所有内置工具到工具字典中。此处硬编码了天气和距离两个工具。"""
        self.register("weather", WeatherTool())            # 注册名称为 "weather" 的天气查询工具
        self.register("distance", DistanceTool())          # 注册名称为 "distance" 的距离计算工具

    def register(self, name: str, tool: Any):
        """
        将一个新工具注册到提供商中。

        参数：
            name: 工具的唯一标识名称
            tool: 工具实例，需包含 description 和 parameters 属性以及 execute 方法
        """
        self._tools[name] = tool                           # 按名称存入字典

    def list_tools(self) -> list[ToolInfo]:
        """
        返回所有已注册工具的元信息列表。

        每个 ToolInfo 包含工具的名称、描述和参数 Schema，
        可供客户端（如 HTTP 接口）展示或调用。
        """
        tools = []
        for name, tool in self._tools.items():             # 遍历已注册的所有工具
            tools.append(
                ToolInfo(
                    name=name,                             # 工具名称
                    description=tool.description,           # 工具功能描述
                    parameters=tool.parameters,             # 工具接受的参数 JSON Schema
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: dict) -> MCPCallToolResponse:
        """
        根据名称查找并执行指定的工具。

        流程：
        1. 检查工具是否存在，不存在则返回错误响应
        2. 调用工具的 execute 方法并传入参数
        3. 将执行结果序列化为 JSON 字符串
        4. 捕获执行过程中的任何异常，返回错误响应
        """
        if name not in self._tools:                        # 工具名称未注册
            return MCPCallToolResponse(
                content=[{"type": "text", "text": f"Tool '{name}' not found"}],  # 返回工具不存在提示
                is_error=True,                             # 标记为错误响应
            )
        try:
            # 异步执行对应工具的 execute 方法，传入解包后的参数
            result = await self._tools[name].execute(**arguments)
            # 将执行结果序列化为 JSON（保留中文，不加转义）并包装为协议响应
            return MCPCallToolResponse(
                content=[{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
            )
        except Exception as e:
            # 捕获所有异常（如参数错误、网络异常等），返回友好错误消息
            return MCPCallToolResponse(
                content=[{"type": "text", "text": f"Tool error: {e}"}],
                is_error=True,                             # 标记为错误响应
            )

    def get_tools_description(self) -> str:
        """
        生成可供 LLM 理解的工具描述文本。

        格式为每行一个工具，包含名称、描述和参数列表，
        通常用于构建 system prompt 让 LLM 知道可用工具。
        """
        lines = []
        for name, tool in self._tools.items():             # 遍历所有已注册工具
            lines.append(f"- {name}: {tool.description}")  # 工具名称和功能描述
            lines.append(f"  Parameters: {tool.parameters}")  # 工具接受的参数说明
        return "\n".join(lines)                            # 用换行符拼接为完整文本


# 模块级单例：应用全局共享同一个 MCPProvider 实例
mcp_provider = MCPProvider()
