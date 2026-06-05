"""
Task Decomposition Engine.

Breaks down complex travel planning tasks into ordered subtasks.
Each subtask has dependencies and can be executed independently.
"""

from __future__ import annotations

import json

from agent.llm import chat

DECOMPOSE_SYSTEM_PROMPT = """你是一个任务分解专家。将复杂的旅行规划任务拆解为可执行的子任务。

## 拆解原则
1. 每个子任务应该是独立的、可执行的
2. 子任务之间应有明确的依赖关系
3. 按照旅行规划的自然顺序排列

## 子任务类型
- weather_check: 查询天气
- destination_research: 目的地研究介绍
- itinerary_plan: 行程安排
- food_recommend: 美食推荐
- accommodation_recommend: 住宿推荐
- transport_plan: 交通规划
- adjustment: 动态调整（根据天气/人数/老幼等）
- final_synthesis: 最终综合输出

## 输出格式
返回 JSON 数组，每个元素包含:
- id: 子任务ID
- type: 子任务类型
- description: 子任务描述
- depends_on: 依赖的子任务ID列表
- priority: 优先级 (1-5, 1最高)

只返回 JSON 数组，不要其他内容。"""


class TaskDecomposer:
    """Decomposes complex tasks into ordered subtask DAG."""

    async def decompose(self, task: str) -> list[dict]:
        """Break down task into subtasks."""
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请拆解以下旅行规划任务：\n\n{task}"},
        ]

        response = await chat(messages, temperature=0.3)
        return self._parse_subtasks(response)

    def _parse_subtasks(self, response: str) -> list[dict]:
        # Extract JSON array
        try:
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    async def execute(
        self,
        task: str,
        execute_fn,  # async function(subtask) -> result
    ) -> dict:
        """Decompose and execute subtasks in dependency order."""
        subtasks = await self.decompose(task)

        if not subtasks:
            # Fallback: linear decomposition
            subtasks = self._fallback_decompose(task)

        # Topological sort by dependencies
        executed = {}
        results = {}

        while len(executed) < len(subtasks):
            ready = [
                s
                for s in subtasks
                if s["id"] not in executed
                and all(d in executed for d in s.get("depends_on", []))
            ]
            if not ready:
                # Circular dependency or all remaining blocked - execute forcibly
                ready = [s for s in subtasks if s["id"] not in executed]

            # Sort by priority
            ready.sort(key=lambda s: s.get("priority", 3))

            for subtask in ready:
                context = {
                    "task": subtask["description"],
                    "parent_task": task,
                    "previous_results": {
                        dep_id: results.get(dep_id)
                        for dep_id in subtask.get("depends_on", [])
                    },
                }
                try:
                    result = await execute_fn(subtask, context)
                    results[subtask["id"]] = result
                except Exception as e:
                    results[subtask["id"]] = {"error": str(e)}
                executed[subtask["id"]] = True

        return {"subtasks": subtasks, "results": results}

    def _fallback_decompose(self, task: str) -> list[dict]:
        """Linear fallback decomposition."""
        return [
            {"id": "1", "type": "destination_research", "description": f"研究目的地: {task}", "depends_on": [], "priority": 1},
            {"id": "2", "type": "weather_check", "description": "查询出行期间的天气", "depends_on": [], "priority": 1},
            {"id": "3", "type": "itinerary_plan", "description": "规划每日行程", "depends_on": ["1", "2"], "priority": 2},
            {"id": "4", "type": "food_recommend", "description": "推荐当地美食", "depends_on": ["1", "3"], "priority": 3},
            {"id": "5", "type": "accommodation_recommend", "description": "推荐住宿", "depends_on": ["3"], "priority": 3},
            {"id": "6", "type": "transport_plan", "description": "规划交通", "depends_on": ["3"], "priority": 3},
            {"id": "7", "type": "adjustment", "description": "根据天气/人数/老幼动态调整", "depends_on": ["2", "3", "4", "5", "6"], "priority": 4},
            {"id": "8", "type": "final_synthesis", "description": "综合所有结果生成最终旅行规划", "depends_on": ["7"], "priority": 5},
        ]
