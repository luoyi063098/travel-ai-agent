"""
Task Decomposition Engine.

Breaks down complex travel planning tasks into ordered subtasks.
Each subtask has dependencies and can be executed independently.
"""

from __future__ import annotations  # 启用延迟求值类型注解，支持在类型提示中使用类名自引用

import json  # 用于解析 LLM 返回的 JSON 格式子任务列表

from agent.llm import chat  # 导入与大语言模型进行对话的异步函数

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
        """
        调用 LLM 将给定任务拆解为子任务列表。
        使用低温度（0.3）以获得更稳定、结构化的输出。
        返回子任务字典列表，每个字典包含 id, type, description, depends_on, priority。
        """
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请拆解以下旅行规划任务：\n\n{task}"},
        ]

        response = await chat(messages, temperature=0.3)
        # 解析 LLM 返回的文本，提取 JSON 格式的子任务列表
        return self._parse_subtasks(response)

    def _parse_subtasks(self, response: str) -> list[dict]:
        """
        从 LLM 返回的文本中提取 JSON 数组并解析为 Python 对象。
        策略：在文本中查找第一个 "[" 和最后一个 "]" 之间的内容作为 JSON 字符串。
        如果解析失败，返回空列表。
        """
        try:
            # Find JSON array in response
            start = response.find("[")  # 定位 JSON 数组的起始位置
            end = response.rfind("]") + 1  # 定位 JSON 数组的结束位置（含右括号）
            if start >= 0 and end > start:
                json_str = response[start:end]  # 截取 JSON 子串
                return json.loads(json_str)  # 解析为 Python 对象（列表 of dict）
        except (json.JSONDecodeError, ValueError):
            # JSON 格式不合法或解析出错，忽略异常并执行 fallback
            pass
        return []

    async def execute(
        self,
        task: str,  # 原始任务描述
        execute_fn,  # 异步执行函数，接收 (subtask, context) 并返回执行结果
    ) -> dict:
        """
        分解任务并按依赖顺序执行所有子任务。
        使用拓扑排序原则：只有所有依赖项都已执行完毕的子任务才能被执行。
        返回字典，包含原始子任务列表和每个子任务的执行结果。
        """
        # 首先尝试通过 LLM 智能拆解任务
        subtasks = await self.decompose(task)

        if not subtasks:
            # 如果 LLM 拆解失败（返回空列表），使用硬编码的线性分解作为 fallback
            subtasks = self._fallback_decompose(task)

        # 按依赖关系进行拓扑排序执行
        executed = {}  # 记录已执行的子任务 ID
        results = {}  # 记录每个子任务的执行结果

        # 循环直到所有子任务都被执行完毕
        while len(executed) < len(subtasks):
            # 找出所有可以执行的子任务：尚未执行且所有依赖项都已完成
            ready = [
                s
                for s in subtasks
                if s["id"] not in executed
                and all(d in executed for d in s.get("depends_on", []))
            ]
            if not ready:
                # 如果没有任何"就绪"的任务，说明存在循环依赖或所有剩余任务因其他原因被阻塞
                # 强行执行所有尚未执行的子任务
                ready = [s for s in subtasks if s["id"] not in executed]

            # 按优先级排序（数字越小优先级越高），确保高优先级任务先被执行
            ready.sort(key=lambda s: s.get("priority", 3))

            for subtask in ready:
                # 构建执行上下文：包含当前子任务的描述、原始任务、以及依赖项的执行结果
                context = {
                    "task": subtask["description"],
                    "parent_task": task,
                    "previous_results": {
                        dep_id: results.get(dep_id)
                        for dep_id in subtask.get("depends_on", [])
                    },
                }
                try:
                    # 执行子任务（调用传入的 execute_fn 回调）
                    result = await execute_fn(subtask, context)
                    results[subtask["id"]] = result
                except Exception as e:
                    # 子任务执行异常时，记录错误信息而非中断整个流程
                    results[subtask["id"]] = {"error": str(e)}
                # 标记该子任务为已执行完毕
                executed[subtask["id"]] = True

        return {"subtasks": subtasks, "results": results}

    def _fallback_decompose(self, task: str) -> list[dict]:
        """
        Fallback 分解方案：
        当 LLM 智能分解失败时，使用预定义的线性任务列表。
        按照旅行规划的自然流程顺序分为 8 个子任务：
        目的地研究 -> 天气查询 -> 行程规划 -> 美食推荐 -> 住宿推荐 -> 交通规划 -> 动态调整 -> 综合输出。
        每个子任务标注了合理的依赖关系，构成有向无环图（DAG）。
        """
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
