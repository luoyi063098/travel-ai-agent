"""
ReAct (Reasoning + Acting) Engine.

The ReAct pattern interleaves reasoning steps and tool-calling actions:
  Thought → Action → Observation → Thought → ... → Final Answer

This is the default strategy for tasks requiring tool use (weather lookups, etc.).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.llm import chat
from config import MAX_REACT_STEPS


REACT_SYSTEM_PROMPT = """你是一个资深的旅行规划顾问，拥有丰富的旅行策划经验。使用 ReAct 模式来思考和行动。

## 可用工具
{tools_description}

## 响应格式
每次回复必须严格遵循以下格式之一：

```
Thought: <你的推理过程，分析当前信息和下一步>
Action: <工具名称>
Action Input: <JSON 参数>
```

或最终回答：

```
Thought: <你的最终推理总结>
Final Answer: <面向用户的友好回答>
```

## 核心原则
- 每次只能调用一个工具，等待观察结果后再决定下一步
- 查询天气后，根据天气状况调整建议（雨天推室内、高温避午间）
- 兼顾交通便利性、预算约束、群体需求（老人/小孩）
- 推荐时给出理由，不只给结论
- 回答使用中文，风格温暖专业，像真正的旅行顾问
- 可引用具体数据（温度、距离、价格）增强说服力

## 旅行规划要点
- 景点推荐考虑：季节适宜性、地理位置、开放时间、适合人群
- 餐饮推荐考虑：当地特色、口味适配、人均价位、是否需预订
- 住宿推荐考虑：交通便利、周边配套、房型适合、噪音/安静
- 交通推荐考虑：时间成本、费用、舒适度、换乘便利性

当前任务: {task}"""


class ReActEngine:
    """ReAct reasoning engine that alternates reasoning with tool actions."""

    def __init__(self, max_steps: int = MAX_REACT_STEPS):
        self.max_steps = max_steps

    async def reason(
        self,
        task: str,
        tools_description: str,
        call_tool: Callable[[str, dict], Any],
        system_extra: str = "",
    ) -> dict:
        """
        Execute ReAct loop.

        Returns:
            dict with keys: answer, steps, tool_calls
        """
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=tools_description, task=task
        )
        if system_extra:
            system_prompt += f"\n\n{system_extra}"

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        steps = []
        tool_calls_made = []

        for step_num in range(self.max_steps):
            response = await chat(messages)

            # Parse response
            parsed = self._parse(response)
            steps.append({"step": step_num + 1, "raw": response, "parsed": parsed})

            if parsed["type"] == "final_answer":
                return {
                    "answer": parsed["content"],
                    "steps": steps,
                    "tool_calls": tool_calls_made,
                }

            if parsed["type"] == "action":
                action = parsed["action"]
                action_input = parsed["action_input"]
                tool_calls_made.append({"action": action, "input": action_input})

                # Execute tool
                try:
                    observation = await call_tool(action, action_input)
                    obs_text = (
                        observation
                        if isinstance(observation, str)
                        else json.dumps(observation, ensure_ascii=False)
                    )
                except Exception as e:
                    obs_text = f"工具调用错误: {e}"

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: {obs_text}\n\n请继续思考并给出下一步或 Final Answer。",
                    }
                )
                steps[-1]["observation"] = obs_text
            else:
                # Unparseable response - push it back
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "请按照格式回复: Thought: ... Action: ... Action Input: ... 或 Final Answer: ...",
                    }
                )

        # Max steps reached - force final answer
        messages.append(
            {
                "role": "user",
                "content": "已达到最大步骤数，请基于当前信息给出最终的 Final Answer。",
            }
        )
        final_response = await chat(messages)
        return {
            "answer": final_response,
            "steps": steps,
            "tool_calls": tool_calls_made,
        }

    def _parse(self, text: str) -> dict:
        # Try Final Answer first
        fa_match = re.search(
            r"Final Answer:\s*(.+?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE
        )
        if fa_match:
            return {"type": "final_answer", "content": fa_match.group(1).strip()}

        # Try Action
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        action_match = re.search(r"Action:\s*(\S+)", text, re.IGNORECASE)
        input_match = re.search(
            r"Action Input:\s*(\{.+?\})", text, re.DOTALL | re.IGNORECASE
        )

        if action_match:
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip()
            action_input = {}
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {"raw": input_match.group(1).strip()}

            return {
                "type": "action",
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }

        # Fallback - treat as partial
        return {"type": "unknown", "content": text}
