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
        # 保存最大迭代步数，默认从配置中读取，防止无限循环
        self.max_steps = max_steps

    async def reason(
        self,
        task: str,
        tools_description: str,
        call_tool: Callable[[str, dict], Any],  # 接收 (工具名, 参数字典) 并返回执行结果
        system_extra: str = "",
    ) -> dict:
        """
        Execute ReAct loop.

        Returns:
            dict with keys: answer, steps, tool_calls
        """
        # 将 tools_description 和 task 填充到系统提示模板中
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=tools_description, task=task
        )
        if system_extra:
            # 如果有额外的系统级别指令，追加到 prompt 末尾
            system_prompt += f"\n\n{system_extra}"

        # 初始化对话消息列表，第一条为系统级提示
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        steps = []  # 记录每一步的原始输出和解析结果
        tool_calls_made = []  # 记录所有已执行的工具调用

        for step_num in range(self.max_steps):
            # 向 LLM 发送当前完整对话历史，获取模型回复
            response = await chat(messages)

            # 使用正则解析模型回复，提取 Thought / Action / Final Answer 等字段
            parsed = self._parse(response)
            # 将当前步数、原始回复和解析结果记录下来
            steps.append({"step": step_num + 1, "raw": response, "parsed": parsed})

            if parsed["type"] == "final_answer":
                # 模型已给出最终答案，提前结束循环并返回
                return {
                    "answer": parsed["content"],
                    "steps": steps,
                    "tool_calls": tool_calls_made,
                }

            if parsed["type"] == "action":
                # 模型要求调用工具：提取工具名称和参数
                action = parsed["action"]
                action_input = parsed["action_input"]
                tool_calls_made.append({"action": action, "input": action_input})

                # 执行工具调用
                try:
                    # 调用传入的 call_tool 回调，执行实际的工具操作（如查询天气）
                    observation = await call_tool(action, action_input)
                    # 如果 observation 不是字符串，则将其序列化为 JSON 字符串（保持中文字符不被转义）
                    obs_text = (
                        observation
                        if isinstance(observation, str)
                        else json.dumps(observation, ensure_ascii=False)
                    )
                except Exception as e:
                    # 工具调用失败时，将错误信息作为观察结果返回给模型
                    obs_text = f"工具调用错误: {e}"

                # 将模型的本次回复追加到对话历史，作为 assistant 消息
                messages.append({"role": "assistant", "content": response})
                # 将工具调用的观察结果作为 user 消息追加，引导模型继续推理
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: {obs_text}\n\n请继续思考并给出下一步或 Final Answer。",
                    }
                )
                # 在步骤记录中补充观察结果
                steps[-1]["observation"] = obs_text
            else:
                # 模型回复无法解析为任何合法格式（既不是 Final Answer 也不是 Action）
                # 将原始回复追加到对话历史，然后提示模型按正确格式重新回复
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "请按照格式回复: Thought: ... Action: ... Action Input: ... 或 Final Answer: ...",
                    }
                )

        # 达到最大步数限制，模型仍未给出最终答案
        # 强制要求模型基于已有信息给出 Final Answer
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
        """
        解析模型输出文本，识别并提取结构化字段。
        优先级：Final Answer > Action > 未知格式。
        """
        # 首先尝试匹配 Final Answer 格式
        # 正则含义：以 "Final Answer:" 开头，后面跟任意内容（非贪婪），直到遇到两个换行或字符串结尾
        fa_match = re.search(
            r"Final Answer:\s*(.+?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE
        )
        if fa_match:
            # 匹配成功，返回最终答案类型及去除首尾空白的内容
            return {"type": "final_answer", "content": fa_match.group(1).strip()}

        # 尝试匹配 Action 格式
        # 提取 Thought 字段：从 "Thought:" 到换行或行尾
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        # 提取 Action 字段：从 "Action:" 到下一个空白字符（工具名称不包含空格）
        action_match = re.search(r"Action:\s*(\S+)", text, re.IGNORECASE)
        # 提取 Action Input 字段：匹配花括号包裹的 JSON 对象（跨行支持）
        input_match = re.search(
            r"Action Input:\s*(\{.+?\})", text, re.DOTALL | re.IGNORECASE
        )

        if action_match:
            # 至少存在 Action 字段，说明这是一次工具调用请求
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip()
            action_input = {}
            if input_match:
                try:
                    # 尝试将 Action Input 解析为 JSON 字典
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    # 如果 JSON 解析失败，将原始字符串包裹在 {"raw": ...} 中作为 fallback
                    action_input = {"raw": input_match.group(1).strip()}

            return {
                "type": "action",
                "thought": thought,
                "action": action,
                "action_input": action_input,
            }

        # 既不是 Final Answer 也不是 Action，返回未知类型，保留原始文本
        return {"type": "unknown", "content": text}
