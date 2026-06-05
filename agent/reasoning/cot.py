"""
Chain of Thought (CoT) Engine.

Generates step-by-step reasoning for tasks that benefit from explicit intermediate steps.
No tool use - pure reasoning chain.
"""

from __future__ import annotations

from agent.llm import chat

COT_SYSTEM_PROMPT = """你是一个旅行规划助手。使用 Chain of Thought (思维链) 模式进行逐步推理。

## 方法
对于用户的问题，请：
1. 首先理解问题的核心
2. 逐步分析各个关键因素
3. 每一步给出明确的推理
4. 最后综合所有分析给出结论

## 分析维度
在旅行规划场景中，请考虑：
- 目的地特点（季节、景点、文化）
- 天气影响
- 出行方式与距离
- 人数规模的影响
- 老人/小孩的特殊需求
- 预算约束

请用中文回答，结构清晰。"""


class CoTEngine:
    """Chain of Thought reasoning engine for step-by-step analysis."""

    async def reason(self, task: str, system_extra: str = "") -> dict:
        system_prompt = COT_SYSTEM_PROMPT
        if system_extra:
            system_prompt += f"\n\n{system_extra}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"请用思维链方式逐步分析以下问题：\n\n{task}\n\n请依次分析每个关键维度，最后给出综合建议。",
            },
        ]

        response = await chat(messages, temperature=0.5)

        # Extract reasoning steps
        steps = self._extract_steps(response)

        return {"answer": response, "reasoning_steps": steps}

    def _extract_steps(self, text: str) -> list[str]:
        """Extract labeled reasoning steps from CoT response."""
        steps = []
        lines = text.split("\n")
        current_step = []
        for line in lines:
            stripped = line.strip()
            if stripped and (
                stripped[0].isdigit() and (". " in stripped or "、" in stripped or "）" in stripped)
                or stripped.startswith("第")
                or stripped.startswith("步骤")
            ):
                if current_step:
                    steps.append(" ".join(current_step))
                current_step = [stripped]
            elif current_step:
                current_step.append(stripped)
        if current_step:
            steps.append(" ".join(current_step))

        return steps if steps else [text]
