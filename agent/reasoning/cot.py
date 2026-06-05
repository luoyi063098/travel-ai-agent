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
        """
        执行 Chain of Thought 推理流程。
        向 LLM 发送带有 CoT 指令的提示，获取逐步推理的回复，然后从中提取各个推理步骤。

        返回字典，包含:
        - answer: 模型生成的完整原始回复
        - reasoning_steps: 从回复中解析出的各个推理步骤列表
        """
        system_prompt = COT_SYSTEM_PROMPT
        if system_extra:
            # 如果有额外的系统级指令，追加到 CoT 提示末尾
            system_prompt += f"\n\n{system_extra}"

        # 构建对话消息列表：系统级指令 + 用户问题
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"请用思维链方式逐步分析以下问题：\n\n{task}\n\n请依次分析每个关键维度，最后给出综合建议。",
            },
        ]

        # 调用 LLM，使用较低温度（0.5）以获得更稳定、逻辑性更强的推理输出
        response = await chat(messages, temperature=0.5)

        # 从模型的回复文本中提取标注的推理步骤
        steps = self._extract_steps(response)

        # 返回完整回答和结构化后的推理步骤
        return {"answer": response, "reasoning_steps": steps}

    def _extract_steps(self, text: str) -> list[str]:
        """
        从 Chain of Thought 回复文本中提取带有编号或步骤标记的推理步骤。

        识别规则：
        - 以数字开头且后面跟 ". "、"、" 或 "）" 的行，视为新步骤的起始
        - 以 "第" 或 "步骤" 开头的行，也视为新步骤的起始
        - 不属于步骤起始行的其他内容，归入当前正在构建的步骤中
        - 如果最终没有任何步骤被识别出来，则将整段文本作为一个步骤返回
        """
        steps = []  # 存放所有提取出的步骤
        lines = text.split("\n")  # 按换行分割文本
        current_step = []  # 暂存当前正在构建的步骤的各行内容
        for line in lines:
            stripped = line.strip()
            # 判断当前行是否为新的推理步骤的开始
            if stripped and (
                stripped[0].isdigit() and (". " in stripped or "、" in stripped or "）" in stripped)
                or stripped.startswith("第")  # 匹配 "首先"、"第一"、"第二步" 等中文步骤标记
                or stripped.startswith("步骤")  # 匹配 "步骤1"、"步骤一" 等格式
            ):
                if current_step:
                    # 将之前累积的步骤内容拼接为一个字符串，存入结果列表
                    steps.append(" ".join(current_step))
                # 开始一个新步骤，当前行作为第一行
                current_step = [stripped]
            elif current_step:
                # 当前行不是新步骤的开始，且当前存在正在构建的步骤，则追加到当前步骤
                current_step.append(stripped)
        # 循环结束后，处理最后一个累积的步骤
        if current_step:
            steps.append(" ".join(current_step))

        # 如果没有识别出任何步骤，则返回整段文本作为唯一的一个步骤
        return steps if steps else [text]
