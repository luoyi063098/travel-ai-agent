"""
Reflexion Engine.

Self-improvement loop: Generate → Evaluate → Reflect → Revise → Re-evaluate.
The agent critiques its own output and iteratively improves it.
"""

from __future__ import annotations

from agent.llm import chat
from config import MAX_REFLECTION_ROUNDS

REFLEXION_GENERATE_PROMPT = """你是一个旅行规划助手。请根据以下要求生成旅行规划。

{task}

{reflection_context}

请生成详细的旅行规划建议。"""

REFLEXION_EVALUATE_PROMPT = """你是一个旅行规划评审专家。请评估以下旅行规划的质量。

规划:
{plan}

评估维度：
1. 行程合理性 (1-10)
2. 天气适配 (1-10)
3. 个性化需求满足度 (1-10)
4. 吃住行全面性 (1-10)

请给出总体评分和各维度评分，以及具体的改进建议。

格式：
Overall: X/10
合理性: X/10, 天气: X/10, 个性化: X/10, 全面性: X/10
改进建议:
1. ...
2. ..."""


class ReflexionEngine:
    """Reflexion engine for self-improving generation."""

    def __init__(self, max_rounds: int = MAX_REFLECTION_ROUNDS):
        # 保存最大反思轮数，默认从配置读取
        self.max_rounds = max_rounds

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """
        执行 Reflexion 自反思循环：
        1. Generate: 生成旅行规划
        2. Evaluate: 评估规划质量
        3. Reflect: 基于评估结果进行反思，总结改进方向
        4. Revise: 将反思结果融入下一轮的生成提示中
        5. 重复直到达到最大轮数或评分足够高

        返回字典，包含：
        - answer: 评分最高的规划文本
        - rounds: 所有轮次的详细记录（规划、评估、评分、反思）
        - final_score: 最终最高评分
        """
        reflections = []  # 记录所有轮次的反思日志
        reflection_context = ""  # 累积的反思上下文，传递给下一轮生成提示
        best_plan = ""  # 记录评分最高的方案
        best_score = 0  # 记录最高评分

        for round_num in range(self.max_rounds):
            # Generate 阶段：根据任务和此前反思上下文生成（或修订）旅行规划
            plan = await self._generate(task, reflection_context)

            # Evaluate 阶段：使用 LLM 评估当前规划的质量
            evaluation = await self._evaluate(plan, task)
            # 从评估文本中解析出 Overall 分数
            score = self._parse_score(evaluation)

            # 如果当前轮次评分高于历史最佳，更新最佳方案和最佳评分
            if score > best_score:
                best_score = score
                best_plan = plan

            # Reflect 阶段：基于评估结果，让模型反思本次规划的不足和改进方向
            reflection = await self._reflect(plan, evaluation, task)
            # 记录本轮完整信息
            reflections.append({
                "round": round_num + 1,
                "plan": plan,
                "evaluation": evaluation,
                "score": score,
                "reflection": reflection,
            })

            # 更新反思上下文，用于下一轮的生成提示
            reflection_context = self._build_reflection_context(reflections)

            # 如果评分 >= 8.5，认为质量已足够高，提前终止循环
            if score >= 8.5:
                break

        # 返回最佳方案（或最后一轮方案）、所有轮次记录及最终评分
        return {
            "answer": best_plan or plan,
            "rounds": reflections,
            "final_score": best_score,
        }

    async def _generate(self, task: str, reflection_context: str) -> str:
        """
        Generate 阶段：
        使用 REFLEXION_GENERATE_PROMPT 模板生成旅行规划。
        如果 reflection_context 非空，将其作为前一方案的反馈和改进方向追加到提示中。
        使用较高温度（0.7）以允许在反思后产生创造性的改进。
        """
        prompt = REFLEXION_GENERATE_PROMPT.format(
            task=task,
            reflection_context=(
                f"\n## 此前方案的反馈与改进方向\n{reflection_context}"
                if reflection_context
                else ""
            ),
        )
        return await chat([{"role": "user", "content": prompt}], temperature=0.7)

    async def _evaluate(self, plan: str, task: str) -> str:
        """
        Evaluate 阶段：
        使用 REFLEXION_EVALUATE_PROMPT 让 LLM 对旅行规划进行多维评分。
        使用较低温度（0.3）以获得更稳定、一致的评估结果。
        返回评估文本（包含评分和改进建议）。
        """
        prompt = REFLEXION_EVALUATE_PROMPT.format(plan=plan)
        return await chat([{"role": "user", "content": prompt}], temperature=0.3)

    async def _reflect(self, plan: str, evaluation: str, task: str) -> str:
        """
        Reflect 阶段：
        基于评估结果，让 LLM 总结本次规划的最大不足和下一轮的改进方向。
        使用中等温度（0.5）以获得有洞察力但不过于发散的分析。
        返回反思文本，包含三个问题的回答（不足之处、改进重点、被忽略的因素）。
        """
        prompt = f"""基于以下评估结果，总结需要改进的关键点。

原始任务: {task}
评估结果: {evaluation}

请列出：
1. 本次规划最大的不足是什么？
2. 下次规划应该重点改进哪些方面？
3. 有哪些被忽略的重要考虑因素？"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.5)

    def _build_reflection_context(self, reflections: list[dict]) -> str:
        """
        从历史反思记录中构建上下文摘要。
        只保留最近 3 轮反思（防止上下文过长），
        每轮记录轮次编号、评分和反思关键点（截取前 300 字符）。
        """
        lines = []
        for r in reflections[-3:]:  # Keep last 3 rounds of reflection
            lines.append(f"\n第{r['round']}轮 (评分: {r['score']}/10):")
            lines.append(f"问题: {r['reflection'][:300]}")
        return "\n".join(lines)

    def _parse_score(self, evaluation: str) -> float:
        """
        从评估文本中解析 Overall 分数。
        优先匹配 "Overall: X/10" 格式；
        如果未找到，退而求其次匹配所有 "X/10" 格式的数字并取平均；
        如果仍然找不到，默认返回 5.0。
        """
        import re
        # 优先匹配 "Overall: X" 或 "Overall: X.X" 格式
        match = re.search(r"Overall:\s*(\d+(?:\.\d+)?)", evaluation)
        if match:
            return float(match.group(1))
        # 如果找不到 Overall 行，尝试查找所有 "X/10" 模式的评分并计算平均值
        scores = re.findall(r"(\d+(?:\.\d+)?)/10", evaluation)
        if scores:
            return sum(float(s) for s in scores) / len(scores)
        # 完全解析失败时返回默认分
        return 5.0
