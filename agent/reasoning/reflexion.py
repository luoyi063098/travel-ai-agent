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
        self.max_rounds = max_rounds

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """Execute Reflexion loop: generate, evaluate, reflect, revise."""
        reflections = []
        reflection_context = ""
        best_plan = ""
        best_score = 0

        for round_num in range(self.max_rounds):
            # Generate (or revise with reflection)
            plan = await self._generate(task, reflection_context)

            # Evaluate
            evaluation = await self._evaluate(plan, task)
            score = self._parse_score(evaluation)

            if score > best_score:
                best_score = score
                best_plan = plan

            # Reflect
            reflection = await self._reflect(plan, evaluation, task)
            reflections.append({
                "round": round_num + 1,
                "plan": plan,
                "evaluation": evaluation,
                "score": score,
                "reflection": reflection,
            })

            # Update context for next round
            reflection_context = self._build_reflection_context(reflections)

            # Stop if score is high enough
            if score >= 8.5:
                break

        return {
            "answer": best_plan or plan,
            "rounds": reflections,
            "final_score": best_score,
        }

    async def _generate(self, task: str, reflection_context: str) -> str:
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
        prompt = REFLEXION_EVALUATE_PROMPT.format(plan=plan)
        return await chat([{"role": "user", "content": prompt}], temperature=0.3)

    async def _reflect(self, plan: str, evaluation: str, task: str) -> str:
        prompt = f"""基于以下评估结果，总结需要改进的关键点。

原始任务: {task}
评估结果: {evaluation}

请列出：
1. 本次规划最大的不足是什么？
2. 下次规划应该重点改进哪些方面？
3. 有哪些被忽略的重要考虑因素？"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.5)

    def _build_reflection_context(self, reflections: list[dict]) -> str:
        lines = []
        for r in reflections[-3:]:  # Keep last 3 rounds of reflection
            lines.append(f"\n第{r['round']}轮 (评分: {r['score']}/10):")
            lines.append(f"问题: {r['reflection'][:300]}")
        return "\n".join(lines)

    def _parse_score(self, evaluation: str) -> float:
        import re
        match = re.search(r"Overall:\s*(\d+(?:\.\d+)?)", evaluation)
        if match:
            return float(match.group(1))
        # Try to find any score-like pattern
        scores = re.findall(r"(\d+(?:\.\d+)?)/10", evaluation)
        if scores:
            return sum(float(s) for s in scores) / len(scores)
        return 5.0
