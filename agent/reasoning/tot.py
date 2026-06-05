"""
Tree of Thoughts (ToT) Engine.

Explores multiple reasoning paths in parallel, evaluates them, and selects the best.
Good for comparison tasks: A vs B, multiple itinerary options, etc.
"""

from __future__ import annotations

import asyncio

from agent.llm import chat
from config import TOT_BREADTH, TOT_DEPTH

TOT_EVALUATE_PROMPT = """你是一个旅行规划评估专家。请评估以下方案的质量。

评分标准 (1-10分):
- 合理性: 行程是否合理可行
- 天气适配: 是否考虑了天气因素
- 个性化: 是否考虑了用户需求（人数、老人小孩、出行方式）
- 全面性: 是否覆盖吃住行游

方案:
{thought}

请给出评分和简短理由。格式:
Score: X/10
Reason: ..."""


class ToTEngine:
    """Tree of Thoughts engine for multi-path reasoning."""

    def __init__(self, breadth: int = TOT_BREADTH, depth: int = TOT_DEPTH):
        self.breadth = breadth
        self.depth = depth

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """Execute Tree of Thoughts reasoning."""
        # Step 1: Generate candidate approaches
        candidates = await self._generate_candidates(task, self.breadth)

        # Step 2: For each depth level, expand and evaluate
        tree = []
        for d in range(self.depth):
            new_level = []
            for candidate in candidates:
                # Expand this candidate
                expanded = await self._expand(candidate, task, d + 1)
                # Evaluate
                score = await self._evaluate(expanded)
                new_level.append({"thought": expanded, "score": score, "depth": d + 1})

            # Keep top breadths
            new_level.sort(key=lambda x: x["score"], reverse=True)
            tree.extend(new_level)
            candidates = [n["thought"] for n in new_level[: self.breadth]]

        # Step 3: Select best
        best = max(tree, key=lambda x: x["score"]) if tree else {"thought": candidates[0] if candidates else "", "score": 0}

        # Step 4: Generate final polished answer
        final = await self._polish(best["thought"], task)

        return {
            "answer": final,
            "tree": tree,
            "best_score": best["score"],
        }

    async def _generate_candidates(self, task: str, n: int) -> list[str]:
        prompt = f"""针对以下旅行规划任务，生成 {n} 个不同的方案思路。每个方案用2-3句话简要描述核心策略。

任务: {task}

请生成 {n} 个方案，编号列出:"""

        response = await chat(
            [{"role": "user", "content": prompt}], temperature=0.9
        )
        return self._parse_list(response, n)

    async def _expand(self, thought: str, task: str, depth: int) -> str:
        prompt = f"""基于以下方案思路，进一步细化（深度 {depth}），给出具体的行程安排。

原始任务: {task}
当前方案: {thought}

请详细展开此方案，包括具体的时间安排、景点、餐饮、住宿建议。"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.7)

    async def _evaluate(self, thought: str) -> float:
        prompt = TOT_EVALUATE_PROMPT.format(thought=thought)
        response = await chat([{"role": "user", "content": prompt}], temperature=0.3)
        # Parse score
        import re
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", response)
        if match:
            return float(match.group(1))
        return 5.0  # default

    async def _polish(self, best_thought: str, task: str) -> str:
        prompt = f"""基于以下最佳方案，生成一份完整的旅行规划建议。

任务: {task}
最佳方案思路: {best_thought}

请用中文生成一份完整的旅行规划，包括：
1. 总体概述
2. 每日行程安排
3. 美食推荐
4. 住宿建议
5. 交通建议
6. 注意事项"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.6)

    def _parse_list(self, text: str, n: int) -> list[str]:
        items = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() and (". " in stripped or "、" in stripped)):
                items.append(stripped.split(". ", 1)[-1].split("、", 1)[-1])
        return items[:n] if items else [text]
