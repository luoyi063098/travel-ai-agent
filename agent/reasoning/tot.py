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
        # 保存树的宽度：每层保留的候选节点数
        self.breadth = breadth
        # 保存树的深度：最多扩展多少层
        self.depth = depth

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """Execute Tree of Thoughts reasoning."""
        # Step 1: 初始生成 breadth 个候选方案思路
        candidates = await self._generate_candidates(task, self.breadth)

        # Step 2: 逐层扩展和评估，构建思维树
        tree = []  # 记录树中所有已评估的节点（含得分和深度）
        for d in range(self.depth):
            new_level = []  # 当前深度层的新节点列表
            for candidate in candidates:
                # 对每个候选方案进行细化扩展，得到更详细的内容
                expanded = await self._expand(candidate, task, d + 1)
                # 对扩展后的方案进行评分
                score = await self._evaluate(expanded)
                # 将扩展结果、评分和深度打包存入当前层级
                new_level.append({"thought": expanded, "score": score, "depth": d + 1})

            # 对当前层所有节点按评分从高到低排序
            new_level.sort(key=lambda x: x["score"], reverse=True)
            # 将当前层所有节点追加到全局树记录中
            tree.extend(new_level)
            # 只保留评分最高的 breadth 个方案作为下一轮扩展的父节点
            candidates = [n["thought"] for n in new_level[: self.breadth]]

        # Step 3: 从整棵树中选择评分最高的方案
        best = max(tree, key=lambda x: x["score"]) if tree else {"thought": candidates[0] if candidates else "", "score": 0}

        # Step 4: 基于最佳方案，生成经过润色的最终完整回复
        final = await self._polish(best["thought"], task)

        # 返回最终答案、完整树结构信息和最佳评分
        return {
            "answer": final,
            "tree": tree,
            "best_score": best["score"],
        }

    async def _generate_candidates(self, task: str, n: int) -> list[str]:
        """
        调用 LLM 生成 n 个不同的初始方案思路。
        使用较高温度（0.9）以鼓励多样性和创造性。
        """
        prompt = f"""针对以下旅行规划任务，生成 {n} 个不同的方案思路。每个方案用2-3句话简要描述核心策略。

任务: {task}

请生成 {n} 个方案，编号列出:"""

        response = await chat(
            [{"role": "user", "content": prompt}], temperature=0.9
        )
        # 解析返回文本中的编号列表，提取方案内容
        return self._parse_list(response, n)

    async def _expand(self, thought: str, task: str, depth: int) -> str:
        """
        对给定的方案思路进行细化扩展，生成更具体的行程安排。
        使用中等温度（0.7）在稳定性和创造性之间取得平衡。
        depth 参数用于在提示中说明当前扩展的深度层级。
        """
        prompt = f"""基于以下方案思路，进一步细化（深度 {depth}），给出具体的行程安排。

原始任务: {task}
当前方案: {thought}

请详细展开此方案，包括具体的时间安排、景点、餐饮、住宿建议。"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.7)

    async def _evaluate(self, thought: str) -> float:
        """
        使用 LLM 作为评估器，对方案进行评分（1-10分）。
        使用较低温度（0.3）以获得更稳定、一致的评分结果。
        如果解析失败，返回默认分 5.0。
        """
        prompt = TOT_EVALUATE_PROMPT.format(thought=thought)
        response = await chat([{"role": "user", "content": prompt}], temperature=0.3)
        # 用正则从回复中提取 "Score: X" 格式的分数
        import re
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", response)
        if match:
            return float(match.group(1))
        return 5.0  # default

    async def _polish(self, best_thought: str, task: str) -> str:
        """
        基于最佳方案思路，生成一份完整的、面向用户的旅行规划建议。
        使用中等温度（0.6）以获得自然流畅的文案。
        """
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
        """
        解析 LLM 返回的编号列表文本，提取方案内容。
        支持 "1. xxx" 和 "1、xxx" 两种编号格式。
        如果解析不出任何条目，则返回整段文本。
        """
        items = []
        for line in text.split("\n"):
            stripped = line.strip()
            # 匹配以数字开头且包含 ". " 或 "、" 的行
            if stripped and (stripped[0].isdigit() and (". " in stripped or "、" in stripped)):
                # 去掉编号前缀，提取实际内容
                items.append(stripped.split(". ", 1)[-1].split("、", 1)[-1])
        # 最多返回 n 个条目；如果一个都没解析到，将整段文本作为内容返回
        return items[:n] if items else [text]
