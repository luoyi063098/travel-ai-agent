"""
Monte Carlo Tree Search (MCTS) Engine.

Uses MCTS to explore and optimize travel plans through iterative tree search.
Good for optimization problems like route planning, itinerary optimization.

MCTS steps: Selection → Expansion → Simulation → Backpropagation
"""

from __future__ import annotations

import math
import random

from agent.llm import chat
from config import MCTS_ITERATIONS

MCTS_EVALUATE_PROMPT = """评估以下旅行计划的质量（1-10分），只返回数字。

计划: {plan}

考虑因素：合理性、天气适配、个性化需求、全面性。

Score:"""


class MCTSNode:
    __slots__ = ("state", "parent", "children", "visits", "value", "action")

    def __init__(self, state: str, parent=None, action: str = ""):
        self.state = state
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.action = action

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term

    def best_child(self) -> "MCTSNode | None":
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value / max(c.visits, 1))


class MCTSEngine:
    """Monte Carlo Tree Search for travel plan optimization."""

    def __init__(self, iterations: int = MCTS_ITERATIONS):
        self.iterations = iterations

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """Execute MCTS to find best travel plan."""
        # Initialize root
        root = MCTSNode(state="初始行程方案")

        # Run MCTS iterations
        for i in range(self.iterations):
            node = self._select(root)
            if node.visits > 0 and not node.children:
                await self._expand(node, task)
            child = self._select_child_to_simulate(node)
            score = await self._simulate(child, task)
            self._backpropagate(child, score)

        # Get best path
        best = root.best_child()
        best_plan = best.state if best else ""

        # Generate final answer from best plan
        final = await self._generate_final(best_plan, task, root)

        return {
            "answer": final,
            "best_state": best_plan,
            "root_visits": root.visits,
            "best_score": best.value / max(best.visits, 1) if best else 0,
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            # Select child with highest UCB1
            best = max(node.children, key=lambda c: c.ucb1())
            node = best
        return node

    async def _expand(self, node: MCTSNode, task: str) -> None:
        """Expand node by generating candidate plan variations."""
        prompt = f"""基于以下旅行计划，生成 3 个不同的优化变体，每个用1-2句话描述：

任务: {task}
当前计划: {node.state}

生成3个优化方向（编号列出）："""

        response = await chat([{"role": "user", "content": prompt}], temperature=0.9)
        candidates = self._parse_list(response, 3)

        for action, state in enumerate(candidates):
            child = MCTSNode(state=state, parent=node, action=f"opt_{action}")
            node.children.append(child)

    def _select_child_to_simulate(self, node: MCTSNode) -> MCTSNode:
        """If node has children, randomly pick one. Otherwise return node."""
        if node.children:
            return random.choice(node.children)
        return node

    async def _simulate(self, node: MCTSNode, task: str) -> float:
        """Simulate by evaluating the plan quality via LLM."""
        prompt = MCTS_EVALUATE_PROMPT.format(plan=f"任务: {task}\n计划: {node.state}")
        try:
            response = await chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=50)
            score = float(response.strip())
            return max(1, min(10, score))
        except (ValueError, TypeError):
            return 5.0

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    async def _generate_final(self, best_plan: str, task: str, root: MCTSNode) -> str:
        """Generate polished final answer from best plan."""
        best = root.best_child()
        score = best.value / max(best.visits, 1) if best else 0

        prompt = f"""基于 MCTS 搜索的结果，生成一份完整的旅行规划。

任务: {task}
最优方案（评分 {score:.1f}/10）: {best_plan}

请用中文生成完整旅行规划，包括：
1. 总体概述
2. 每日行程
3. 吃住行推荐
4. 注意事项"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.6)

    def _parse_list(self, text: str, n: int) -> list[str]:
        items = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and ". " in stripped:
                items.append(stripped.split(". ", 1)[1])
        if not items:
            items = [text]
        return items[:n]
