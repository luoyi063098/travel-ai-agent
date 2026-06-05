"""
Monte Carlo Tree Search (MCTS) Engine.

Uses MCTS to explore and optimize travel plans through iterative tree search.
Good for optimization problems like route planning, itinerary optimization.

MCTS steps: Selection → Expansion → Simulation → Backpropagation
"""

from __future__ import annotations

import math  # 提供数学函数，用于 UCB1 公式中的对数计算
import random  # 提供随机选择功能，用于 Simulate 阶段随机选取子节点

from agent.llm import chat
from config import MCTS_ITERATIONS

MCTS_EVALUATE_PROMPT = """评估以下旅行计划的质量（1-10分），只返回数字。

计划: {plan}

考虑因素：合理性、天气适配、个性化需求、全面性。

Score:"""


class MCTSNode:
    """
    MCTS 搜索树中的节点。
    每个节点代表一个旅行计划状态，包含访问次数、累积价值和子节点列表。
    使用 __slots__ 限制属性以节省内存（在大量节点场景下效果显著）。
    """
    __slots__ = ("state", "parent", "children", "visits", "value", "action")

    def __init__(self, state: str, parent=None, action: str = ""):
        # state: 当前节点代表的旅行计划描述文本
        self.state = state
        # parent: 父节点引用，用于反向传播时向上遍历
        self.parent = parent
        # children: 子节点列表，由 Expansion 阶段生成
        self.children: list["MCTSNode"] = []
        # visits: 该节点被访问的总次数（每次反向传播 +1）
        self.visits = 0
        # value: 该节点的累积评分总和（用于计算平均分）
        self.value = 0.0
        # action: 从父节点到此节点所执行的动作描述（如 "opt_0"）
        self.action = action

    def ucb1(self, exploration: float = 1.414) -> float:
        """
        计算 UCB1（Upper Confidence Bound 1）值，用于在 Selection 阶段选择最 promising 的子节点。

        UCB1 公式：
            UCB1 = exploitation + exploration * sqrt(ln(parent.visits) / visits)

        其中：
        - exploitation = value / visits：该节点的平均得分，利用已有知识（exploitation）
        - exploration_term = exploration * sqrt(ln(parent.visits) / self.visits)：
          鼓励探索访问次数较少的节点（exploration）
        - exploration 参数（默认 1.414 = sqrt(2)）控制探索与利用的权衡

        如果节点尚未被访问（visits == 0），返回无穷大以确保它被优先选择。
        """
        if self.visits == 0:
            # 未访问的节点具有最高的优先选择权，保证每个节点至少被探索一次
            return float("inf")
        # 利用项：该节点的平均得分
        exploitation = self.value / self.visits
        # 探索项：鼓励访问次数较少的节点，sqrt(ln(父节点访问数) / 自身访问数)
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        # UCB1 = 利用 + 探索
        return exploitation + exploration_term

    def best_child(self) -> "MCTSNode | None":
        """
        返回所有子节点中平均得分最高的节点（最终选择最佳方案时使用）。
        与 UCB1 不同，这里只考虑 exploitation（value / visits），不考虑探索项。
        """
        if not self.children:
            return None
        # 取 value/visits 最大的子节点，用 max(visits, 1) 避免除以零
        return max(self.children, key=lambda c: c.value / max(c.visits, 1))


class MCTSEngine:
    """Monte Carlo Tree Search for travel plan optimization."""

    def __init__(self, iterations: int = MCTS_ITERATIONS):
        # 保存 MCTS 主循环的迭代次数（每次迭代=一次 Selection→Expansion→Simulation→Backpropagation）
        self.iterations = iterations

    async def reason(self, task: str, system_extra: str = "") -> dict:
        """Execute MCTS to find best travel plan."""
        # 初始化根节点，初始状态为占位文本 "初始行程方案"
        root = MCTSNode(state="初始行程方案")

        # 运行 MCTS 主循环，迭代指定的次数
        for i in range(self.iterations):
            # Selection 阶段：从根节点出发，依据 UCB1 策略向下选择直到叶节点
            node = self._select(root)
            # Expansion 阶段：如果已访问过且还没有子节点，则展开该节点
            if node.visits > 0 and not node.children:
                await self._expand(node, task)
            # 从当前节点（或其子节点）中随机挑选一个用于 Simulation
            child = self._select_child_to_simulate(node)
            # Simulation 阶段：通过 LLM 评估该节点代表的计划质量，获得评分
            score = await self._simulate(child, task)
            # Backpropagation 阶段：将评分沿路径反向传播到根节点
            self._backpropagate(child, score)

        # MCTS 搜索结束后，从根节点取评分最高的子节点作为最优方案
        best = root.best_child()
        best_plan = best.state if best else ""

        # 基于最优方案生成最终的、经过润色的完整回答
        final = await self._generate_final(best_plan, task, root)

        # 返回最终回答、最佳状态、根节点访问次数和最佳评分
        return {
            "answer": final,
            "best_state": best_plan,
            "root_visits": root.visits,
            "best_score": best.value / max(best.visits, 1) if best else 0,
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection 阶段：
        从当前节点开始，沿着 UCB1 值最高的子节点路径向下遍历，直到到达一个叶节点。
        叶节点定义为没有子节点的节点。
        """
        while node.children:
            # 在子节点中选择 UCB1 值最高的那个继续向下
            best = max(node.children, key=lambda c: c.ucb1())
            node = best
        return node

    async def _expand(self, node: MCTSNode, task: str) -> None:
        """
        Expansion 阶段：
        对当前叶节点，调用 LLM 生成不同的优化变体，作为子节点添加到树中。
        使用高温度（0.9）以鼓励生成多样化的候选方案。
        """
        prompt = f"""基于以下旅行计划，生成 3 个不同的优化变体，每个用1-2句话描述：

任务: {task}
当前计划: {node.state}

生成3个优化方向（编号列出）："""

        response = await chat([{"role": "user", "content": prompt}], temperature=0.9)
        # 解析 LLM 返回的编号列表，提取最多 3 个候选方案
        candidates = self._parse_list(response, 3)

        # 将每个候选方案创建为 MCTSNode，并挂到当前节点的 children 列表中
        for action, state in enumerate(candidates):
            child = MCTSNode(state=state, parent=node, action=f"opt_{action}")
            node.children.append(child)

    def _select_child_to_simulate(self, node: MCTSNode) -> MCTSNode:
        """
        Simulation 阶段前的子节点选择：
        如果当前节点有子节点，则随机选择一个进行模拟评估；
        如果没有子节点，则直接对当前节点本身进行评估。
        """
        if node.children:
            return random.choice(node.children)
        return node

    async def _simulate(self, node: MCTSNode, task: str) -> float:
        """
        Simulation 阶段：
        使用 LLM 评估指定节点的旅行计划质量，返回 1-10 分的评分。
        设置 max_tokens=50 以控制评估输出长度（只需要一个数字）。
        如果解析失败，返回默认分 5.0。
        """
        prompt = MCTS_EVALUATE_PROMPT.format(plan=f"任务: {task}\n计划: {node.state}")
        try:
            response = await chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=50)
            # 尝试将 LLM 返回的文本直接转换为浮点数
            score = float(response.strip())
            # 确保评分在 1-10 的有效范围内
            return max(1, min(10, score))
        except (ValueError, TypeError):
            # 如果 LLM 返回的文本不能转为数字，返回默认分 5.0
            return 5.0

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """
        Backpropagation 阶段：
        从当前节点开始，沿 parent 链向上遍历直到根节点。
        对路径上的每个节点：访问次数 +1，累积评分 += score。
        """
        while node:
            node.visits += 1  # 增加访问计数
            node.value += score  # 累加评分
            node = node.parent  # 上移到父节点

    async def _generate_final(self, best_plan: str, task: str, root: MCTSNode) -> str:
        """
        基于 MCTS 搜索得到的最优方案，生成一份经过润色的完整旅行规划。
        从根节点获取最佳子节点及其平均评分，作为生成提示的一部分。
        """
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
        """
        解析 LLM 返回的编号列表文本，提取列表项内容。
        支持 "1. xxx" 格式的编号。
        如果解析失败，返回整段文本。
        """
        items = []
        for line in text.split("\n"):
            stripped = line.strip()
            # 匹配以数字开头且包含 ". " 的行
            if stripped and stripped[0].isdigit() and ". " in stripped:
                items.append(stripped.split(". ", 1)[1])
        if not items:
            items = [text]
        return items[:n]
