"""
Agent Core —— 编排推理策略、工具调用和旅行规划流程。

TravelAgent 是整个 Agent 系统的核心协调器，负责：
1. 根据用户输入自动选择合适的推理策略（ReAct / CoT / ToT / MCTS / Reflexion / Decompose）
2. 管理 MCP 工具调用（查询天气等外部服务）
3. 接入记忆存储（Memory Store），实现会话级上下文持久化
4. 将复杂的旅行规划任务路由到专门的规划器（Planner）模块
"""

# ──────────────────────────────────────────────
# 标准库导入
# ──────────────────────────────────────────────
from __future__ import annotations  # 启用类型注解的延迟求值，避免循环导入问题

import asyncio       # 异步 I/O 支持，用于并行执行多个规划任务
import json          # JSON 序列化 / 反序列化，处理天气等工具返回的数据
import logging       # 日志记录，方便排查运行时问题
import uuid          # 生成唯一会话 ID
from typing import Any  # 通用的类型注解


# ──────────────────────────────────────────────
# 项目内部模块导入
# ──────────────────────────────────────────────
from models.schemas import Strategy  # 枚举类型，定义所有可用的推理策略名称

from agent.llm import chat           # 通用 LLM 对话接口（无工具调用）
from agent.mcp.provider import mcp_provider       # MCP 工具提供器，集中管理所有外部工具
from agent.mcp.weather import WeatherTool         # 天气工具，负责格式化天气数据为提示词
from agent.memory import memory_store             # 记忆存储，持久化会话消息和用户偏好

# 导入所有推理引擎 —— 每种策略对应一个独立的引擎类
from agent.reasoning import (
    ReActEngine,       # ReAct：推理 + 工具调用交替进行
    CoTEngine,         # CoT：链式思维，逐步推理
    ToTEngine,         # ToT：思维树，探索多条推理路径
    MCTSEngine,        # MCTS：蒙特卡洛树搜索，适合优化类问题
    ReflexionEngine,   # Reflexion：自我反思改进
    TaskDecomposer,    # Decompose：任务分解，将大任务拆解为子任务逐一执行
)

# 导入规划器模块 —— 各司其职，分别处理旅行规划的特定环节
from agent.planner.destination import get_destination_intro  # 目的地介绍生成
from agent.planner.itinerary import generate_itinerary       # 行程安排生成
from agent.planner.food_stay import recommend_food, recommend_accommodation  # 美食与住宿推荐
from agent.planner.transport import plan_transport           # 交通方案规划
from agent.planner.adjuster import dynamic_adjuster          # 动态调整器，根据天气等因素优化行程

logger = logging.getLogger("travel_agent.core")  # 获取当前模块的日志记录器


class StrategySelector:
    """策略选择器 —— 根据任务描述自动选择最合适的推理策略。"""

    @staticmethod
    def select(task: str, user_strategy: Strategy | None = None) -> str:
        """
        分析用户输入的任务文本，返回对应的策略名称。

        参数：
            task: 用户的输入文本（中文）。
            user_strategy: 用户显式指定的策略（可选）。如果提供了，直接返回，不做自动分析。

        返回：
            策略名称字符串，例如 "react"、"cot"、"tot" 等。
        """
        # 如果用户明确指定了策略，直接使用，跳过自动选择逻辑
        if user_strategy:
            return user_strategy.value

        # 将任务文本转为小写，方便关键词匹配
        task_lower = task.lower()

        # 关键词匹配规则 1：优化 / 路线规划类问题 → 使用 ToT（思维树）
        # ToT 擅长探索多条路径并找到最优解，适合路线优化和方案对比
        if any(kw in task_lower for kw in ["最优", "路线", "最短", "最快", "优化", "方案对比"]):
            return "tot"

        # 关键词匹配规则 2：多步骤复杂规划 → 使用 Decompose（任务分解）
        # 当用户需要规划多日行程时，分解为子任务再分别执行效果更好
        if any(kw in task_lower for kw in [
            "规划", "行程", "攻略", "安排", "几日", "天游", "三日", "五日", "七日",
        ]):
            return "decompose"

        # 关键词匹配规则 3：自我改进 / 重新规划 → 使用 Reflexion（反思）
        # Reflexion 会对已有方案进行自我评估和优化改进
        if any(kw in task_lower for kw in ["改进", "优化已有", "重新规划", "调整行程"]):
            return "reflexion"

        # 关键词匹配规则 4：分析 / 比较类问题 → 使用 CoT（思维链）
        # CoT 通过逐步推理展示思考过程，适合解释性任务
        if any(kw in task_lower for kw in ["分析", "为什么", "如何选择", "比较", "推荐理由"]):
            return "cot"

        # 默认策略：ReAct（推理 + 行动）
        # ReAct 是通用策略，支持工具调用，适用于大多数对话场景
        return "react"


class TravelAgent:
    """主旅行 Agent —— 集成推理、记忆和全流程规划能力。"""

    def __init__(self):
        """初始化所有推理引擎和策略选择器。每个引擎在首次调用时才会真正加载模型配置。"""
        self.react = ReActEngine()         # ReAct 推理引擎
        self.cot = CoTEngine()             # 链式思维推理引擎
        self.tot = ToTEngine()             # 思维树推理引擎
        self.mcts = MCTSEngine()           # 蒙特卡洛树搜索引擎
        self.reflexion = ReflexionEngine() # 自我反思引擎
        self.decomposer = TaskDecomposer() # 任务分解引擎
        self.selector = StrategySelector() # 策略自动选择器

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        strategy: Strategy | None = None,
    ) -> dict:
        """
        通用对话入口 —— 接收用户消息，自动选择推理策略并执行。

        参数：
            message: 用户的输入文本。
            session_id: 会话 ID（可选）。不提供则自动生成。
            strategy: 用户指定的推理策略（可选）。不提供则自动选择。

        返回：
            包含 session_id、response（回答文本）、strategy_used（所用策略）的字典。
        """
        # 如果没有传入会话 ID，自动生成一个 12 字符的十六进制随机串
        session_id = session_id or uuid.uuid4().hex[:12]

        # 尝试在记忆存储中创建会话记录（如果会话已存在则忽略错误）
        try:
            await memory_store.create_session(session_id)
        except Exception as e:
            logger.error("Failed to create session %s: %s", session_id, e)

        # 从记忆存储中获取历史消息，并结合当前消息构建 LLM 上下文
        messages_context = await memory_store.build_context(session_id, message)

        # 将用户消息持久化保存到记忆存储中
        try:
            await memory_store.add_message(session_id, "user", message)
        except Exception as e:
            logger.error("Failed to save user message: %s", e)

        # 调用策略选择器，决定本次对话使用哪种推理策略
        selected = self.selector.select(message, strategy)
        logger.info("Session=%s strategy=%s message=%.100s...", session_id, selected, message)

        # 获取所有可用 MCP 工具的描述文本，供 ReAct 引擎使用
        tools_desc = mcp_provider.get_tools_description()

        # 定义工具调用闭包 —— 封装 MCP 提供器的 call_tool 方法，提取返回文本
        async def call_tool(name: str, arguments: dict) -> Any:
            """调用指定名称的 MCP 工具并返回文本结果。"""
            result = await mcp_provider.call_tool(name, arguments)
            if result.is_error:
                return result.content[0]["text"]  # 工具调用失败时返回错误信息文本
            return result.content[0]["text"]       # 工具调用成功时返回结果文本

        # 根据选定的策略，路由到对应的推理引擎执行
        try:
            if selected == "react":
                # ReAct 策略：推理 + 工具调用交替进行，需要传入工具描述和调用函数
                result = await self.react.reason(message, tools_desc, call_tool)
            elif selected == "cot":
                # CoT 策略：纯思维链推理，不需要工具
                result = await self.cot.reason(message)
            elif selected == "tot":
                # ToT 策略：思维树，探索多条推理分支
                result = await self.tot.reason(message)
            elif selected == "mcts":
                # MCTS 策略：蒙特卡洛树搜索，模拟多条路径
                result = await self.mcts.reason(message)
            elif selected == "reflexion":
                # Reflexion 策略：先生成初始方案，再进行自我反思和改进
                result = await self.reflexion.reason(message)
            elif selected == "decompose":
                # Decompose 策略：先分解任务，再对每个子任务执行 ReAct 推理
                async def execute_subtask(subtask, context):
                    """执行单个子任务 —— 每个子任务都会调用 ReAct 引擎。"""
                    return await self.react.reason(
                        subtask["description"], tools_desc, call_tool
                    )
                # 调用任务分解器：将大任务拆解为多个子任务，逐一执行
                decomposed = await self.decomposer.execute(message, execute_subtask)
                # 将所有子任务的结果综合起来，生成一份连贯的最终回答
                synthesis = await self._synthesize_results(message, decomposed)
                result = {"answer": synthesis, "decomposed": decomposed}
            else:
                # 兜底：遇到未知策略时回退到 ReAct
                result = await self.react.reason(message, tools_desc, call_tool)
        except Exception as e:
            # 捕获推理过程中的任何异常，返回友好的错误提示给用户
            logger.error("Reasoning failed for session=%s strategy=%s: %s", session_id, selected, e)
            answer = f"抱歉，处理您的请求时遇到错误: {e}。请稍后重试。"
            result = {"answer": answer}

        # 从推理结果中提取回答文本
        answer = result.get("answer", "")

        # 将助手的回答持久化保存到记忆存储中，同时附带元数据（所用策略、推理步骤数）
        try:
            await memory_store.add_message(
                session_id, "assistant", answer,
                metadata={"strategy": selected, "steps": len(result.get("steps", []))},
            )
        except Exception as e:
            logger.error("Failed to save assistant message: %s", e)

        # 返回标准化的响应格式：会话 ID、回答文本、所用策略
        return {
            "session_id": session_id,
            "response": answer,
            "strategy_used": selected,
        }

    async def plan_travel(self, request) -> dict:
        """
        完整的旅行规划工作流 —— 分为 8 个有序步骤，优雅降级。

        步骤概览：
          1. 查询目的地天气（非关键，失败不阻塞）
          2. 获取目的地介绍
          3. 生成详细行程安排
          4-6. 并行查询美食、住宿和交通方案
          7. 根据天气和人数动态调整行程
          8. 生成出行温馨提示

        每个步骤都有独立的 try/except 保护，确保单个步骤失败不影响整体流程。
        """
        # 在函数内部导入 TravelPlanRequest 模型，避免模块级别的循环依赖
        from models.schemas import TravelPlanRequest

        req: TravelPlanRequest = request       # 类型强转，获得类型提示支持
        session_id = uuid.uuid4().hex[:12]     # 为本次规划生成唯一的会话 ID

        # 创建会话记录（失败不影响后续流程）
        try:
            await memory_store.create_session(session_id)
        except Exception as e:
            logger.error("Failed to create plan session: %s", e)

        # 记录开始时间，用于统计每个步骤的耗时
        import time as _time
        t0 = _time.time()
        logger.info(
            "Planning trip to %s (%s~%s) %d travelers elderly=%d children=%d mode=%s",
            req.destination, req.start_date, req.end_date,
            req.num_travelers, req.num_elderly, req.num_children, req.travel_mode.value,
        )

        # ────────────────────────────────────────────
        # 第 1 步：查询目的地天气（非关键步骤，优雅降级）
        # ────────────────────────────────────────────
        weather_data = None
        try:
            weather_result = await mcp_provider.call_tool("weather", {
                "city": req.destination,
                "date": req.start_date,
            })
            if not weather_result.is_error:
                weather_data = json.loads(weather_result.content[0]["text"])  # 解析 JSON 格式的天气数据
        except Exception as e:
            logger.warning("Weather query failed for %s: %s", req.destination, e)

        logger.info("Weather done (%.1fs)", _time.time() - t0)
        # 将天气数据格式化为提示词所需的文本格式，如果获取失败则使用占位文本
        weather_summary = WeatherTool.format_for_prompt(weather_data) if weather_data else "天气数据暂不可用"

        # ────────────────────────────────────────────
        # 第 2 步：获取目的地介绍
        # ────────────────────────────────────────────
        t_dest = _time.time()
        try:
            dest_intro = await get_destination_intro(
                req.destination, req.interests,
                req.start_date[:7] if len(req.start_date) >= 7 else None,  # 提取月份信息（YYYY-MM）
            )
        except Exception as e:
            logger.error("Destination intro failed: %s", e)
            dest_intro = f"目的地介绍暂时无法生成 ({e})"  # 优雅降级：返回错误信息而非抛出异常
        logger.info("Destination intro done (%.1fs)", _time.time() - t_dest)

        # ────────────────────────────────────────────
        # 第 3 步：生成详细行程安排
        # ────────────────────────────────────────────
        t_itin = _time.time()
        try:
            itinerary = await generate_itinerary(
                req.destination, req.departure_from,
                req.start_date, req.end_date,
                req.travel_mode.value, req.num_travelers,
                req.num_elderly, req.num_children,
                req.budget_min, req.budget_max, req.interests, weather_data,
            )
        except Exception as e:
            logger.error("Itinerary generation failed: %s", e)
            itinerary = f"行程生成失败: {e}"
        logger.info("Itinerary done (%.1fs)", _time.time() - t_itin)

        # ────────────────────────────────────────────
        # 第 4 步 ~ 第 6 步：美食推荐、住宿建议、交通方案 —— 并行执行
        # 这三个任务彼此独立，没有数据依赖，使用 asyncio.gather 同时启动以节省时间
        # ────────────────────────────────────────────
        t_parallel = _time.time()
        food, accommodation, transport = await asyncio.gather(
            self._safe_food(req),           # 第 4 步：美食推荐（带异常保护）
            self._safe_accommodation(req),   # 第 5 步：住宿推荐（带异常保护）
            self._safe_transport(req),       # 第 6 步：交通规划（带异常保护）
        )
        logger.info("Food/Accommodation/Transport done (%.1fs)", _time.time() - t_parallel)

        # ────────────────────────────────────────────
        # 第 7 步：动态调整 —— 根据天气、人数、预算等信息对行程进行优化调整
        # ────────────────────────────────────────────
        try:
            adjustments = await dynamic_adjuster.adjust(
                itinerary, weather_data, req.num_travelers,
                req.num_elderly, req.num_children,
                req.travel_mode.value, req.budget_min, req.budget_max,
            )
        except Exception as e:
            logger.error("Dynamic adjustment failed: %s", e)
            adjustments = "动态调整暂时不可用"

        # ────────────────────────────────────────────
        # 第 8 步：生成出行温馨提示
        # 根据天气状况、人员构成、出行方式等因素，生成个性化的建议列表
        # ────────────────────────────────────────────
        tips = self._generate_tips(req, weather_data)

        # ────────────────────────────────────────────
        # 组装完整的 Markdown 响应文本
        # 将所有步骤的结果合并为一份结构清晰的旅行规划文档
        # ────────────────────────────────────────────
        full_response = f"""# {req.destination} 旅行规划

{weather_summary}

---

## 目的地介绍
{dest_intro}

---

## 行程安排
{itinerary}

---

## 动态调整
{adjustments}

---

## 美食推荐
{food}

---

## 住宿建议
{accommodation}

---

## 交通方案
{transport}

---

## 温馨提示
{chr(10).join(f'- {t}' for t in tips)}
"""

        # 将本次规划请求和结果保存到记忆存储中，方便后续会话回顾
        await memory_store.add_message(session_id, "user",
            f"规划 {req.destination} {req.start_date}-{req.end_date} "
            f"{req.num_travelers}人 老人:{req.num_elderly} 小孩:{req.num_children}")
        await memory_store.add_message(session_id, "assistant", full_response,
            metadata={"type": "travel_plan"})  # 标记消息类型为旅行规划

        logger.info("Plan complete for %s (%.1fs total)", req.destination, _time.time() - t0)

        # 返回结构化结果，包含所有规划环节的独立字段和完整的 Markdown 文本
        return {
            "session_id": session_id,
            "destination": req.destination,
            "weather": weather_data or {},
            "itinerary": itinerary,
            "food_recommendations": food,
            "accommodation_recommendations": accommodation,
            "transport_plan": transport,
            "destination_intro": dest_intro,
            "adjustments": adjustments,
            "tips": tips,
            "full_response": full_response,
        }

    async def _synthesize_results(self, task: str, decomposed: dict) -> str:
        """
        综合多个子任务的结果，生成一份连贯的最终回答。

        参数：
            task: 原始的完整任务描述。
            decomposed: 任务分解器的执行结果，包含所有子任务及其结果。

        返回：
            综合后的完整回答文本（字符串）。
        """
        # 遍历所有子任务，将每个子任务的描述和结果拼接到一起
        results_text = ""
        for subtask in decomposed.get("subtasks", []):
            sid = subtask["id"]                                      # 子任务 ID
            result = decomposed.get("results", {}).get(sid, {})      # 子任务的执行结果
            results_text += f"\n## {subtask.get('description', sid)}\n"  # 子任务标题
            # 子任务结果内容（兼容字典和字符串两种格式）
            results_text += result.get("answer", str(result)) if isinstance(result, dict) else str(result)

        # 构建综合提示词，要求 LLM 将所有子任务结果合并为一份连贯的规划回答
        prompt = f"""请将以下子任务结果综合为一份完整的旅行规划回答。

原始任务: {task}

子任务结果:
{results_text}

请用中文生成一份连贯、完整的旅行规划。"""
        # 调用通用 LLM 接口（无工具调用），以较低温度（0.6）获取更有创造性的综合结果
        return await chat([{"role": "user", "content": prompt}], temperature=0.6)

    def _generate_tips(self, req, weather_data: dict | None) -> list[str]:
        """
        根据天气状况、出行人员构成和交通方式，生成个性化的温馨提示列表。

        参数：
            req: 旅行规划请求对象，包含人数、老人小孩数量、出行方式等字段。
            weather_data: 天气数据（可能为 None，表示天气查询失败）。

        返回：
            温馨提示字符串列表。
        """
        tips = []  # 初始化空列表，逐步添加提示项

        # ───────────── 基于天气的提示 ─────────────
        if weather_data and "forecast" in weather_data:
            for day in weather_data.get("forecast", []):
                precip = day.get("precip_prob", 0)    # 降水概率
                if precip > 50:
                    tips.append(f"{day['date']} 降水概率{precip}%, 请携带雨具")
                temp = day.get("temp_max", 25)         # 最高温度
                if temp > 35:
                    tips.append(f"{day['date']} 高温{temp}°C, 注意防暑防晒")
                elif temp < 10:
                    tips.append(f"{day['date']} 低温{temp}°C, 注意保暖")

        # ───────────── 基于人员构成的提示 ─────────────
        if req.num_elderly > 0:
            tips.append(f"有{req.num_elderly}位老人: 携带常用药品, 选择平坦路线, 随身带折叠椅或坐垫")
        if req.num_children > 0:
            tips.append(f"有{req.num_children}个小孩: 准备零食和水, 安排趣味互动环节, 随身带备用衣物")
        if req.num_travelers >= 5:
            tips.append("建立微信群方便联络, 提前预订门票和餐厅, 设置每日集合时间和地点")

        # ───────────── 基于出行方式的提示 ─────────────
        if req.travel_mode.value == "car":
            tips.append("出发前检查车况, 下载离线地图, 确认沿途充电/加油站点")
        elif req.travel_mode.value == "train":
            tips.append("提前30分钟到站, 确认接驳交通, 大件行李可提前托运")
        elif req.travel_mode.value == "plane":
            tips.append("提前2小时到机场, 关注航班动态, 准备机上娱乐")

        # ───────────── 通用出行提示 ─────────────
        tips.extend([
            "下载离线地图和翻译软件备用",
            "保存紧急联系电话（酒店、医院、报警）",
            "购买旅行保险",
        ])

        return tips


    async def _safe_food(self, req) -> str:
        """
        带异常保护的美食推荐调用。
        如果推荐服务发生异常，返回占位文本而非抛出异常。
        """
        try:
            return await recommend_food(
                req.destination, req.num_travelers,
                req.num_elderly, req.num_children,
                req.budget_min, req.budget_max,
            )
        except Exception as e:
            logger.error("Food recommendation failed: %s", e)
            return "美食推荐暂时不可用"

    async def _safe_accommodation(self, req) -> str:
        """
        带异常保护的住宿推荐调用。
        如果推荐服务发生异常，返回占位文本而非抛出异常。
        """
        try:
            return await recommend_accommodation(
                req.destination, req.num_travelers,
                req.num_elderly, req.num_children,
                req.budget_min, req.budget_max,
            )
        except Exception as e:
            logger.error("Accommodation recommendation failed: %s", e)
            return "住宿推荐暂时不可用"

    async def _safe_transport(self, req) -> str:
        """
        带异常保护的交通规划调用。
        如果规划服务发生异常，返回占位文本而非抛出异常。
        """
        try:
            return await plan_transport(
                req.destination, req.departure_from,
                req.start_date, req.end_date,
                req.travel_mode.value, req.num_travelers,
                req.num_elderly, req.num_children,
                req.budget_min, req.budget_max,
            )
        except Exception as e:
            logger.error("Transport planning failed: %s", e)
            return "交通规划暂时不可用"


# 全局单例 —— 整个应用共享同一个 TravelAgent 实例
travel_agent = TravelAgent()
