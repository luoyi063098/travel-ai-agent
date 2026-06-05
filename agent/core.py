"""
Agent Core - orchestrates reasoning strategies, tool calls, and planning.

The TravelAgent is the central coordinator that:
1. Selects the appropriate reasoning strategy for each task
2. Manages MCP tool calls (weather, etc.)
3. Integrates memory for persistent context
4. Routes tasks to specialized planners
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from models.schemas import Strategy
from agent.llm import chat
from agent.mcp.provider import mcp_provider
from agent.mcp.weather import WeatherTool
from agent.memory import memory_store
from agent.reasoning import (
    ReActEngine,
    CoTEngine,
    ToTEngine,
    MCTSEngine,
    ReflexionEngine,
    TaskDecomposer,
)
from agent.planner.destination import get_destination_intro
from agent.planner.itinerary import generate_itinerary
from agent.planner.food_stay import recommend_food, recommend_accommodation
from agent.planner.transport import plan_transport
from agent.planner.adjuster import dynamic_adjuster

logger = logging.getLogger("travel_agent.core")


class StrategySelector:
    """Selects the best reasoning strategy based on task analysis."""

    @staticmethod
    def select(task: str, user_strategy: Strategy | None = None) -> str:
        if user_strategy:
            return user_strategy.value

        task_lower = task.lower()

        # Optimization / route planning → MCTS
        if any(kw in task_lower for kw in ["最优", "路线", "最短", "最快", "优化", "方案对比"]):
            return "tot"

        # Multi-step complex planning → decompose + react
        if any(kw in task_lower for kw in [
            "规划", "行程", "攻略", "安排", "几日", "天游", "三日", "五日", "七日",
        ]):
            return "decompose"

        # Self-improvement / review
        if any(kw in task_lower for kw in ["改进", "优化已有", "重新规划", "调整行程"]):
            return "reflexion"

        # Step-by-step analysis
        if any(kw in task_lower for kw in ["分析", "为什么", "如何选择", "比较", "推荐理由"]):
            return "cot"

        # Default: ReAct (supports tool use)
        return "react"


class TravelAgent:
    """Main AI Travel Agent with reasoning, memory, and planning capabilities."""

    def __init__(self):
        self.react = ReActEngine()
        self.cot = CoTEngine()
        self.tot = ToTEngine()
        self.mcts = MCTSEngine()
        self.reflexion = ReflexionEngine()
        self.decomposer = TaskDecomposer()
        self.selector = StrategySelector()

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        strategy: Strategy | None = None,
    ) -> dict:
        """General chat endpoint with automatic strategy selection."""
        session_id = session_id or uuid.uuid4().hex[:12]

        try:
            await memory_store.create_session(session_id)
        except Exception as e:
            logger.error("Failed to create session %s: %s", session_id, e)

        # Build message context with memory
        messages_context = await memory_store.build_context(session_id, message)

        # Save user message
        try:
            await memory_store.add_message(session_id, "user", message)
        except Exception as e:
            logger.error("Failed to save user message: %s", e)

        # Select strategy
        selected = self.selector.select(message, strategy)
        logger.info("Session=%s strategy=%s message=%.100s...", session_id, selected, message)

        # Execute reasoning
        tools_desc = mcp_provider.get_tools_description()

        async def call_tool(name: str, arguments: dict) -> Any:
            result = await mcp_provider.call_tool(name, arguments)
            if result.is_error:
                return result.content[0]["text"]
            return result.content[0]["text"]

        try:
            if selected == "react":
                result = await self.react.reason(message, tools_desc, call_tool)
            elif selected == "cot":
                result = await self.cot.reason(message)
            elif selected == "tot":
                result = await self.tot.reason(message)
            elif selected == "mcts":
                result = await self.mcts.reason(message)
            elif selected == "reflexion":
                result = await self.reflexion.reason(message)
            elif selected == "decompose":
                async def execute_subtask(subtask, context):
                    return await self.react.reason(
                        subtask["description"], tools_desc, call_tool
                    )
                decomposed = await self.decomposer.execute(message, execute_subtask)
                synthesis = await self._synthesize_results(message, decomposed)
                result = {"answer": synthesis, "decomposed": decomposed}
            else:
                result = await self.react.reason(message, tools_desc, call_tool)
        except Exception as e:
            logger.error("Reasoning failed for session=%s strategy=%s: %s", session_id, selected, e)
            answer = f"抱歉，处理您的请求时遇到错误: {e}。请稍后重试。"
            result = {"answer": answer}

        answer = result.get("answer", "")

        # Save assistant message
        try:
            await memory_store.add_message(
                session_id, "assistant", answer,
                metadata={"strategy": selected, "steps": len(result.get("steps", []))},
            )
        except Exception as e:
            logger.error("Failed to save assistant message: %s", e)

        return {
            "session_id": session_id,
            "response": answer,
            "strategy_used": selected,
        }

    async def plan_travel(self, request) -> dict:
        """Full travel planning workflow with graceful degradation."""
        from models.schemas import TravelPlanRequest

        req: TravelPlanRequest = request
        session_id = uuid.uuid4().hex[:12]

        try:
            await memory_store.create_session(session_id)
        except Exception as e:
            logger.error("Failed to create plan session: %s", e)

        import time as _time
        t0 = _time.time()
        logger.info(
            "Planning trip to %s (%s~%s) %d travelers elderly=%d children=%d mode=%s",
            req.destination, req.start_date, req.end_date,
            req.num_travelers, req.num_elderly, req.num_children, req.travel_mode.value,
        )

        # Step 1: Get weather (non-critical - graceful degradation)
        weather_data = None
        try:
            weather_result = await mcp_provider.call_tool("weather", {
                "city": req.destination,
                "date": req.start_date,
            })
            if not weather_result.is_error:
                weather_data = json.loads(weather_result.content[0]["text"])
        except Exception as e:
            logger.warning("Weather query failed for %s: %s", req.destination, e)

        logger.info("Weather done (%.1fs)", _time.time() - t0)
        weather_summary = WeatherTool.format_for_prompt(weather_data) if weather_data else "天气数据暂不可用"

        # Step 2: Destination introduction
        t_dest = _time.time()
        try:
            dest_intro = await get_destination_intro(
                req.destination, req.interests,
                req.start_date[:7] if len(req.start_date) >= 7 else None,
            )
        except Exception as e:
            logger.error("Destination intro failed: %s", e)
            dest_intro = f"目的地介绍暂时无法生成 ({e})"
        logger.info("Destination intro done (%.1fs)", _time.time() - t_dest)

        # Step 3: Generate itinerary
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

        # Steps 4-6: Food, Accommodation, Transport — run in parallel
        t_parallel = _time.time()
        food, accommodation, transport = await asyncio.gather(
            self._safe_food(req),
            self._safe_accommodation(req),
            self._safe_transport(req),
        )
        logger.info("Food/Accommodation/Transport done (%.1fs)", _time.time() - t_parallel)

        # Step 7: Dynamic adjustment
        try:
            adjustments = await dynamic_adjuster.adjust(
                itinerary, weather_data, req.num_travelers,
                req.num_elderly, req.num_children,
                req.travel_mode.value, req.budget_min, req.budget_max,
            )
        except Exception as e:
            logger.error("Dynamic adjustment failed: %s", e)
            adjustments = "动态调整暂时不可用"

        # Step 8: Generate tips
        tips = self._generate_tips(req, weather_data)

        # Build full response
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

        # Save to memory
        await memory_store.add_message(session_id, "user",
            f"规划 {req.destination} {req.start_date}-{req.end_date} "
            f"{req.num_travelers}人 老人:{req.num_elderly} 小孩:{req.num_children}")
        await memory_store.add_message(session_id, "assistant", full_response,
            metadata={"type": "travel_plan"})

        logger.info("Plan complete for %s (%.1fs total)", req.destination, _time.time() - t0)

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
        """Synthesize decomposed subtask results into a coherent answer."""
        results_text = ""
        for subtask in decomposed.get("subtasks", []):
            sid = subtask["id"]
            result = decomposed.get("results", {}).get(sid, {})
            results_text += f"\n## {subtask.get('description', sid)}\n"
            results_text += result.get("answer", str(result)) if isinstance(result, dict) else str(result)

        prompt = f"""请将以下子任务结果综合为一份完整的旅行规划回答。

原始任务: {task}

子任务结果:
{results_text}

请用中文生成一份连贯、完整的旅行规划。"""
        return await chat([{"role": "user", "content": prompt}], temperature=0.6)

    def _generate_tips(self, req, weather_data: dict | None) -> list[str]:
        tips = []

        # Weather tips
        if weather_data and "forecast" in weather_data:
            for day in weather_data.get("forecast", []):
                precip = day.get("precip_prob", 0)
                if precip > 50:
                    tips.append(f"{day['date']} 降水概率{precip}%, 请携带雨具")
                temp = day.get("temp_max", 25)
                if temp > 35:
                    tips.append(f"{day['date']} 高温{temp}°C, 注意防暑防晒")
                elif temp < 10:
                    tips.append(f"{day['date']} 低温{temp}°C, 注意保暖")

        # Traveler tips
        if req.num_elderly > 0:
            tips.append(f"有{req.num_elderly}位老人: 携带常用药品, 选择平坦路线, 随身带折叠椅或坐垫")
        if req.num_children > 0:
            tips.append(f"有{req.num_children}个小孩: 准备零食和水, 安排趣味互动环节, 随身带备用衣物")
        if req.num_travelers >= 5:
            tips.append("建立微信群方便联络, 提前预订门票和餐厅, 设置每日集合时间和地点")

        # Travel mode tips
        if req.travel_mode.value == "car":
            tips.append("出发前检查车况, 下载离线地图, 确认沿途充电/加油站点")
        elif req.travel_mode.value == "train":
            tips.append("提前30分钟到站, 确认接驳交通, 大件行李可提前托运")
        elif req.travel_mode.value == "plane":
            tips.append("提前2小时到机场, 关注航班动态, 准备机上娱乐")

        # General tips
        tips.extend([
            "下载离线地图和翻译软件备用",
            "保存紧急联系电话（酒店、医院、报警）",
            "购买旅行保险",
        ])

        return tips


    async def _safe_food(self, req) -> str:
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


travel_agent = TravelAgent()
