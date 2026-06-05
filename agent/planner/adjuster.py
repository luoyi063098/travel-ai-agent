# agent/planner/adjuster.py
# 动态调整模块 —— 根据天气、人群特征、出行方式、预算等实时约束条件，
# 对已有行程进行动态分析和调整建议

from __future__ import annotations  # 支持在类型注解中使用类名自身（Python 3.7+ 兼容）

from agent.llm import chat  # 导入 LLM 聊天接口

# 动态调整提示词模板
# 接收原始行程和约束条件，让 LLM 分析匹配度并给出调整建议
ADJUSTER_PROMPT = """你是一个旅行规划动态调整专家。请根据以下约束条件对行程进行调整。

## 原始行程
{plan}

## 约束条件
{constraints}

## 调整要求
请分析原行程与约束条件的匹配度，给出调整后的建议。关注：
1. 天气导致的行程调整（雨天改室内、高温调整时段）
2. 老人/小孩特殊需求（节奏、休息、安全）
3. 多人出行协作（分流、预订、集合点）
4. 出行方式限制（停车、接驳、时间）

请用中文给出具体的调整建议。"""


class DynamicAdjuster:
    """动态行程调整器：根据实时约束条件（天气、人群、预算等）生成行程优化建议。"""

    def __init__(self):
        pass  # 目前无初始化状态，保持简单

    def build_constraints_from_request(
        self,
        weather_data: dict | None,
        num_travelers: int,
        num_elderly: int,
        num_children: int,
        travel_mode: str,
        budget_min: int,
        budget_max: int,
    ) -> list[str]:
        """根据请求参数生成约束条件列表，每条约束是一个字符串。"""
        constraints = []

        # ================================================================
        # 1. 天气约束（逐日分析）
        # ================================================================
        if weather_data and "forecast" in weather_data:
            for day in weather_data.get("forecast", []):
                # 提取该天的天气指标
                precip = day.get("precip_prob", 0)    # 降水概率（百分比）
                w = day.get("weather", "")             # 天气描述（晴/雨/雪等）
                temp_max = day.get("temp_max", 25)     # 最高温度

                # ---- 降水约束 ----
                # 降水概率 > 60% 或明确下雨：标记为红色预警，建议改室内
                if precip > 60 or w.endswith("雨"):
                    constraints.append(f"🔴 {day['date']}: 高降水概率({precip}%), 户外活动需改为室内备选")
                # 降水概率 30%~60%：标记为黄色预警，建议携带雨具
                elif precip > 30:
                    constraints.append(f"🟡 {day['date']}: 中等降水概率({precip}%), 建议携带雨具并准备室内备选")

                # ---- 高温/低温约束 ----
                # 最高温 > 35°C：高温预警，避免中午户外活动
                if temp_max > 35:
                    constraints.append(f"🔴 {day['date']}: 高温{temp_max}°C, 避免中午户外活动, 安排室内/水上项目")
                # 最高温 < 5°C：低温预警，注意保暖
                elif temp_max < 5:
                    constraints.append(f"🔴 {day['date']}: 低温{temp_max}°C, 注意保暖, 减少户外停留时间")

        # ================================================================
        # 2. 老人约束
        # ================================================================
        if num_elderly > 0:
            constraints.append(f"👴 {num_elderly}位老人同行: 节奏放缓, 每2小时休息, 避免爬山长步行, 优先有电梯/无障碍设施")

        # ================================================================
        # 3. 儿童约束
        # ================================================================
        if num_children > 0:
            constraints.append(f"👶 {num_children}个儿童同行: 安排亲子景点, 避免危险活动, 预留午休时间")

        # ================================================================
        # 4. 团队规模约束
        # ================================================================
        if num_travelers >= 5:
            constraints.append(f"👥 多人出行({num_travelers}人): 提前预订餐厅/门票, 考虑包车, 设置集合点")

        # ================================================================
        # 5. 出行方式约束
        # ================================================================
        if travel_mode == "car":
            constraints.append("🚗 自驾: 每天驾驶≤3小时, 确认停车位, 避免拥堵路段")
        elif travel_mode == "train":
            constraints.append("🚄 高铁: 确认车站接驳, 首日预留到达缓冲时间")

        # ================================================================
        # 6. 预算约束（根据人均预算分档生成不同档位的建议）
        # ================================================================
        avg_budget = (budget_min + budget_max) // 2  # 计算预算中位数
        if avg_budget <= 1500:
            # 经济型预算（≤ 1500 元/人）：推荐公共交通、经济住宿、性价比餐厅
            constraints.append(f"💰 经济预算({budget_min}~{budget_max}元/人): 优选公共交通, 经济型住宿, 性价比餐厅")
        elif avg_budget >= 6000:
            # 高端预算（≥ 6000 元/人）：推荐精品酒店、包车、特色餐厅
            constraints.append(f"💎 高端预算({budget_min}~{budget_max}元/人): 可选精品酒店, 包车服务, 特色餐厅")
        else:
            # 中等预算（1500~6000 元/人）：舒适型住宿、地铁+打车混合、当地特色餐厅
            constraints.append(f"💵 中等预算({budget_min}~{budget_max}元/人): 舒适型住宿, 地铁+打车, 当地特色餐厅")

        return constraints

    async def adjust(
        self,
        plan: str,
        weather_data: dict | None,
        num_travelers: int,
        num_elderly: int,
        num_children: int,
        travel_mode: str,
        budget_min: int,
        budget_max: int,
    ) -> str:
        """对已有行程进行动态调整。"""
        # ---- 第 1 步：构建约束条件列表 ----
        constraints = self.build_constraints_from_request(
            weather_data, num_travelers, num_elderly, num_children,
            travel_mode, budget_min, budget_max,
        )

        # ---- 第 2 步：如果没有约束条件，直接返回无需调整 ----
        if not constraints:
            return "无需调整，当前行程已满足所有条件。"

        # ---- 第 3 步：将约束列表格式化为 Markdown 列表 ----
        constraint_text = "\n".join(f"- {c}" for c in constraints)

        # ---- 第 4 步：注入提示词模板并调用 LLM ----
        prompt = ADJUSTER_PROMPT.format(plan=plan, constraints=constraint_text)

        # temperature 设为 0.5，较低的温度使调整建议更加保守和可预测
        return await chat([{"role": "user", "content": prompt}], temperature=0.5)


# 模块级单例，方便全局使用
dynamic_adjuster = DynamicAdjuster()
