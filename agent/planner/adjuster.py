"""Dynamic adjuster - adjusts travel plans based on constraints."""

from __future__ import annotations

from agent.llm import chat

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
    """Adjusts travel plans based on real-world constraints."""

    def __init__(self):
        pass

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
        constraints = []

        # Weather constraints
        if weather_data and "forecast" in weather_data:
            for day in weather_data.get("forecast", []):
                precip = day.get("precip_prob", 0)
                w = day.get("weather", "")
                temp_max = day.get("temp_max", 25)

                if precip > 60 or w.endswith("雨"):
                    constraints.append(f"🔴 {day['date']}: 高降水概率({precip}%), 户外活动需改为室内备选")
                elif precip > 30:
                    constraints.append(f"🟡 {day['date']}: 中等降水概率({precip}%), 建议携带雨具并准备室内备选")

                if temp_max > 35:
                    constraints.append(f"🔴 {day['date']}: 高温{temp_max}°C, 避免中午户外活动, 安排室内/水上项目")
                elif temp_max < 5:
                    constraints.append(f"🔴 {day['date']}: 低温{temp_max}°C, 注意保暖, 减少户外停留时间")

        # Elderly constraints
        if num_elderly > 0:
            constraints.append(f"👴 {num_elderly}位老人同行: 节奏放缓, 每2小时休息, 避免爬山长步行, 优先有电梯/无障碍设施")

        # Children constraints
        if num_children > 0:
            constraints.append(f"👶 {num_children}个儿童同行: 安排亲子景点, 避免危险活动, 预留午休时间")

        # Group size
        if num_travelers >= 5:
            constraints.append(f"👥 多人出行({num_travelers}人): 提前预订餐厅/门票, 考虑包车, 设置集合点")

        # Travel mode
        if travel_mode == "car":
            constraints.append("🚗 自驾: 每天驾驶≤3小时, 确认停车位, 避免拥堵路段")
        elif travel_mode == "train":
            constraints.append("🚄 高铁: 确认车站接驳, 首日预留到达缓冲时间")

        # Budget
        avg_budget = (budget_min + budget_max) // 2
        if avg_budget <= 1500:
            constraints.append(f"💰 经济预算({budget_min}~{budget_max}元/人): 优选公共交通, 经济型住宿, 性价比餐厅")
        elif avg_budget >= 6000:
            constraints.append(f"💎 高端预算({budget_min}~{budget_max}元/人): 可选精品酒店, 包车服务, 特色餐厅")
        else:
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
        constraints = self.build_constraints_from_request(
            weather_data, num_travelers, num_elderly, num_children,
            travel_mode, budget_min, budget_max,
        )

        if not constraints:
            return "无需调整，当前行程已满足所有条件。"

        constraint_text = "\n".join(f"- {c}" for c in constraints)
        prompt = ADJUSTER_PROMPT.format(plan=plan, constraints=constraint_text)

        return await chat([{"role": "user", "content": prompt}], temperature=0.5)


dynamic_adjuster = DynamicAdjuster()
