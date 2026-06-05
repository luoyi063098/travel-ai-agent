"""Itinerary generation with dynamic adjustment."""

from agent.llm import chat

ITINERARY_PROMPT = """你是一个专业旅行行程规划师，擅长设计合理、舒适、个性化的旅行路线。

请根据以下信息规划详细行程，确保每个建议都切实可行。

## 行程参数
- 目的地: {destination}
- 出发地: {departure_from}
- 日期: {start_date} 至 {end_date} ({days}天)
- 出行方式: {travel_mode}
- 人数: {num_travelers}人
- 老人: {num_elderly}人
- 小孩: {num_children}人
- 预算: {budget_min}~{budget_max} 元/人
- 兴趣偏好: {interests}
{weather_section}

## 约束条件
{constraints}

## 输出格式（每天按此结构）

### 第 N 天 (日期 星期几)
**今日主题**: <一句话概括>

**上午** (8:00-12:00)
- 活动1 (建议时间 + 交通方式)
- 活动2

**午餐** (12:00-13:30)
- 推荐餐厅 + 特色菜 + 人均消费

**下午** (13:30-18:00)
- 活动1 (建议时间)
- 活动2

**晚餐** (18:00-19:30)
- 推荐餐厅 + 特色菜 + 人均消费

**晚间** (19:30-)
- 可选活动或休息建议

**住宿**: 推荐住宿区域 + 理由

**今日交通**: 主要交通方式 + 预估费用

**今日贴士**: 针对当天的特别提醒（穿衣、防晒、预订等）

{adjustment_notes}

## 核心要求
- 每天活动量适中，景点间距离合理，不顺路的不放一起
- 标注大致时间，让用户有预期
- 考虑体能消耗，连续高强度活动后安排轻松时段
- 预算匹配：经济型推公共交通+街头美食，高端型推专车+精品餐厅

请用中文回答，具体详尽。"""


def build_constraints(
    num_elderly: int,
    num_children: int,
    num_travelers: int,
    travel_mode: str,
) -> str:
    constraints = []
    if num_elderly > 0:
        constraints.append(f"- ⚠️ {num_elderly}位老人同行：行程节奏放缓，避免长时间步行和爬山，每2小时安排休息点，优选有电梯/无障碍设施的场所")
    if num_children > 0:
        constraints.append(f"- ⚠️ {num_children}个小孩同行：安排亲子友好景点，避免危险活动，考虑午休时间，准备儿童餐选项")
    if num_travelers >= 5:
        constraints.append(f"- ⚠️ 多人出行({num_travelers}人)：优先餐厅预订，考虑包车/打车，景点门票提前预约")
    if travel_mode == "car":
        constraints.append("- 🚗 自驾出行：考虑停车便利性，避免拥堵路段，每天驾驶不超过3小时")
    elif travel_mode == "train":
        constraints.append("- 🚄 高铁出行：注意车站与市区/景点的接驳交通")
    elif travel_mode == "plane":
        constraints.append("- ✈️ 飞行出行：考虑机场到市区的交通和时间，首尾日行程不宜过满")
    return "\n".join(constraints) if constraints else "无特殊约束"


def build_weather_section(weather_data: dict | None) -> str:
    if not weather_data or "error" in weather_data:
        return ""
    forecast = weather_data.get("forecast", [])
    if not forecast:
        return ""

    lines = ["\n## 天气信息"]
    for day in forecast:
        w = day.get("weather", "")
        precip = day.get("precip_prob", 0)
        flags = []
        if precip > 60 or w.endswith("雨") or w.endswith("雪"):
            flags.append("⚠️不适合户外活动，准备室内备选方案")
        elif precip > 30:
            flags.append("建议携带雨具")

        lines.append(
            f"- {day['date']}: {w}, {day.get('temp_min')}~{day.get('temp_max')}°C, "
            f"降水{precip}%"
            + (f" {' '.join(flags)}" if flags else "")
        )
    return "\n".join(lines)


def build_adjustment_notes(
    num_elderly: int, num_children: int, weather_data: dict | None = None
) -> str:
    notes = []
    if num_elderly > 0:
        notes.append(f"- 考虑到有{num_elderly}位老人，行程安排较为轻松，预留充足休息时间")
    if num_children > 0:
        notes.append(f"- 考虑到有{num_children}个小孩，安排寓教于乐的亲子景点")
    if weather_data and "forecast" in weather_data:
        for day in weather_data.get("forecast", []):
            if day.get("precip_prob", 0) > 60:
                notes.append(f"- {day['date']} 有降雨，该天活动以室内为主")
    return "\n".join(notes)


async def generate_itinerary(
    destination: str,
    departure_from: str,
    start_date: str,
    end_date: str,
    travel_mode: str,
    num_travelers: int,
    num_elderly: int,
    num_children: int,
    budget_min: int,
    budget_max: int,
    interests: list[str],
    weather_data: dict | None = None,
) -> str:
    """Generate a complete itinerary."""
    from datetime import datetime

    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    days = (d2 - d1).days + 1

    constraints = build_constraints(num_elderly, num_children, num_travelers, travel_mode)
    weather_section = build_weather_section(weather_data)
    adjustment_notes = build_adjustment_notes(num_elderly, num_children, weather_data)

    prompt = ITINERARY_PROMPT.format(
        destination=destination,
        departure_from=departure_from,
        start_date=start_date,
        end_date=end_date,
        days=days,
        travel_mode=travel_mode,
        num_travelers=num_travelers,
        num_elderly=num_elderly,
        num_children=num_children,
        budget_min=budget_min,
        budget_max=budget_max,
        interests=", ".join(interests) if interests else "无特殊偏好",
        weather_section=weather_section,
        constraints=constraints,
        adjustment_notes=f"## 动态调整说明\n{adjustment_notes}" if adjustment_notes else "",
    )

    return await chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=4096)
