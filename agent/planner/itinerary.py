# agent/planner/itinerary.py
# 行程规划模块 —— 根据目的地、人数、年龄、预算、天气等参数生成每日详细行程

from agent.llm import chat  # 导入 LLM 聊天接口

# 行程规划的系统提示词模板
# 使用 {占位符} 在运行时通过 .format() 注入具体参数
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
    """根据人群特征和出行方式构建约束条件描述字符串。"""
    constraints = []
    # ---- 老人约束 ----
    # 如果有老人同行，需要放缓行程节奏、增加休息点、优先无障碍设施
    if num_elderly > 0:
        constraints.append(f"- ⚠️ {num_elderly}位老人同行：行程节奏放缓，避免长时间步行和爬山，每2小时安排休息点，优选有电梯/无障碍设施的场所")
    # ---- 儿童约束 ----
    # 有小孩时需要亲子景点、危险规避、午休时间、儿童餐
    if num_children > 0:
        constraints.append(f"- ⚠️ {num_children}个小孩同行：安排亲子友好景点，避免危险活动，考虑午休时间，准备儿童餐选项")
    # ---- 多人出行约束 ----
    # 5 人以上需要提前预订、考虑包车
    if num_travelers >= 5:
        constraints.append(f"- ⚠️ 多人出行({num_travelers}人)：优先餐厅预订，考虑包车/打车，景点门票提前预约")
    # ---- 出行方式约束 ----
    # 自驾：注意停车和驾驶时长；高铁：注意接驳；飞机：注意机场距离
    if travel_mode == "car":
        constraints.append("- 🚗 自驾出行：考虑停车便利性，避免拥堵路段，每天驾驶不超过3小时")
    elif travel_mode == "train":
        constraints.append("- 🚄 高铁出行：注意车站与市区/景点的接驳交通")
    elif travel_mode == "plane":
        constraints.append("- ✈️ 飞行出行：考虑机场到市区的交通和时间，首尾日行程不宜过满")
    # 如果没有特殊约束，返回"无特殊约束"
    return "\n".join(constraints) if constraints else "无特殊约束"


def build_weather_section(weather_data: dict | None) -> str:
    """根据天气数据构建天气信息的格式化字符串（供行程提示词使用）。"""
    # 如果天气数据为空或包含错误信息，返回空字符串（不展示天气部分）
    if not weather_data or "error" in weather_data:
        return ""
    forecast = weather_data.get("forecast", [])
    if not forecast:
        return ""

    # 逐日解析天气预报，生成 Markdown 格式的天气段落
    lines = ["\n## 天气信息"]
    for day in forecast:
        # 提取该天的天气描述、降水概率
        w = day.get("weather", "")
        precip = day.get("precip_prob", 0)
        flags = []  # 存储警示标记

        # ---- 降水风险评估 ----
        # 降水概率 > 60% 或明确下雨/下雪：强烈建议准备室内备选
        if precip > 60 or w.endswith("雨") or w.endswith("雪"):
            flags.append("⚠️不适合户外活动，准备室内备选方案")
        # 降水概率 30%~60%：建议携带雨具
        elif precip > 30:
            flags.append("建议携带雨具")

        # 组装该天天气行：日期、天气、温度范围、降水概率、警示标记
        lines.append(
            f"- {day['date']}: {w}, {day.get('temp_min')}~{day.get('temp_max')}°C, "
            f"降水{precip}%"
            + (f" {' '.join(flags)}" if flags else "")
        )
    return "\n".join(lines)


def build_adjustment_notes(
    num_elderly: int, num_children: int, weather_data: dict | None = None
) -> str:
    """构建动态调整说明，告知 LLM 在规划时需要考虑的特殊人群和天气因素。"""
    notes = []
    # ---- 老人调整说明 ----
    if num_elderly > 0:
        notes.append(f"- 考虑到有{num_elderly}位老人，行程安排较为轻松，预留充足休息时间")
    # ---- 儿童调整说明 ----
    if num_children > 0:
        notes.append(f"- 考虑到有{num_children}个小孩，安排寓教于乐的亲子景点")
    # ---- 雨天调整说明 ----
    # 遍历天气预报，如果有天降水概率超过 60%，提示该天以室内活动为主
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
    """生成完整的每日行程规划。"""
    from datetime import datetime  # 导入 datetime 用于日期计算

    # ---- 解析日期并计算总天数 ----
    # 输入格式为 "YYYY-MM-DD"，解析为 datetime 对象
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    # 计算天数：结束日期 - 开始日期 + 1（包含首尾两天）
    days = (d2 - d1).days + 1

    # ---- 构建提示词的各个组成部分 ----
    # 约束条件：根据人群和交通方式生成的文字约束
    constraints = build_constraints(num_elderly, num_children, num_travelers, travel_mode)
    # 天气信息段：逐日天气预报
    weather_section = build_weather_section(weather_data)
    # 动态调整说明：针对老人/小孩/雨天的特别提示
    adjustment_notes = build_adjustment_notes(num_elderly, num_children, weather_data)

    # ---- 将参数注入提示词模板 ----
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
        # 如果有调整说明则添加标题，否则置空
        adjustment_notes=f"## 动态调整说明\n{adjustment_notes}" if adjustment_notes else "",
    )

    # ---- 调用 LLM 生成行程 ----
    # temperature 设为 0.7 以鼓励多样化输出，max_tokens 设为 4096 以保证行程详细
    return await chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=4096)
