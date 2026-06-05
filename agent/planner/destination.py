"""Destination introduction and research."""

from agent.llm import chat

DESTINATION_PROMPT = """你是一个资深旅行目的地专家，对国内各大旅游城市和景点有深入了解。

请为用户全面介绍旅行目的地，按以下结构组织：

## 目的地概述
- 城市/地区的历史文化背景（1-2句）
- 核心特色标签（如：海滨度假、历史文化、美食天堂、自然风光）

## 必游景点 TOP 5
每个景点说明：
- 名称 + 推荐理由（1-2句）
- 适合人群（亲子/情侣/老人/独自）
- 建议游玩时长 + 最佳时段
- 大致门票参考

## 最佳旅游季节
- 各季节特点对比
- 当前出行月份的建议

## 当地美食文化
- 特色菜系/小吃概述
- 2-3道必尝美食

## 风俗与注意事项
- 当地特殊风俗习惯
- 安全提示（防坑、防晒、保暖等）
- 交通特色（如：打车贵、公交方便等）

请用中文回答，信息具体准确，不要只说空泛的好话，也要如实提醒不足之处。"""


async def get_destination_intro(
    destination: str,
    interests: list[str] | None = None,
    travel_month: str | None = None,
) -> str:
    """Generate destination introduction."""
    interest_hint = ""
    if interests:
        interest_hint = f"\n用户兴趣偏好: {', '.join(interests)}"

    month_hint = ""
    if travel_month:
        month_hint = f"\n出行月份: {travel_month}"

    prompt = f"{DESTINATION_PROMPT}{interest_hint}{month_hint}\n\n请介绍目的地: {destination}"

    return await chat([{"role": "user", "content": prompt}], temperature=0.6)
