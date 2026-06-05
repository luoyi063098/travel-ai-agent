# agent/planner/transport.py
# 交通规划模块 —— 为大模型提供交通规划的提示词，生成包含大交通、市内交通、
# 景点间交通及费用估算的完整交通方案

from agent.llm import chat  # 导入 LLM 聊天接口

# 交通规划提示词模板
# 让 LLM 扮演交通规划专家，根据出发地、目的地、出行方式等参数提供全面交通建议
TRANSPORT_PROMPT = """你是一个交通规划专家，熟悉全国交通网络和城市内部交通。

## 场景信息
- 出发地: {departure_from}
- 目的地: {destination}
- 出行方式: {travel_mode}
- 日期: {start_date} 至 {end_date}
- 人数: {num_travelers}人
- 老人: {num_elderly}人
- 小孩: {num_children}人
- 预算: {budget_min}~{budget_max} 元/人

## 输出要求

### 1. 大交通（出发地 → 目的地）
- 推荐方案（含交通方式、大致时间、参考票价）
- 备选方案（如高铁备选飞机）
- 购票建议（提前几天、哪个平台）
- 首日/末日时间安排（几点出发、几点到达、缓冲时间）

### 2. 市内交通
- 当地主要交通方式介绍（地铁/公交/打车/共享单车/租车）
- 支付方式（现金/公交卡/二维码）
- 推荐方案：结合行程的市内交通建议
- 老人/小孩：是否需要租车或包车

### 3. 景点间交通
- 主要景点之间的交通方式和时间
- 是否需要包车一日游
- 景区内部交通（缆车、观光车等）

### 4. 费用估算
- 大交通往返费用
- 市内交通每日预算
- 总交通费用估算

请用中文回答，提供具体可操作的建议。"""


async def plan_transport(
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
) -> str:
    """生成交通规划方案。"""
    # 将场景参数注入交通规划提示词模板
    prompt = TRANSPORT_PROMPT.format(
        departure_from=departure_from,
        destination=destination,
        travel_mode=travel_mode,
        start_date=start_date,
        end_date=end_date,
        num_travelers=num_travelers,
        num_elderly=num_elderly,
        num_children=num_children,
        budget_min=budget_min,
        budget_max=budget_max,
    )
    # 调用 LLM 生成交通方案，temperature 设为 0.6 以保持信息的准确性和一致性
    return await chat([{"role": "user", "content": prompt}], temperature=0.6)
