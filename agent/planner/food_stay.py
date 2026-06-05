"""Food and accommodation recommendations."""

from agent.llm import chat

FOOD_PROMPT = """你是一个资深美食推荐专家，对全国各地的特色美食和口碑餐厅了如指掌。

## 场景信息
- 目的地: {destination}
- 人数: {num_travelers}人
- 老人: {num_elderly}人
- 小孩: {num_children}人
- 预算: {budget_min}~{budget_max} 元/人

## 输出要求

### 必吃美食 (5-8道)
每道说明：
- 菜名 + 一句话描述
- 人均参考价格
- 推荐理由（为什么是当地代表）
- 适合场景（早餐/正餐/夜宵/小吃）

### 推荐餐厅 (3-5家)
每家说明：
- 餐厅名称 + 大致位置/商圈
- 招牌菜推荐
- 人均消费
- 预订建议（是否需要、提前多久）
- 特别说明（排队情况、环境、适合多人等）

### 饮食注意事项
- 老人饮食建议（软烂、清淡选项）
- 小孩饮食建议（不辣、趣味选项）
- 当地饮食特点提醒（如：偏辣/偏甜/海鲜多）

请用中文回答，推荐具体可操作的选项。"""

ACCOMMODATION_PROMPT = """你是一个住宿推荐专家，熟悉各旅游城市的酒店和民宿资源。

## 场景信息
- 目的地: {destination}
- 人数: {num_travelers}人
- 老人: {num_elderly}人
- 小孩: {num_children}人
- 预算: {budget_min}~{budget_max} 元/人

## 输出要求

### 住宿推荐 (3-5个选项)
每个选项说明：
- 类型（酒店/民宿/青旅）+ 推荐档次
- 推荐具体区域或商圈 + 理由
- 大致价格区间（每晚）
- 适合人群和优势
- 不足或注意事项

### 区域分析
- 不同住宿区域的特点对比（交通、餐饮、景点距离、安静程度）
- 针对本次出行的最优区域建议

### 特殊需求考虑
- 老人：电梯、低楼层、安静、步行距离
- 小孩：亲子设施、安全、加床服务
- 多人：家庭房、连通房、民宿整租

请用中文回答。"""


async def recommend_food(
    destination: str,
    num_travelers: int,
    num_elderly: int,
    num_children: int,
    budget_min: int,
    budget_max: int,
) -> str:
    prompt = FOOD_PROMPT.format(
        destination=destination,
        num_travelers=num_travelers,
        num_elderly=num_elderly,
        num_children=num_children,
        budget_min=budget_min,
        budget_max=budget_max,
    )
    return await chat([{"role": "user", "content": prompt}], temperature=0.7)


async def recommend_accommodation(
    destination: str,
    num_travelers: int,
    num_elderly: int,
    num_children: int,
    budget_min: int,
    budget_max: int,
) -> str:
    prompt = ACCOMMODATION_PROMPT.format(
        destination=destination,
        num_travelers=num_travelers,
        num_elderly=num_elderly,
        num_children=num_children,
        budget_min=budget_min,
        budget_max=budget_max,
    )
    return await chat([{"role": "user", "content": prompt}], temperature=0.7)
