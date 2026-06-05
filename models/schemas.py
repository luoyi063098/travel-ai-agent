"""
Pydantic 数据模型定义（请求 / 响应 / 枚举）。
用于 API 参数校验和结构化数据交换。
"""

from __future__ import annotations                      # 启用类型注解的延迟求值，支持前向引用

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field                    # BaseModel：Pydantic 数据模型基类；Field：字段约束与描述


class Strategy(str, Enum):
    """智能体推理策略枚举。"""
    REACT = "react"                                      # ReAct：推理 + 行动交替循环
    COT = "cot"                                          # Chain-of-Thought：思维链逐步推理
    TOT = "tot"                                          # Tree-of-Thoughts：树状多分支推理
    MCTS = "mcts"                                        # Monte Carlo Tree Search：蒙特卡洛树搜索
    REFLEXION = "reflexion"                              # Reflexion：带自我反思的推理
    DECOMPOSE = "decompose"                              # Decompose：复杂问题分解为子任务


class TravelMode(str, Enum):
    """出行方式枚举。"""
    CAR = "car"                                          # 自驾 / 汽车
    TRAIN = "train"                                      # 火车 / 高铁
    PLANE = "plane"                                      # 飞机
    MIXED = "mixed"                                      # 混合多种交通方式


class TravelPlanRequest(BaseModel):
    """用户提交的旅游规划请求体。"""
    destination: str = Field(..., description="目的地城市/地区")          # 必填：旅游目的地
    start_date: str = Field(..., description="出发日期 YYYY-MM-DD")      # 必填：行程开始日期
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD")        # 必填：行程结束日期
    departure_from: str = Field(default="上海", description="出发城市")   # 出发城市，默认上海
    travel_mode: TravelMode = Field(default=TravelMode.TRAIN)            # 首选交通方式，默认火车
    num_travelers: int = Field(default=1, ge=1)                          # 出行总人数，至少 1 人
    num_elderly: int = Field(default=0, ge=0, description="老人数量")    # 同行老人数量
    num_children: int = Field(default=0, ge=0, description="小孩数量")   # 同行小孩数量
    budget_min: int = Field(default=2000, ge=0, description="预算下限（元/人）")   # 人均预算下限
    budget_max: int = Field(default=5000, ge=0, description="预算上限（元/人）")   # 人均预算上限
    interests: list[str] = Field(default_factory=list, description="兴趣标签: 自然风光, 历史人文, 美食, 购物, 户外")  # 兴趣标签列表
    strategy: Optional[Strategy] = None                                  # 可选：指定推理策略，为空则由后端自动选择


class ChatRequest(BaseModel):
    """用户发送的对话请求体。"""
    message: str                                                         # 必填：用户消息内容
    session_id: Optional[str] = None                                     # 可选：会话 ID，为空则创建新会话
    strategy: Optional[Strategy] = None                                  # 可选：指定推理策略


class ChatResponse(BaseModel):
    """对话响应用户。"""
    session_id: str                                                      # 当前会话 ID
    response: str                                                        # AI 回复文本
    strategy_used: str                                                   # 实际使用的推理策略名称


class TravelPlanResponse(BaseModel):
    """旅游规划结果响应。"""
    session_id: str                                                      # 当前会话 ID
    destination: str                                                     # 目的地名称
    weather: dict                                                        # 目的地天气预报数据
    itinerary: list[dict]                                                # 逐日行程安排列表
    food_recommendations: list[dict]                                     # 美食推荐列表
    accommodation_recommendations: list[dict]                            # 住宿推荐列表
    transport_plan: dict                                                 # 交通方案详情
    tips: list[str]                                                      # 出行贴士列表
    adjustments: list[str]                                               # 针对特殊需求的调整建议列表


class PreferenceUpdate(BaseModel):
    """用户偏好更新请求体。"""
    key: str                                                             # 偏好键名
    value: str                                                           # 偏好值


class ToolInfo(BaseModel):
    """MCP 工具元信息描述。"""
    name: str                                                            # 工具名称
    description: str                                                     # 工具功能描述
    parameters: dict                                                     # 工具参数 JSON Schema


class MCPListToolsResponse(BaseModel):
    """列出所有可用 MCP 工具的响应。"""
    tools: list[ToolInfo]                                                # 工具信息列表


class MCPCallToolRequest(BaseModel):
    """调用 MCP 工具的请求体。"""
    name: str                                                            # 要调用的工具名称
    arguments: dict                                                      # 工具参数键值对


class MCPCallToolResponse(BaseModel):
    """MCP 工具调用结果响应。"""
    content: list[dict]                                                  # 工具返回的内容列表（支持多段文本/图片等）
    is_error: bool = False                                               # 是否发生错误，True 表示调用失败
