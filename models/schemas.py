from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Strategy(str, Enum):
    REACT = "react"
    COT = "cot"
    TOT = "tot"
    MCTS = "mcts"
    REFLEXION = "reflexion"
    DECOMPOSE = "decompose"


class TravelMode(str, Enum):
    CAR = "car"
    TRAIN = "train"
    PLANE = "plane"
    MIXED = "mixed"


class TravelPlanRequest(BaseModel):
    destination: str = Field(..., description="目的地城市/地区")
    start_date: str = Field(..., description="出发日期 YYYY-MM-DD")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD")
    departure_from: str = Field(default="北京", description="出发城市")
    travel_mode: TravelMode = Field(default=TravelMode.TRAIN)
    num_travelers: int = Field(default=1, ge=1)
    num_elderly: int = Field(default=0, ge=0, description="老人数量")
    num_children: int = Field(default=0, ge=0, description="小孩数量")
    budget_min: int = Field(default=2000, ge=0, description="预算下限（元/人）")
    budget_max: int = Field(default=5000, ge=0, description="预算上限（元/人）")
    interests: list[str] = Field(default_factory=list, description="兴趣标签: 自然风光, 历史人文, 美食, 购物, 户外")
    strategy: Optional[Strategy] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    strategy: Optional[Strategy] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    strategy_used: str


class TravelPlanResponse(BaseModel):
    session_id: str
    destination: str
    weather: dict
    itinerary: list[dict]
    food_recommendations: list[dict]
    accommodation_recommendations: list[dict]
    transport_plan: dict
    tips: list[str]
    adjustments: list[str]


class PreferenceUpdate(BaseModel):
    key: str
    value: str


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict


class MCPListToolsResponse(BaseModel):
    tools: list[ToolInfo]


class MCPCallToolRequest(BaseModel):
    name: str
    arguments: dict


class MCPCallToolResponse(BaseModel):
    content: list[dict]
    is_error: bool = False
