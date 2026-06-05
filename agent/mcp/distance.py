"""Distance calculation tool for estimating travel distances between cities."""

from __future__ import annotations

from agent.mcp.city_data import lookup_coords, haversine_distance


class DistanceTool:
    """Calculate approximate distance and travel time between cities."""

    name = "distance"
    description = "计算两个城市之间的直线距离和估算出行时间"

    parameters = {
        "from_city": {"type": "string", "description": "出发城市名称"},
        "to_city": {"type": "string", "description": "目的城市名称"},
    }

    SPEEDS = {"car": 100, "train": 250, "plane": 700}

    async def execute(self, from_city: str = "北京", to_city: str = "上海") -> dict:
        from_coords = lookup_coords(from_city)
        to_coords = lookup_coords(to_city)

        if not from_coords:
            return {"error": f"未找到出发城市: {from_city}"}
        if not to_coords:
            return {"error": f"未找到目的城市: {to_city}"}

        distance = haversine_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])

        return {
            "from": from_city,
            "to": to_city,
            "distance_km": round(distance),
            "estimated_time": {
                "car": f"{distance / self.SPEEDS['car']:.1f}小时",
                "train": f"{distance / self.SPEEDS['train']:.1f}小时",
                "plane": f"{distance / self.SPEEDS['plane']:.1f}小时 (含机场时间)",
            },
            "recommendation": self._recommend(distance),
        }

    @staticmethod
    def _recommend(distance: float) -> str:
        if distance < 300:
            return "🚗 推荐自驾或高铁，1-3小时到达"
        elif distance < 800:
            return "🚄 推荐高铁，方便快捷"
        elif distance < 1500:
            return "🚄 高铁或 ✈️ 飞机均可，高铁4-6小时，飞行含机场约3小时"
        else:
            return "✈️ 推荐飞机，距离较远"
