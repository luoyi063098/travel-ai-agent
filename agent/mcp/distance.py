"""Distance calculation tool for estimating travel distances between cities."""

from __future__ import annotations
# 启用 PEP 604 的类型注解语法（允许使用 | 代替 Union），让类型标注更简洁

from agent.mcp.city_data import lookup_coords, haversine_distance
# lookup_coords：根据城市中文名查找经纬度坐标，返回 (纬度, 经度) 元组或 None
# haversine_distance：利用 Haversine 球面三角公式计算两点间的直线距离（单位：公里）


class DistanceTool:
    """Calculate approximate distance and travel time between cities."""

    name = "distance"

    description = "计算两个城市之间的直线距离和估算出行时间"
    # 工具的功能描述文本，LLM 会根据此描述决定何时调用该工具
    # 注意：计算的是球面直线距离（大圆距离），而非实际道路或铁路里程

    parameters = {
        "from_city": {"type": "string", "description": "出发城市名称"},
        "to_city": {"type": "string", "description": "目的城市名称"},
    }
    # 工具的参数 schema 定义
    # LLM 会根据此 schema 从用户输入中提取参数值

    SPEEDS = {"car": 100, "train": 250, "plane": 700}
    # 类属性：三种常用交通工具的平均行驶速度（单位：km/h）
    # car：汽车，平均时速 100 km/h（考虑高速公路限速和城市路段）
    # train：高铁，平均时速 250 km/h（考虑中间停站减速等因素）
    # plane：飞机，平均时速 700 km/h（巡航速度，不含起降和地面时间）
    # 这些速度值是经验估算值，用于将距离换算为大致出行时间

    async def execute(self, from_city: str = "北京", to_city: str = "上海") -> dict:

        from_coords = lookup_coords(from_city)

        to_coords = lookup_coords(to_city)

        if not from_coords:
            return {"error": f"未找到出发城市: {from_city}"}

        if not to_coords:
            return {"error": f"未找到目的城市: {to_city}"}

        distance = haversine_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        # 调用 Haversine 球面距离公式计算两点间的直线距离
        # 参数顺序：纬度1, 经度1, 纬度2, 经度2
        # from_coords[0]：出发城市纬度，from_coords[1]：出发城市经度
        # to_coords[0]：目的城市纬度，to_coords[1]：目的城市经度
        # 返回值：两点间的球面直线距离，单位是公里（km）

        return {
            "from": from_city,

            "to": to_city,

            "distance_km": round(distance),

            "estimated_time": {
                # 为不同交通工具估算出行时间
                # 实际出行时间会因道路状况、交通拥堵、停站次数等因素而不同
                # 这里的估算是基于直线距离和平均速度的理想值

                "car": f"{distance / self.SPEEDS['car']:.1f}小时",
                # 汽车时间 = 直线距离 / 100 km/h
                # 注意：实际公路距离通常比直线距离长 20%-50%，因此实际驾车时间更长
                # 使用 :.1f 格式保留一位小数

                "train": f"{distance / self.SPEEDS['train']:.1f}小时",
                # 高铁时间 = 直线距离 / 250 km/h
                # 高铁线路通常较为直顺，与直线距离偏差较小
                # 使用 :.1f 格式保留一位小数

                "plane": f"{distance / self.SPEEDS['plane']:.1f}小时 (含机场时间)",
                # 飞行时间 = 直线距离 / 700 km/h
                # 添加"含机场时间"说明：最终结果已考虑值机、安检、行李提取等地面耗时
                # 使用 :.1f 格式保留一位小数
            },

            "recommendation": self._recommend(distance),
        }

    @staticmethod
    def _recommend(distance: float) -> str:

        if distance < 300:
            # 距离小于 300 公里：短途出行
            return "🚗 推荐自驾或高铁，1-3小时到达"
            # 推荐自驾或高铁，两种方式时间相近
            # 自驾更灵活，高铁更舒适

        elif distance < 800:
            # 距离在 300 到 800 公里之间：中途出行
            return "🚄 推荐高铁，方便快捷"
            # 推荐高铁，这个距离高铁相比汽车有显著时间优势
            # 相比飞机无需值机候机，全程时间更有优势

        elif distance < 1500:
            # 距离在 800 到 1500 公里之间：中长途出行
            return "🚄 高铁或 ✈️ 飞机均可，高铁4-6小时，飞行含机场约3小时"
            # 高铁和飞机各有优势，取决于出发地和目的地的机场/高铁站位置
            # 高铁提供门到门的便利性，飞机速度更快但需考虑机场往返时间

        else:
            # 距离大于 1500 公里：长途出行
            return "✈️ 推荐飞机，距离较远"
            # 推荐飞机，高铁在这个距离下耗时过长（6 小时以上）
            # 飞机即使算上机场时间也通常比高铁快
