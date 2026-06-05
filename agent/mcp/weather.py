from __future__ import annotations
# 启用 PEP 604 的类型注解语法（允许使用 | 代替 Union），让类型标注更简洁

import time

from datetime import datetime

import httpx
# 导入 httpx 异步 HTTP 客户端库，用于向 Open-Meteo API 发送请求并获取天气数据

from config import WEATHER_CACHE_TTL
# 从项目配置中读取天气查询结果的缓存有效期（单位：秒），避免在短时间内重复请求 API

from agent.mcp.city_data import lookup_coords, CITY_COORDS
# lookup_coords：根据城市中文名查找对应经纬度坐标，返回 (纬度, 经度) 元组或 None
# CITY_COORDS：预定义的城市坐标字典，键为城市名，值为 (纬度, 经度)


class WeatherTool:
    """Weather query tool using Open-Meteo free API (no API key required)."""

    name = "weather"

    description = "查询指定城市和日期的天气信息，包括温度、降水概率、风速、天气状况等"
    # 工具的功能描述文本，会被 LLM 读取以决定何时调用此工具

    parameters = {
        "city": {"type": "string", "description": "城市名称（中文），如 北京、上海、三亚"},
        "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认为今天"},
    }
    # 工具的参数 schema 定义，描述每个参数的类型和含义，LLM 会根据此 schema 生成参数值

    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}
        # 键：缓存的 key（由经纬度和日期拼接而成）
        # 值：(存入时的时间戳, 缓存的天气数据字典)
        # 这样设计可以在查询缓存时对比时间戳判断是否过期

    def _lookup_coords(self, city: str) -> tuple[float, float] | None:
        return lookup_coords(city)

    async def execute(self, city: str = "北京", date: str | None = None) -> dict:

        coords = self._lookup_coords(city)

        if not coords:
            return {"error": f"未找到城市 '{city}' 的坐标，支持的城市: {', '.join(sorted(CITY_COORDS.keys()))}"}
            # 如果坐标查找失败（城市名不在已知列表中），立即返回错误信息
            # 错误信息中列出所有支持的城市名称，方便用户修正输入

        lat, lon = coords

        today = (date or datetime.now().strftime("%Y-%m-%d"))
        # 如果调用方传入了 date 参数则使用该日期，否则获取当前系统时间的日期字符串
        # 使用 strftime("%Y-%m-%d") 格式化为标准日期格式，如 "2026-06-05"

        # Check cache
        cache_key = f"{lat:.2f}_{lon:.2f}_{today}"
        # 构建缓存键：将经纬度保留两位小数后与日期拼接
        # 示例：lat=39.9042, lon=116.4074, today=2026-06-05 -> "39.90_116.41_2026-06-05"
        # 保留两位小数的目的是允许微小坐标差异的查询共享缓存

        if cache_key in self._cache:
            # 检查缓存字典中是否存在当前查询对应的缓存项

            ts, cached = self._cache[cache_key]
            # 解包缓存项：ts 为存入时的时间戳，cached 为之前缓存的天气数据字典

            if time.time() - ts < WEATHER_CACHE_TTL:
                # 计算当前时间与缓存存入时间的差值
                # 如果差值小于配置的缓存有效期（WEATHER_CACHE_TTL 秒），说明缓存仍有效
                return cached
                # 直接返回缓存的天气数据，避免重复请求 API

        try:

            async with httpx.AsyncClient(timeout=10) as client:
                # 创建异步 HTTP 客户端，设置 10 秒超时
                # async with 确保请求完成后自动释放连接资源

                # Get daily forecast
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    # Open-Meteo 免费天气 API 的端点地址
                    # 该 API 无需 API Key，只需提供经纬度参数即可获取天气数据

                    params={
                        # HTTP 查询参数，httpx 会自动进行 URL 编码和拼接
                        "latitude": lat,
                        # 纬度，由之前的城市坐标查找获得

                        "longitude": lon,
                        # 经度，由之前的城市坐标查找获得

                        "daily": [
                            # 请求每日预报数据，参数值为列表，httpx 会自动处理为重复的 key
                            "temperature_2m_max",
                            # 每日最高气温（2 米高度），单位：摄氏度
                            "temperature_2m_min",
                            # 每日最低气温（2 米高度），单位：摄氏度
                            "precipitation_probability_max",
                            # 最大降水概率，取值范围 0-100，单位：百分比
                            "weather_code",
                            # WMO 天气代码（整数），用于映射到文字描述（晴天/多云/雨/雪等）
                            "wind_speed_10m_max",
                            # 每日最大风速（10 米高度），单位：km/h
                            "uv_index_max",
                            # 每日最大紫外线指数
                        ],

                        "current": [
                            # 请求当前实时天气数据
                            "temperature_2m",
                            # 当前温度（2 米高度），单位：摄氏度
                            "relative_humidity_2m",
                            # 当前相对湿度（2 米高度），单位：百分比
                            "weather_code",
                            # 当前 WMO 天气代码
                            "wind_speed_10m",
                            # 当前风速（10 米高度），单位：km/h
                        ],

                        "timezone": "Asia/Shanghai",
                        # 时区设置为"Asia/Shanghai"，确保返回的时间数据为中国时区

                        "forecast_days": 7,
                        # 请求未来 7 天的每日预报数据
                    },
                )

                resp.raise_for_status()

                data = resp.json()

        except Exception as e:
            return {"error": f"天气查询失败: {e}"}

        weather = self._parse_weather(data, city, today)

        self._cache[cache_key] = (time.time(), weather)

        return weather

    def _parse_weather(self, data: dict, city: str, target_date: str) -> dict:

        weather_codes = {
            # WMO（世界气象组织）天气代码到中文描述的映射字典
            # 代码范围 0-99，覆盖了大多数常见天气状况

            0: "晴天",        # 天空无云或少云，晴朗天气
            1: "大部晴朗",    # 大部分时间晴朗，可能有少量云层
            2: "多云",        # 云量较多，但仍有阳光透射
            3: "阴天",        # 云层覆盖整个天空，无阳光直射

            45: "有雾",       # 水平能见度因雾降低
            48: "霜雾",       # 雾伴随地面结霜

            51: "小毛毛雨",   # 极细微的雨滴，降水量很小
            53: "毛毛雨",     # 中等强度的细密雨滴
            55: "大毛毛雨",   # 密集的细密雨滴

            61: "小雨",       # 降水量小于 2.5 mm/h
            63: "中雨",       # 降水量 2.5-10 mm/h
            65: "大雨",       # 降水量大于 10 mm/h

            71: "小雪",       # 降雪量较小
            73: "中雪",       # 降雪量中等
            75: "大雪",       # 降雪量较大

            80: "阵雨",       # 短时间内的降雨，来得快去得也快
            81: "中阵雨",     # 中等强度的阵雨
            82: "大阵雨",     # 强度较大的阵雨

            95: "雷暴",       # 伴随雷电的暴风雨
            96: "冰雹雷暴",   # 伴随冰雹的雷暴
            99: "强冰雹雷暴", # 伴随大冰雹的强雷暴
        }

        result = {
            "city": city,
            "query_date": target_date,
            "current": {},
            # 当前实时天气数据容器，初始为空字典
            # 如果 API 返回了 current 数据则会填充，否则保持空字典

            "forecast": [],
            # 未来多日预报列表容器，初始为空列表
            # 每项为一个字典，包含当天的最高温度、最低温度、降水概率等信息
        }

        # Current weather
        if "current" in data:
            # 检查 API 响应中是否包含当前天气数据（current 字段）
            # 某些情况下（如 API 版本变更或请求参数问题）可能缺失该字段

            c = data["current"]

            code = c.get("weather_code", 0)
            # 获取 WMO 天气代码，如果缺失则默认取 0（晴天）

            result["current"] = {
                "temperature": c.get("temperature_2m", "N/A"),
                # 当前温度（摄氏度），如果数据缺失则返回 "N/A"

                "humidity": c.get("relative_humidity_2m", "N/A"),
                # 当前相对湿度（百分比），如果数据缺失则返回 "N/A"

                "weather": weather_codes.get(code, f"未知({code})"),
                # 将 WMO 代码映射为中文天气描述
                # 如果代码不在映射表中，格式化为"未知(代码值)"以便调试

                "wind_speed": c.get("wind_speed_10m", "N/A"),
                # 当前风速（km/h），如果数据缺失则返回 "N/A"
            }

        # Daily forecast
        if "daily" in data:
            # 检查 API 响应中是否包含每日预报数据（daily 字段）

            daily = data["daily"]

            for i in range(len(daily.get("time", []))):
                # 遍历 daily 中的"time"数组长度，确定有多少天的预报数据
                # time 数组的每个元素是一个日期字符串，如 "2026-06-05"
                # 使用索引 i 同时从其他平行数组中取出对应的气象数据
                # 这种结构是 Open-Meteo API 的设计模式：多个等长数组共用索引

                day = {
                    "date": daily["time"][i],
                    # 预报日期，格式为 "YYYY-MM-DD"

                    "temp_max": daily.get("temperature_2m_max", [None])[i],
                    # 当日最高气温（摄氏度）
                    # 使用 .get() 提供默认空列表 [None]，避免 KeyError

                    "temp_min": daily.get("temperature_2m_min", [None])[i],
                    # 当日最低气温（摄氏度）

                    "precip_prob": daily.get("precipitation_probability_max", [None])[i],
                    # 当日最大降水概率（百分比），0 表示无降水可能，100 表示必然降水

                    "weather": weather_codes.get(
                        daily.get("weather_code", [0])[i], "未知"
                    ),
                    # 将 WMO 天气代码映射为中文描述，默认代码为 0（晴天）

                    "wind_max": daily.get("wind_speed_10m_max", [None])[i],
                    # 当日最大风速（km/h）

                    "uv_index": daily.get("uv_index_max", [None])[i],
                    # 当日最大紫外线指数，数值越大紫外线越强
                }

                result["forecast"].append(day)

        return result

    @staticmethod
    def format_for_prompt(weather_data: dict) -> str:
        """Format weather data as a text summary for LLM prompt injection."""

        if "error" in weather_data:
            return f"天气数据不可用: {weather_data['error']}"

        lines = [f"### {weather_data['city']} 天气"]

        current = weather_data.get("current", {})

        if current:

            lines.append(
                f"当前: {current.get('weather')}, {current.get('temperature')}°C, "
                f"湿度 {current.get('humidity')}%, 风速 {current.get('wind_speed')} km/h"
            )

        forecast = weather_data.get("forecast", [])

        if forecast:

            lines.append("\n未来几日预报:")

            for day in forecast:

                lines.append(
                    f"  {day['date']}: {day['weather']}, "
                    f"{day['temp_min']}~{day['temp_max']}°C, "
                    f"降水概率 {day['precip_prob']}%, "
                    f"UV指数 {day['uv_index']}"

                    + (" ⚠️不适合户外活动" if day.get("precip_prob", 0) > 60 or day.get("weather", "").endswith("雨") or day.get("weather", "").endswith("雪") else "")
                    # 根据天气条件给出户外活动建议：
                    # 如果降水概率大于 60%，或者天气描述以"雨"或"雪"结尾，则附加警告标识
                    # 使用 endswith 判断是因为多个天气代码都对应以"雨"或"雪"结尾的中文描述（如"小雨"、"阵雨"、"大雪"）
                )

        return "\n".join(lines)
