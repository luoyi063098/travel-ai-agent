from __future__ import annotations

import time
from datetime import datetime

import httpx

from config import WEATHER_CACHE_TTL
from agent.mcp.city_data import lookup_coords, CITY_COORDS


class WeatherTool:
    """Weather query tool using Open-Meteo free API (no API key required)."""

    name = "weather"
    description = "查询指定城市和日期的天气信息，包括温度、降水概率、风速、天气状况等"

    parameters = {
        "city": {"type": "string", "description": "城市名称（中文），如 北京、上海、三亚"},
        "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认为今天"},
    }

    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}

    def _lookup_coords(self, city: str) -> tuple[float, float] | None:
        return lookup_coords(city)

    async def execute(self, city: str = "北京", date: str | None = None) -> dict:
        coords = self._lookup_coords(city)
        if not coords:
            return {"error": f"未找到城市 '{city}' 的坐标，支持的城市: {', '.join(sorted(CITY_COORDS.keys()))}"}

        lat, lon = coords
        today = (date or datetime.now().strftime("%Y-%m-%d"))

        # Check cache
        cache_key = f"{lat:.2f}_{lon:.2f}_{today}"
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < WEATHER_CACHE_TTL:
                return cached

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Get daily forecast
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "daily": [
                            "temperature_2m_max",
                            "temperature_2m_min",
                            "precipitation_probability_max",
                            "weather_code",
                            "wind_speed_10m_max",
                            "uv_index_max",
                        ],
                        "current": [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "weather_code",
                            "wind_speed_10m",
                        ],
                        "timezone": "Asia/Shanghai",
                        "forecast_days": 7,
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
            0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
            45: "有雾", 48: "霜雾", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "阵雨", 81: "中阵雨", 82: "大阵雨",
            95: "雷暴", 96: "冰雹雷暴", 99: "强冰雹雷暴",
        }

        result = {
            "city": city,
            "query_date": target_date,
            "current": {},
            "forecast": [],
        }

        # Current weather
        if "current" in data:
            c = data["current"]
            code = c.get("weather_code", 0)
            result["current"] = {
                "temperature": c.get("temperature_2m", "N/A"),
                "humidity": c.get("relative_humidity_2m", "N/A"),
                "weather": weather_codes.get(code, f"未知({code})"),
                "wind_speed": c.get("wind_speed_10m", "N/A"),
            }

        # Daily forecast
        if "daily" in data:
            daily = data["daily"]
            for i in range(len(daily.get("time", []))):
                day = {
                    "date": daily["time"][i],
                    "temp_max": daily.get("temperature_2m_max", [None])[i],
                    "temp_min": daily.get("temperature_2m_min", [None])[i],
                    "precip_prob": daily.get("precipitation_probability_max", [None])[i],
                    "weather": weather_codes.get(
                        daily.get("weather_code", [0])[i], "未知"
                    ),
                    "wind_max": daily.get("wind_speed_10m_max", [None])[i],
                    "uv_index": daily.get("uv_index_max", [None])[i],
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
                )

        return "\n".join(lines)
