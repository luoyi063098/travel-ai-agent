"""
应用全局配置文件。
所有可配置项优先通过环境变量读取，若未设置则使用默认值。
"""

import os
from dotenv import load_dotenv                          # 用于从 .env 文件加载环境变量

load_dotenv()                                           # 加载项目根目录下的 .env 文件到进程环境变量

# ─── DeepSeek LLM 配置 ──────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")    # DeepSeek API 密钥，必须配置才能调用大模型
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # API 的基础地址，可切换为私有部署地址
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")  # 使用的模型名称，可切换为其他 DeepSeek 模型

# ─── 数据库配置 ──────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/travel_agent.db")  # SQLite 数据库文件路径，用于持久化会话和偏好数据

# ─── 缓存配置 ────────────────────────────────────────────────────
WEATHER_CACHE_TTL = int(os.getenv("WEATHER_CACHE_TTL", "1800"))  # 天气缓存有效期（秒），默认 30 分钟

# ─── HTTP 服务配置 ──────────────────────────────────────────────
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")       # HTTP 服务器监听地址，0.0.0.0 表示监听所有网络接口
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))     # HTTP 服务器监听端口

# ─── 推理策略参数 ──────────────────────────────────────────────
MAX_REACT_STEPS = 10                                    # ReAct 策略最大推理-行动循环步数
MAX_REFLECTION_ROUNDS = 3                               # Reflexion 策略最大反思轮数
MCTS_ITERATIONS = 50                                    # MCTS（蒙特卡洛树搜索）策略的模拟迭代次数
TOT_BREADTH = 3                                         # Tree-of-Thoughts 每层保留的分支数（广度）
TOT_DEPTH = 3                                           # Tree-of-Thoughts 最大搜索深度

# ─── 网络请求超时与重试 ──────────────────────────────────────────
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # 调用大模型失败时的最大重试次数
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))       # 大模型 API 请求超时时间（秒）
WEATHER_TIMEOUT = int(os.getenv("WEATHER_TIMEOUT", "15"))  # 天气 API 请求超时时间（秒）

# ─── 日志配置 ────────────────────────────────────────────────────
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")              # 日志级别，可选 DEBUG / INFO / WARNING / ERROR
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),    # 将字符串日志级别映射为 logging 模块的常量
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # 日志格式：时间 [级别] 记录器名: 消息
    datefmt="%Y-%m-%d %H:%M:%S",                        # 时间戳格式，精确到秒
)
logger = logging.getLogger("travel_agent")              # 获取应用全局日志记录器，其他模块通过该名称继承配置
