import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DB_PATH = os.getenv("DB_PATH", "data/travel_agent.db")

WEATHER_CACHE_TTL = int(os.getenv("WEATHER_CACHE_TTL", "1800"))

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

MAX_REACT_STEPS = 10
MAX_REFLECTION_ROUNDS = 3
MCTS_ITERATIONS = 50
TOT_BREADTH = 3
TOT_DEPTH = 3

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
WEATHER_TIMEOUT = int(os.getenv("WEATHER_TIMEOUT", "15"))

import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("travel_agent")
