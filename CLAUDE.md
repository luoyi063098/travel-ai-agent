# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# Start server
python main.py

# Run in development mode with auto-reload
uvicorn main:app --reload --port 8000

# Run with custom port
python main.py  # respects SERVER_PORT env var, or uvicorn main:app --port 8080
```

## Architecture

This is a travel planning AI Agent built on FastAPI + DeepSeek LLM with multiple reasoning strategies.

**Core flow**: API request → `TravelAgent` selects reasoning strategy → strategy engine reasons with LLM + MCP tools → response with memory persistence.

**Key modules**:

- `agent/core.py` — `TravelAgent` central coordinator + `StrategySelector` that auto-picks reasoning strategy based on task keywords
- `agent/llm.py` — Thin wrapper around OpenAI-compatible DeepSeek client (`deepseek-chat` model)
- `agent/mcp/provider.py` — MCP protocol: tool registry, `list_tools()` and `call_tool()`. Weather tool registered by default.
- `agent/mcp/weather.py` — Open-Meteo free API wrapper with city coordinate lookup and TTL cache. Format output via `format_for_prompt()`.
- `agent/memory/__init__.py` — SQLite-backed `MemoryStore` managing sessions, messages, and key-value preferences. `build_context()` injects preferences as system messages before conversation history.
- `agent/reasoning/` — 6 strategy engines, each exposing `async reason()` returning dict with `answer` key
- `agent/planner/` — Specialized travel planning functions used by `TravelAgent.plan_travel()`

**Strategy selection logic** (in `StrategySelector.select()`): optimization/route → `tot`, planning/itinerary keywords → `decompose`, improvement/revision → `reflexion`, analysis/comparison → `cot`, default → `react`.

**`/api/plan` workflow**: weather → destination intro → itinerary → food → accommodation → transport → adjuster → tips. All steps run sequentially, weather data passes through to itinerary and adjuster.

**Database**: Single SQLite file at `DB_PATH` (default `data/travel_agent.db`). Tables: `sessions`, `messages` (FK to sessions, indexed by session_id+created_at), `preferences` (key-value).

**Config**: All settings in `config.py`, loaded from env vars with defaults. `DEEPSEEK_API_KEY` is required.
