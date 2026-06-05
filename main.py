"""
FastAPI Server for Travel AI Agent.

Endpoints:
  POST /api/chat       - Conversational travel assistant
  POST /api/plan       - Full travel plan generation
  GET  /api/sessions/{id} - Session history
  GET  /api/preferences   - Get user preferences
  PUT  /api/preferences   - Update user preference
  GET  /api/health        - Health check
  GET  /api/tools         - List available MCP tools
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

from config import SERVER_HOST, SERVER_PORT
from models.schemas import (
    ChatRequest,
    ChatResponse,
    TravelPlanRequest,
    PreferenceUpdate,
)
from agent.core import travel_agent
from agent.mcp.provider import mcp_provider
from agent.memory import memory_store

logger = logging.getLogger("travel_agent.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Travel AI Agent server...")

    # Validate API key
    from config import DEEPSEEK_API_KEY
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set! Set it via env var or .env file.")
    else:
        logger.info("DeepSeek API Key: configured")

    await memory_store.init()
    logger.info("Memory store initialized (DB: data/travel_agent.db)")
    logger.info("MCP tools: %s", [t.name for t in mcp_provider.list_tools()])
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="Travel AI Agent",
    description="基于 DeepSeek 的智能旅行规划 Agent，支持 ReAct/CoT/ToT/MCTS/Reflexion 多种推理策略",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("%s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Travel AI Agent</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; min-height:100vh; }
  .container { max-width:720px; margin:0 auto; padding:40px 20px; }
  h1 { font-size:2em; margin-bottom:4px; }
  .subtitle { color:#64748b; margin-bottom:28px; font-size:0.9em; }
  .card { background: #1e293b; border:1px solid #334155; border-radius:12px; padding:28px; margin-bottom:20px; }
  .card h2 { font-size:1.1em; color:#38bdf8; margin-bottom:20px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }
  .field { flex:1; min-width:200px; display:flex; flex-direction:column; }
  .field label { font-size:0.85em; color:#94a3b8; margin-bottom:6px; }
  .field input, .field select { padding:10px 12px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; font-size:0.95em; outline:none; }
  .field input:focus, .field select:focus { border-color:#38bdf8; }
  .field input::placeholder { color:#475569; }
  .check-row { display:flex; gap:24px; align-items:center; margin-bottom:16px; }
  .check-row label { display:flex; align-items:center; gap:8px; font-size:0.9em; color:#cbd5e1; cursor:pointer; }
  .check-row input[type=checkbox] { width:18px; height:18px; accent-color:#38bdf8; }
  .tag-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
  .tag { padding:6px 14px; border-radius:20px; border:1px solid #475569; background:transparent; color:#94a3b8; cursor:pointer; font-size:0.85em; transition:all .2s; }
  .tag.active { background:#38bdf8; color:#0f172a; border-color:#38bdf8; }
  button { padding:12px 32px; border-radius:8px; border:none; background:#38bdf8; color:#0f172a; font-size:1em; font-weight:600; cursor:pointer; transition:all .2s; }
  button:hover { background:#7dd3fc; }
  button:disabled { opacity:0.4; cursor:not-allowed; }
  .result { margin-top:16px; padding:16px; border-radius:8px; font-size:0.9em; display:none; }
  .result.success { display:block; background:#064e3b; border:1px solid#059669; color:#6ee7b7; }
  .result.error { display:block; background:#7f1d1d; border:1px solid#dc2626; color:#fca5a5; }
  .spinner { display:inline-block; width:16px; height:16px; border:2px solid #0f172a; border-top:2px solid transparent; border-radius:50%; animation:spin .6s linear infinite; vertical-align:middle; margin-right:8px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#38bdf8; }
</style>
</head>
<body>
<div class="container">
  <h1>Travel AI Agent</h1>
  <p class="subtitle">DeepSeek · ReAct · CoT · ToT · MCTS · Reflexion</p>

  <div class="card">
    <h2>生成旅行规划</h2>
    <form id="plan-form">
      <div class="row">
        <div class="field">
          <label>目的地</label>
          <input name="destination" placeholder="如：三亚、成都、丽江" required>
        </div>
        <div class="field">
          <label>出发地</label>
          <input name="departure_from" value="北京">
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label>出发日期</label>
          <input name="start_date" type="date" id="start_date" required>
        </div>
        <div class="field">
          <label>结束日期</label>
          <input name="end_date" type="date" id="end_date" required>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label>出行方式</label>
          <select name="travel_mode">
            <option value="train">高铁</option>
            <option value="plane">飞机</option>
            <option value="car">自驾</option>
            <option value="mixed">混合</option>
          </select>
        </div>
        <div class="field">
          <label>人数</label>
          <select name="num_travelers">
            <option value="1">1人</option>
            <option value="2" selected>2人</option>
            <option value="3">3人</option>
            <option value="4">4人</option>
            <option value="5">5人</option>
            <option value="6">6人</option>
            <option value="8">8人</option>
            <option value="10">10人</option>
          </select>
        </div>
        <div class="field">
          <label>预算下限 (元/人)</label>
          <select name="budget_min">
            <option value="500">500</option>
            <option value="1000">1000</option>
            <option value="2000" selected>2000</option>
            <option value="3000">3000</option>
            <option value="5000">5000</option>
            <option value="8000">8000</option>
            <option value="10000">10000</option>
          </select>
        </div>
        <div class="field">
          <label>预算上限 (元/人)</label>
          <select name="budget_max">
            <option value="1000">1000</option>
            <option value="2000">2000</option>
            <option value="3000">3000</option>
            <option value="5000" selected>5000</option>
            <option value="8000">8000</option>
            <option value="10000">10000</option>
            <option value="15000">15000</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label>推理策略</label>
          <select name="strategy">
            <option value="">自动选择</option>
            <option value="react">ReAct</option>
            <option value="cot">CoT 思维链</option>
            <option value="tot">ToT 思维树</option>
            <option value="mcts">MCTS 树搜索</option>
            <option value="reflexion">Reflexion</option>
            <option value="decompose">任务分解</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label>老人人数</label>
          <select name="num_elderly">
            <option value="0" selected>无</option>
            <option value="1">1人</option>
            <option value="2">2人</option>
            <option value="3">3人</option>
            <option value="4">4人</option>
          </select>
        </div>
        <div class="field">
          <label>小孩人数</label>
          <select name="num_children">
            <option value="0" selected>无</option>
            <option value="1">1人</option>
            <option value="2">2人</option>
            <option value="3">3人</option>
            <option value="4">4人</option>
          </select>
        </div>
      </div>
      <div style="margin-bottom:16px;">
        <label style="font-size:0.85em;color:#94a3b8;display:block;margin-bottom:8px;">兴趣标签</label>
        <div class="tag-row" id="tags">
          <span class="tag" data-v="自然风光">自然风光</span>
          <span class="tag" data-v="历史人文">历史人文</span>
          <span class="tag active" data-v="美食">美食</span>
          <span class="tag" data-v="购物">购物</span>
          <span class="tag" data-v="户外运动">户外运动</span>
          <span class="tag" data-v="亲子">亲子</span>
          <span class="tag" data-v="休闲度假">休闲度假</span>
          <span class="tag" data-v="摄影">摄影</span>
        </div>
      </div>
      <button type="submit" id="submit-btn">生成旅行规划</button>
    </form>
    <div class="result" id="result"></div>
  </div>

  <div style="text-align:center;font-size:0.85em;color:#475569;">
    API 文档: <a href="/docs">/docs</a> | 规划文件: <a href="javascript:void(0)" onclick="alert('文件保存在 outputs/ 目录下')">outputs/</a>
  </div>
</div>

<script>
  // Set min dates and defaults
  const today = new Date().toISOString().slice(0,10)
  const end = new Date(Date.now()+3*86400000).toISOString().slice(0,10)
  const sd = document.getElementById('start_date')
  const ed = document.getElementById('end_date')
  sd.min = today
  sd.value = today
  ed.min = today
  ed.value = end
  sd.onchange = () => { if (ed.value < sd.value) ed.value = sd.value }

  // Tag toggle
  document.querySelectorAll('#tags .tag').forEach(el => {
    el.onclick = () => el.classList.toggle('active')
  })

  document.getElementById('plan-form').onsubmit = async (e) => {
    e.preventDefault()
    const btn = document.getElementById('submit-btn')
    const result = document.getElementById('result')
    btn.disabled = true
    btn.innerHTML = '<span class="spinner"></span>生成中...'
    result.className = 'result'
    result.textContent = ''

    const fd = new FormData(e.target)
    const tags = [...document.querySelectorAll('#tags .tag.active')].map(t => t.dataset.v)
    const body = {
      destination: fd.get('destination'),
      departure_from: fd.get('departure_from'),
      start_date: fd.get('start_date'),
      end_date: fd.get('end_date'),
      travel_mode: fd.get('travel_mode'),
      num_travelers: parseInt(fd.get('num_travelers')),
      num_elderly: parseInt(fd.get('num_elderly')),
      num_children: parseInt(fd.get('num_children')),
      budget_min: parseInt(fd.get('budget_min')),
      budget_max: parseInt(fd.get('budget_max')),
      interests: tags
    }

    try {
      const resp = await fetch('/api/plan', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      })
      const data = await resp.json()
      if (resp.ok && data.status === 'ok') {
        result.className = 'result success'
        result.innerHTML = '已保存: <code>' + data.file + '</code>'
      } else {
        throw new Error(data.detail || '未知错误')
      }
    } catch(err) {
      result.className = 'result error'
      result.textContent = '错误: ' + err.message
    } finally {
      btn.disabled = false
      btn.textContent = '生成旅行规划'
    }
  }
</script>
</body>
</html>"""


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """通用对话接口，自动选择推理策略。"""
    try:
        result = await travel_agent.chat(
            message=req.message,
            session_id=req.session_id,
            strategy=req.strategy,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/plan")
async def plan_travel(req: TravelPlanRequest):
    """生成完整旅行规划，保存为本地 md 文件。"""
    import os
    from datetime import datetime

    try:
        result = await travel_agent.plan_travel(req)

        # Save to local outputs folder as markdown
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{req.destination}_{req.start_date}_{req.end_date}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result["full_response"])

        logger.info("Plan saved to %s", filepath)

        return {
            "status": "ok",
            "file": filepath,
            "session_id": result["session_id"],
        }
    except Exception as e:
        logger.exception("Plan endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话历史消息。"""
    messages = await memory_store.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.get("/api/preferences")
async def get_preferences():
    """获取用户偏好设置。"""
    prefs = await memory_store.get_all_preferences()
    return {"preferences": prefs}


@app.put("/api/preferences")
async def update_preference(pref: PreferenceUpdate):
    """更新用户偏好。"""
    await memory_store.set_preference(pref.key, pref.value)
    return {"status": "ok", "key": pref.key, "value": pref.value}


@app.delete("/api/preferences/{key}")
async def delete_preference(key: str):
    """删除用户偏好。"""
    await memory_store.delete_preference(key)
    return {"status": "ok", "key": key}


@app.get("/api/tools")
async def list_tools():
    """列出可用的 MCP 工具。"""
    tools = mcp_provider.list_tools()
    return {"tools": [t.model_dump() for t in tools]}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话接口，SSE 格式返回。"""
    import json, uuid

    async def generate():
        try:
            sid = req.session_id or uuid.uuid4().hex[:12]
            await memory_store.create_session(sid)
            await memory_store.add_message(sid, "user", req.message)

            from agent.llm import chat_stream as llm_stream
            messages = [
                {"role": "system", "content": "你是一个旅行规划助手。用中文回答，风格温暖专业。结合天气、人数、老人小孩等因素给出个性化建议。"},
                {"role": "user", "content": req.message},
            ]

            full = ""
            async for chunk in llm_stream(messages):
                full += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            await memory_store.add_message(sid, "assistant", full, metadata={"strategy": "stream"})
            yield f"data: {json.dumps({'session_id': sid, 'done': True})}\n\n"
        except Exception as e:
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
