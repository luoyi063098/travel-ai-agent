"""
FastAPI 服务器 —— 智能旅行 AI Agent 的 HTTP API 层。

API 端点一览：
  GET  /                  - 根路径，返回旅行规划前端的 HTML 页面
  POST /api/chat         - 通用对话接口（自动选择推理策略）
  POST /api/plan         - 完整旅行规划生成（异步执行 8 步流程，保存为本地 markdown 文件）
  GET  /api/sessions/{id} - 获取指定会话的历史消息
  GET  /api/preferences   - 获取所有用户偏好设置
  PUT  /api/preferences   - 更新或创建用户偏好
  DELETE /api/preferences/{key} - 删除指定偏好
  GET  /api/tools         - 列出所有可用的 MCP 工具
  POST /api/chat/stream   - 流式对话接口（基于 SSE 协议）
  GET  /api/health        - 健康检查端点
"""

from __future__ import annotations  # 启用类型注解的延迟求值，避免循环导入

import logging
import time      # 用于计算请求处理耗时
from contextlib import asynccontextmanager  # 异步上下文管理器，用于实现 FastAPI lifespan

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware    # CORS 中间件，允许跨域请求
from fastapi.responses import StreamingResponse, HTMLResponse  # 流式响应和 HTML 响应

from config import SERVER_HOST, SERVER_PORT                                   # 服务器配置（主机和端口）
from models.schemas import (                                                  # 请求 / 响应的 Pydantic 模型
    ChatRequest,
    ChatResponse,
    TravelPlanRequest,
    PreferenceUpdate,
)
from agent.core import travel_agent    # 全局 TravelAgent 单例
from agent.mcp.provider import mcp_provider  # MCP 工具提供器
from agent.memory import memory_store        # 记忆存储

logger = logging.getLogger("travel_agent.server")


# ────────────────────────────────────────────
# 应用生命周期管理（lifespan）
# FastAPI 的 lifespan 机制替代了旧的 startup / shutdown 事件
# 在应用启动时执行初始化，在关闭时执行清理
# ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化各组件，关闭时清理资源。"""
    logger.info("Starting Travel AI Agent server...")

    # 验证 DeepSeek API Key 是否已配置
    from config import DEEPSEEK_API_KEY
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set! Set it via env var or .env file.")
    else:
        logger.info("DeepSeek API Key: configured")

    # 初始化记忆存储（创建 SQLite 数据库表等）
    await memory_store.init()
    logger.info("Memory store initialized (DB: data/travel_agent.db)")

    # 记录当前可用的所有 MCP 工具名称
    logger.info("MCP tools: %s", [t.name for t in mcp_provider.list_tools()])

    # yield 之前是启动逻辑，yield 之后是关闭逻辑
    yield

    # 服务关闭时的清理工作
    logger.info("Server shutting down")


# ────────────────────────────────────────────
# 创建 FastAPI 应用实例
# ────────────────────────────────────────────
app = FastAPI(
    title="Travel AI Agent",
    description="基于 DeepSeek 的智能旅行规划 Agent，支持 ReAct/CoT/ToT/MCTS/Reflexion 多种推理策略",
    version="1.0.0",
    lifespan=lifespan,
)

# ────────────────────────────────────────────
# 添加 CORS 中间件
# 允许所有来源的跨域请求，方便前端开发和调试
# ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有来源（生产环境建议限制特定域名）
    allow_credentials=True,     # 允许携带 Cookie 等凭证
    allow_methods=["*"],        # 允许所有 HTTP 方法
    allow_headers=["*"],        # 允许所有请求头
)


# ────────────────────────────────────────────
# 自定义 HTTP 请求日志中间件
# 记录每个请求的方法、路径、状态码和耗时
# ────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有 HTTP 请求的方法、路径、响应状态码和处理耗时。"""
    start = time.time()                              # 记录请求开始时间
    response = await call_next(request)              # 继续处理请求，获取响应
    duration = time.time() - start                   # 计算处理耗时
    logger.info("%s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, duration)
    return response


# ────────────────────────────────────────────
# 根路径端点 —— 返回旅行规划前端 HTML 页面
# ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回完整的旅行规划前端页面（含 CSS 样式和 JavaScript 交互逻辑）。"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Travel AI Agent</title>
<style>
  /* =====================================================
     全局样式重置 —— 去除浏览器默认边距和内边距，使用 border-box 盒模型
     ===================================================== */
  * { margin:0; padding:0; box-sizing:border-box; }

  /* 页面主体 —— 深色渐变背景，使用系统字体栈保证跨平台一致性 */
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; min-height:100vh; }

  /* 居中容器 —— 限制最大宽度 720px，上下留白 40px */
  .container { max-width:720px; margin:0 auto; padding:40px 20px; }
  h1 { font-size:2em; margin-bottom:4px; }

  /* 副标题 —— 浅灰色，用于展示推理策略标签 */
  .subtitle { color:#64748b; margin-bottom:28px; font-size:0.9em; }

  /* 卡片组件 —— 深色背景，带边框和圆角，承载表单内容 */
  .card { background: #1e293b; border:1px solid #334155; border-radius:12px; padding:28px; margin-bottom:20px; }
  .card h2 { font-size:1.1em; color:#38bdf8; margin-bottom:20px; }

  /* 弹性行布局 —— 用于并排放置多个表单字段 */
  .row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }

  /* 表单字段容器 —— 弹性增长，最小宽度 200px，保证响应式换行 */
  .field { flex:1; min-width:200px; display:flex; flex-direction:column; }
  .field label { font-size:0.85em; color:#94a3b8; margin-bottom:6px; }
  .field input, .field select { padding:10px 12px; border-radius:8px; border:1px solid #475569; background:#0f172a; color:#e2e8f0; font-size:0.95em; outline:none; }
  .field input:focus, .field select:focus { border-color:#38bdf8; }
  .field input::placeholder { color:#475569; }

  /* 复选框行 —— 用于老人 / 小孩数量的选项 */
  .check-row { display:flex; gap:24px; align-items:center; margin-bottom:16px; }
  .check-row label { display:flex; align-items:center; gap:8px; font-size:0.9em; color:#cbd5e1; cursor:pointer; }
  .check-row input[type=checkbox] { width:18px; height:18px; accent-color:#38bdf8; }

  /* 兴趣标签行 —— 可点击切换的标签按钮组 */
  .tag-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }

  /* 单个标签样式 —— 胶囊形按钮，点击切换 active 状态 */
  .tag { padding:6px 14px; border-radius:20px; border:1px solid #475569; background:transparent; color:#94a3b8; cursor:pointer; font-size:0.85em; transition:all .2s; }
  .tag.active { background:#38bdf8; color:#0f172a; border-color:#38bdf8; }

  /* 提交按钮 —— 蓝色主色调，hover 时变亮，disabled 时半透明 */
  button { padding:12px 32px; border-radius:8px; border:none; background:#38bdf8; color:#0f172a; font-size:1em; font-weight:600; cursor:pointer; transition:all .2s; }
  button:hover { background:#7dd3fc; }
  button:disabled { opacity:0.4; cursor:not-allowed; }

  /* 结果提示框 —— 默认隐藏，成功显示绿色，失败显示红色 */
  .result { margin-top:16px; padding:16px; border-radius:8px; font-size:0.9em; display:none; }
  .result.success { display:block; background:#064e3b; border:1px solid#059669; color:#6ee7b7; }
  .result.error { display:block; background:#7f1d1d; border:1px solid#dc2626; color:#fca5a5; }

  /* 加载动画 —— 旋转的圆环，在提交按钮中显示等待状态 */
  .spinner { display:inline-block; width:16px; height:16px; border:2px solid #0f172a; border-top:2px solid transparent; border-radius:50%; animation:spin .6s linear infinite; vertical-align:middle; margin-right:8px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  a { color:#38bdf8; }
</style>
</head>
<body>
<div class="container">
  <h1>Travel AI Agent</h1>
  <p class="subtitle">DeepSeek · ReAct · CoT · ToT · MCTS · Reflexion</p>

  <!-- 旅行规划表单卡片 -->
  <div class="card">
    <h2>生成旅行规划</h2>
    <form id="plan-form">
      <!-- 第一行：目的地 + 出发地 -->
      <div class="row">
        <div class="field">
          <label>目的地</label>
          <input name="destination" placeholder="如：三亚、成都、丽江" required>
        </div>
        <div class="field">
          <label>出发地</label>
          <select name="departure_from">
            <option value="上海" selected>上海</option>
            <option value="北京">北京</option>
            <option value="广州">广州</option>
            <option value="深圳">深圳</option>
            <option value="成都">成都</option>
            <option value="杭州">杭州</option>
            <option value="南京">南京</option>
            <option value="武汉">武汉</option>
            <option value="重庆">重庆</option>
            <option value="西安">西安</option>
            <option value="长沙">长沙</option>
            <option value="厦门">厦门</option>
            <option value="青岛">青岛</option>
            <option value="苏州">苏州</option>
            <option value="天津">天津</option>
          </select>
        </div>
      </div>
      <!-- 第二行：出发日期 + 结束日期 -->
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
      <!-- 第三行：出行方式 + 人数 + 预算下限 + 预算上限 -->
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
      <!-- 第四行：推理策略选择 -->
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
      <!-- 第五行：老人 + 小孩人数 -->
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
      <!-- 兴趣标签区域：可点击切换，多个可以同时选中 -->
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
      <!-- 提交按钮 -->
      <button type="submit" id="submit-btn">生成旅行规划</button>
    </form>
    <!-- 结果显示区域（成功或失败信息） -->
    <div class="result" id="result"></div>
  </div>

  <!-- 页面底部链接 -->
  <div style="text-align:center;font-size:0.85em;color:#475569;">
    API 文档: <a href="/docs">/docs</a> | 规划文件: <a href="javascript:void(0)" onclick="alert('文件保存在 outputs/ 目录下')">outputs/</a>
  </div>
</div>

<script>
  /* ========================================
     设置日期输入框的最小值和默认值
     ======================================== */
  const today = new Date().toISOString().slice(0,10)           // 获取今天的日期字符串（YYYY-MM-DD）
  const end = new Date(Date.now()+3*86400000).toISOString().slice(0,10)  // 默认结束日期 = 今天 + 3 天
  const sd = document.getElementById('start_date')
  const ed = document.getElementById('end_date')
  sd.min = today     // 出发日期不能早于今天
  sd.value = today   // 默认出发日期 = 今天
  ed.min = today     // 结束日期不能早于今天
  ed.value = end     // 默认结束日期 = 今天 + 3 天
  sd.onchange = () => { if (ed.value < sd.value) ed.value = sd.value }  // 当出发日期改变时，确保结束日期 >= 出发日期

  /* ========================================
     兴趣标签点击切换 active 状态
     ======================================== */
  document.querySelectorAll('#tags .tag').forEach(el => {
    el.onclick = () => el.classList.toggle('active')
  })

  /* ========================================
     表单提交逻辑 —— 异步请求 /api/plan 端点
     ======================================== */
  document.getElementById('plan-form').onsubmit = async (e) => {
    e.preventDefault()                      // 阻止表单默认提交行为
    const btn = document.getElementById('submit-btn')
    const result = document.getElementById('result')
    btn.disabled = true                     // 禁用按钮，防止重复提交
    btn.innerHTML = '<span class="spinner"></span>生成中...'  // 显示加载动画
    result.className = 'result'             // 重置结果区域样式
    result.textContent = ''                 // 清空结果内容

    // 收集表单数据
    const fd = new FormData(e.target)
    // 收集所有被选中的兴趣标签（active 状态的 span）
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
      // 发送 POST 请求到 /api/plan 端点
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
      btn.disabled = false       // 恢复按钮状态
      btn.textContent = '生成旅行规划'  // 重置按钮文本
    }
  }
</script>
</body>
</html>"""


# ────────────────────────────────────────────
# POST /api/chat —— 通用对话接口
# 接收用户消息，自动选择推理策略并返回回答
# ────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """通用对话接口，自动选择推理策略。"""
    try:
        # 委托给 TravelAgent.chat 方法处理，传入消息、会话 ID 和策略
        result = await travel_agent.chat(
            message=req.message,
            session_id=req.session_id,
            strategy=req.strategy,
        )
        return ChatResponse(**result)  # 将结果字典转换为 Pydantic 响应模型
    except Exception as e:
        logger.exception("Chat endpoint error")                     # 记录完整异常栈
        raise HTTPException(status_code=500, detail=str(e))         # 返回 500 错误


# ────────────────────────────────────────────
# POST /api/plan —— 完整旅行规划生成
# 触发 8 步规划流程，结果保存为本地 markdown 文件
# ────────────────────────────────────────────
@app.post("/api/plan")
async def plan_travel(req: TravelPlanRequest):
    """生成完整旅行规划，保存为本地 md 文件。"""
    import os
    from datetime import datetime  # 生成时间戳用于文件名

    try:
        # 执行完整的旅行规划流程，获得结构化结果
        result = await travel_agent.plan_travel(req)

        # 将完整规划保存到本地 outputs 目录下的 markdown 文件
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # 文件名格式：目的地_开始日期_结束日期_时间戳.md
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
        logger.exception("Plan endpoint error")                     # 记录完整异常栈
        raise HTTPException(status_code=500, detail=str(e))         # 返回 500 错误


# ────────────────────────────────────────────
# GET /api/sessions/{session_id} —— 查询会话历史
# ────────────────────────────────────────────
@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定会话的所有历史消息。"""
    messages = await memory_store.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


# ────────────────────────────────────────────
# GET /api/preferences —— 获取所有用户偏好
# ────────────────────────────────────────────
@app.get("/api/preferences")
async def get_preferences():
    """获取所有用户偏好设置。"""
    prefs = await memory_store.get_all_preferences()
    return {"preferences": prefs}


# ────────────────────────────────────────────
# PUT /api/preferences —— 更新 / 创建用户偏好
# ────────────────────────────────────────────
@app.put("/api/preferences")
async def update_preference(pref: PreferenceUpdate):
    """更新用户偏好（如果不存在则创建）。"""
    await memory_store.set_preference(pref.key, pref.value)
    return {"status": "ok", "key": pref.key, "value": pref.value}


# ────────────────────────────────────────────
# DELETE /api/preferences/{key} —— 删除指定偏好
# ────────────────────────────────────────────
@app.delete("/api/preferences/{key}")
async def delete_preference(key: str):
    """删除指定键的用户偏好。"""
    await memory_store.delete_preference(key)
    return {"status": "ok", "key": key}


# ────────────────────────────────────────────
# GET /api/tools —— 列出所有可用的 MCP 工具
# ────────────────────────────────────────────
@app.get("/api/tools")
async def list_tools():
    """列出当前可用的所有 MCP 工具及其详细信息。"""
    tools = mcp_provider.list_tools()
    return {"tools": [t.model_dump() for t in tools]}  # 将工具对象序列化为字典


# ────────────────────────────────────────────
# POST /api/chat/stream —— 流式对话接口
# 使用 Server-Sent Events（SSE）协议逐块返回 LLM 生成内容
# ────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话接口，SSE 格式返回。"""
    import json, uuid

    # 定义异步生成器函数 —— 每次 yield 一个 SSE 数据块
    async def generate():
        try:
            # 如果没有会话 ID 则自动生成
            sid = req.session_id or uuid.uuid4().hex[:12]
            await memory_store.create_session(sid)
            await memory_store.add_message(sid, "user", req.message)

            # 构建记忆上下文（用户偏好 + 历史消息），注入 system prompt
            memory_context = await memory_store.build_context(sid, req.message)
            extra_lines = []
            for m in memory_context:
                if m["role"] == "system":
                    extra_lines.append(m["content"])
                elif m["role"] == "assistant":
                    extra_lines.append(f"[历史回复] {m['content'][:200]}")
            memory_ctx = "\n".join(extra_lines) if extra_lines else ""

            # 使用策略选择器确定推理策略名称，用于匹配对应的策略提示
            from agent.core import StrategySelector
            strategy_name = StrategySelector.select(req.message, req.strategy) if hasattr(StrategySelector, 'select') else "react"

            strategy_hints = {
                "react": "分析用户需求，主动查询天气和距离等工具来辅助回答。",
                "cot": "逐步分析用户问题的每个关键维度，最后给出综合建议。",
                "tot": "从多个角度对比分析，给出最优方案。",
                "decompose": "将用户需求拆解为子任务，逐一分析后综合回答。",
                "reflexion": "认真评估每个建议的可行性，主动指出潜在问题。",
                "mcts": "探索多个备选方案，选择最优解。",
            }
            hint = strategy_hints.get(strategy_name, "")

            # 在函数内部导入流式 LLM 接口，避免模块级别的循环依赖
            from agent.llm import chat_stream as llm_stream
            # 构建对话消息列表，合并记忆上下文和策略提示
            system_content = "你是一个旅行规划助手。用中文回答，风格温暖专业。结合天气、人数、老人小孩等因素给出个性化建议。" + (f"\n\n{memory_ctx}" if memory_ctx else "") + (f"\n\n策略提示: {hint}" if hint else "")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": req.message},
            ]

            full = ""
            # 逐块读取 LLM 的流式输出
            async for chunk in llm_stream(messages):
                full += chunk
                # 按照 SSE 格式发送数据块：以 "data: " 开头，以两个换行符结尾
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # 流式生成完成后，保存完整的助手回答到记忆存储
            await memory_store.add_message(sid, "assistant", full, metadata={"strategy": strategy_name})
            # 发送结束标记，包含会话 ID
            yield f"data: {json.dumps({'session_id': sid, 'done': True})}\n\n"
        except Exception as e:
            # 流式处理过程中发生异常，发送错误事件
            logger.exception("Stream error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ────────────────────────────────────────────
# GET /api/health —— 健康检查端点
# ────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """健康检查，返回服务运行状态。"""
    return {"status": "ok"}


# ────────────────────────────────────────────
# 主入口 —— 使用 uvicorn 启动服务器
# ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
