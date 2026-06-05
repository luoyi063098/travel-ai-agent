# Travel AI Agent

基于 DeepSeek 大模型的智能旅行规划 Agent，支持多推理策略（ReAct/CoT/ToT/MCTS/Reflexion/任务分解），集成在线天气 MCP 服务，具备记忆系统，可根据天气、距离、出行方式、人数、老人/小孩等因素动态调整行程。

## 特性

- **多推理策略**：自动选择或手动指定 ReAct、CoT、ToT、MCTS、Reflexion、任务分解
- **在线天气**：通过 MCP 协议集成 Open-Meteo 免费天气 API（无需 API Key）
- **记忆系统**：SQLite 持久化会话历史和用户偏好
- **动态调整**：根据天气、人数、老人/小孩、出行方式、预算自动优化行程
- **全流程覆盖**：目的地介绍 → 行程规划 → 美食推荐 → 住宿建议 → 交通方案 → 动态调整

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt --break-system-packages
```

### 2. 配置 API Key

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

或创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### 3. 启动服务

```bash
python main.py
```

服务运行在 `http://localhost:8000`

## API 接口

### 对话接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我想去三亚玩3天，2个大人1个小孩，自驾，有什么推荐？"
  }'
```

响应：
```json
{
  "session_id": "a1b2c3d4e5f6",
  "response": "三亚三日游规划...",
  "strategy_used": "decompose"
}
```

### 旅行规划接口

```bash
curl -X POST http://localhost:8000/api/plan \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "三亚",
    "start_date": "2026-06-10",
    "end_date": "2026-06-12",
    "departure_from": "北京",
    "travel_mode": "plane",
    "num_travelers": 3,
    "has_elderly": false,
    "has_children": true,
    "budget": "medium",
    "interests": ["海滩", "美食", "亲子"]
  }'
```

### 用户偏好管理

```bash
# 设置偏好
curl -X PUT http://localhost:8000/api/preferences \
  -H "Content-Type: application/json" \
  -d '{"key": "preferred_travel_mode", "value": "train"}'

# 获取所有偏好
curl http://localhost:8000/api/preferences
```

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/sessions/{id}` | 获取会话历史 |
| GET | `/api/tools` | 列出 MCP 工具 |
| GET | `/api/health` | 健康检查 |

## 推理策略

| 策略 | 适用场景 | 特点 |
|------|---------|------|
| **ReAct** | 需要工具调用的任务 | 思考-行动-观察循环 |
| **CoT** | 需要逐步分析的任务 | 思维链推理 |
| **ToT** | 多方案对比选择 | 树形搜索+评估 |
| **MCTS** | 路线/方案优化 | 蒙特卡洛树搜索 |
| **Reflexion** | 改进已有方案 | 自我评估+反思+修正 |
| **Decompose** | 复杂综合任务 | 任务分解+依赖排序 |

策略可自动选择，也可通过 `strategy` 参数手动指定。

## 动态调整因子

| 因子 | 调整内容 |
|------|---------|
| 天气 | 雨天改室内，高温调整时段 |
| 距离 | 计算交通时间，建议出行方式 |
| 出行方式 | 自驾关注停车，高铁关注接驳 |
| 人数 | 多人需预订，设置集合点 |
| 老人 | 放缓节奏，避免登山长步行 |
| 小孩 | 亲子景点，预留午休 |

## 项目结构

```
localAgent/
├── main.py                 # FastAPI 入口
├── config.py               # 配置管理
├── requirements.txt        # 依赖
├── agent/
│   ├── core.py             # Agent 核心：策略选择、编排
│   ├── llm.py              # DeepSeek LLM 客户端
│   ├── reasoning/          # 推理引擎
│   │   ├── react.py        # ReAct 推理+行动
│   │   ├── cot.py          # Chain of Thought
│   │   ├── tot.py          # Tree of Thoughts
│   │   ├── mcts.py         # Monte Carlo Tree Search
│   │   ├── reflexion.py    # Reflexion 自改进
│   │   └── decompose.py    # 任务分解
│   ├── memory/             # 记忆系统 (SQLite)
│   ├── mcp/                # MCP 工具协议
│   │   ├── provider.py     # MCP 工具注册/调用
│   │   └── weather.py      # 天气工具 (Open-Meteo)
│   └── planner/            # 旅行规划模块
│       ├── destination.py  # 目的地介绍
│       ├── itinerary.py    # 行程生成
│       ├── food_stay.py    # 美食住宿推荐
│       ├── transport.py    # 交通规划
│       └── adjuster.py     # 动态调整
└── models/
    └── schemas.py          # Pydantic 数据模型
```

## 技术栈

- **Web 框架**: FastAPI
- **LLM**: DeepSeek Chat API
- **数据库**: SQLite (aiosqlite 异步访问)
- **MCP**: 自实现 MCP 协议封装
- **天气数据**: Open-Meteo (免费，无需 API Key)
