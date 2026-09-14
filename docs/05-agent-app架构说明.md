# 阶段⑥：agent-app 项目架构说明

综合实战：一个完整的 Agent 应用 = **服务端（LangGraph Agent API）+ Python SDK 客户端 + CLI 交互聊天**。

## 架构

```
┌──────────┐   HTTP/SSE    ┌───────────────────┐
│ CLI (typer)│ ───────────▶ │  FastAPI server    │
└──────────┘               │  · /chat          │
┌──────────┐               │  · /chat/stream   │
│ SDK client│ ───────────▶ │  · /sessions      │
└──────────┘               │  · LangGraph agent│
                           │  · checkpointer   │
                           └───────────────────┘
```

- **服务端** `agent-app/server/`：LangGraph agent（`create_agent` + 工具 + checkpointer），
  对外暴露普通 JSON 和 SSE 流式两类聊天接口，按 `session_id`（=thread_id）隔离多轮会话。
- **客户端** `agent-app/client/`：`sdk.py` 封装 httpx 调用（普通/流式），
  `cli.py` 用 typer + rich 做交互式终端聊天（流式渲染 Markdown）。
- 服务端与客户端是**独立进程、纯 HTTP 通信**——这是生产 Agent 应用的标准形态。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 同步聊天，返回完整回复 |
| POST | `/chat/stream` | SSE 流式返回 token |
| GET | `/sessions` | 列出活跃会话 |
| DELETE | `/sessions/{id}` | 清除某会话记忆 |
| GET | `/health` | 健康检查 |

统一请求体：`{"session_id": str, "message": str}`；
统一响应体：`{"session_id": str, "reply": str, "tool_calls": [...]}`。

## 运行

```bash
conda activate langchain
cd agent-app

# 1) 配置模型（server/.env）
#    OPENAI_API_KEY=sk-xxx
#    OPENAI_API_BASE=https://...   # 可选，第三方 OpenAI 兼容端点
#    OPENAI_MODEL=gpt-4o-mini

# 2) 启动服务端
python -m server.main            # 默认 127.0.0.1:8000

# 3) 另开终端跑 CLI
python -m client.cli chat                       # 交互聊天
python -m client.cli chat --once "你好"          # 单次提问
python -m client.cli chat --session demo1        # 指定会话
python -m client.cli sessions                    # 查看会话
```

无 API key 时：`python -m server.main --dry-run` 用假模型回显，
用于验证服务端-客户端-CLI 全链路。

## 设计要点（对应文档 03/04）

1. 会话记忆不存客户端，全靠服务端 checkpointer + thread_id —— 客户端只带 `session_id`。
2. 流式用 SSE（`text/event-stream`），SDK 里解析成迭代器，CLI 逐段渲染。
3. 工具层与图层分离：新增工具只改 `server/agent.py` 的 tools 列表。
4. 敏感操作类工具（如"执行命令"）可在图上加 `interrupt_before` 做人工确认，见文档 04。
