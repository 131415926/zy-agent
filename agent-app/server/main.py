"""FastAPI 服务端：/chat 同步、/chat/stream SSE 流式、/sessions 会话管理。

启动：conda activate langchain && cd agent-app && python -m server.main
无 API key 链路验证：DRY_RUN=1 python -m server.main
"""
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from .agent import get_agent, init_agent
from .tracing import get_traces, start_run, summarize


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时初始化 AsyncSqliteSaver 连接与 Agent 图。"""
    await init_agent()
    yield


app = FastAPI(title="LangGraph Agent Server", version="1.2.0", lifespan=lifespan)

# 活跃会话登记（thread_id -> 最近活跃时间戳）
_sessions: dict[str, float] = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class Approval(BaseModel):
    tool: str
    args: dict
    question: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict]
    pending_approval: Optional[Approval] = None  # 非空 = 图已暂停等审批


class ApprovalRequest(BaseModel):
    session_id: str
    decision: str  # approve | reject


def _cfg(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def _pending_approval(session_id: str) -> Optional[Approval]:
    """检查该会话图是否因 interrupt 暂停（等待审批）。"""
    agent = get_agent()
    state = await agent.aget_state(_cfg(session_id))
    if not state.next:
        return None
    for task in state.tasks:
        if task.interrupts:
            payload = task.interrupts[0].value
            return Approval(tool=payload["tool"], args=payload["args"], question=payload["question"])
    return None


def _extract(result: dict, session_id: str) -> ChatResponse:
    """从图状态中取最终回复与途经的工具调用。"""
    tool_calls: list[dict] = []
    reply = ""
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            tool_calls.extend({"name": t["name"], "args": t["args"]} for t in m.tool_calls)
        elif type(m).__name__ == "AIMessage" and m.content:
            reply = m.content
    return ChatResponse(session_id=session_id, reply=reply, tool_calls=tool_calls)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """同步聊天：一次返回完整回复。"""
    if not req.message.strip():
        raise HTTPException(400, "message 不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:8]
    agent = get_agent()
    run = start_run(session_id, req.message)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        _cfg(session_id),
    )
    _sessions[session_id] = time.time()
    resp = _extract(result, session_id)
    resp.pending_approval = await _pending_approval(session_id)
    run.finish(resp.reply)
    return resp


@app.post("/approvals", response_model=ChatResponse)
async def approvals(req: ApprovalRequest) -> ChatResponse:
    """审批接口：approve/reject 后恢复被 interrupt 暂停的图执行。"""
    if req.decision not in ("approve", "always", "reject"):
        raise HTTPException(400, "decision 只能为 approve / always / reject")
    agent = get_agent()
    cfg = _cfg(req.session_id)
    if await _pending_approval(req.session_id) is None:
        raise HTTPException(404, "该会话没有待审批的操作")
    result = await agent.ainvoke(Command(resume=req.decision), cfg)
    _sessions[req.session_id] = time.time()
    resp = _extract(result, req.session_id)
    resp.pending_approval = await _pending_approval(req.session_id)  # 可能还有下一个待审批
    return resp


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式聊天：逐 token 推送，事件类型 message / tool / done。"""
    if not req.message.strip():
        raise HTTPException(400, "message 不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:8]
    agent = get_agent()

    async def event_gen() -> AsyncIterator[str]:
        _sessions[session_id] = time.time()
        run = start_run(session_id, req.message)
        chunks: list[str] = []

        async def sse(event: str, data: str) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # 先告知 session_id：客户端在首个审批到来前就能拿到 sid
            yield await sse("start", session_id)
            # 双模式流：messages（token 级）+ updates（节点级，用于捕获 plan 产出）
            stream = agent.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                _cfg(session_id),
                stream_mode=["messages", "updates"],
                subgraphs=True,  # 透出 executor 子图内部的 token 流
            )
            async for chunk in stream:
                # subgraphs=True + 多模式时为三元组：(namespace, 模式名, 载荷)
                namespace = None
                if isinstance(chunk, tuple) and len(chunk) == 3 and isinstance(chunk[0], tuple):
                    namespace, chunk = chunk[0], chunk[1:]
                # 无子图时为二元组：(模式名, 载荷)
                if not (isinstance(chunk, tuple) and len(chunk) == 2):
                    continue
                mode, payload = chunk
                if mode == "updates":
                    if isinstance(payload, dict):
                        for node, delta in payload.items():
                            if node == "plan" and isinstance(delta, dict) and delta.get("plan"):
                                yield await sse("plan", json.dumps(delta["plan"], ensure_ascii=False))
                    continue
                if mode == "messages":
                    msg_chunk = payload[0] if isinstance(payload, tuple) else payload
                    meta = payload[1] if isinstance(payload, tuple) and len(payload) > 1 else {}
                    # 排除外层图 plan/reflect 节点的内部 LLM 输出（executor 子图内
                    # langgraph_node 为 model/tools 等，不在排除之列）
                    node_name = str(meta.get("langgraph_node", "")) if isinstance(meta, dict) else ""
                    if node_name in ("plan", "reflect"):
                        continue
                    kind = type(msg_chunk).__name__
                    if kind in ("AIMessageChunk", "AIMessage"):
                        # trace：token 用量（chunk 级 usage 汇总）
                        if getattr(msg_chunk, "usage_metadata", None):
                            run.add_usage(msg_chunk.usage_metadata)
                        if getattr(msg_chunk, "tool_calls", None):
                            for t in msg_chunk.tool_calls:
                                run.event("tool", t["name"], json.dumps(t["args"], ensure_ascii=False))
                                yield await sse("tool", f"{t['name']}({t['args']})")
                        if msg_chunk.content:
                            if not chunks:
                                run.event("llm", node_name or "model", "first token")
                            chunks.append(msg_chunk.content)
                            yield await sse("message", msg_chunk.content)
            # 图结束后检查是否有待审批操作（interrupt 发生在工具节点内）
            pa = await _pending_approval(session_id)
            if pa:
                run.event("approval", pa.tool, json.dumps(pa.args, ensure_ascii=False))
                yield await sse("approval", json.dumps(pa.model_dump(), ensure_ascii=False))
            run.finish("".join(chunks))
            yield await sse("done", session_id)
        except Exception as e:  # noqa: BLE001 —— 统一转成 error 事件下发
            run.event("error", detail=str(e))
            run.finish("".join(chunks))
            yield await sse("error", str(e))

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/sessions")
def sessions() -> dict:
    return {"sessions": sorted(_sessions, key=_sessions.get, reverse=True)}  # type: ignore[arg-type]


@app.get("/traces/{session_id}")
async def traces(session_id: str, limit: int = 5) -> dict:
    """会话执行 trace：每次请求的耗时/token/事件序列（最近 limit 个 run）。"""
    return {"session_id": session_id, "runs": get_traces(session_id, limit=limit)}


@app.get("/traces/{session_id}/summary")
async def traces_summary(session_id: str) -> dict:
    """会话级汇总：请求数/总耗时/总 token/事件分布。"""
    return summarize(session_id)


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str, limit: int = 20) -> dict:
    """查询某会话的消息历史（来自 checkpointer 持久化的图状态）。"""
    if session_id not in _sessions:
        raise HTTPException(404, "会话不存在")
    agent = get_agent()
    state = await agent.aget_state(_cfg(session_id))
    msgs = state.values.get("messages", [])[-limit * 2:]  # 一轮≈2条，粗略截取
    history = []
    for m in msgs:
        kind = type(m).__name__
        if kind in ("HumanMessage", "AIMessage", "ToolMessage"):
            history.append({"role": kind.removesuffix("Message").lower(), "content": str(m.content)})
    return {"session_id": session_id, "history": history}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """清除会话记忆（checkpointer 中该 thread 的状态随进程保留，这里仅撤销登记）。"""
    if session_id not in _sessions:
        raise HTTPException(404, "会话不存在")
    del _sessions[session_id]
    return {"deleted": session_id}


def main() -> None:
    """CLI 入口：python -m server.main 或根目录 run_server.py。"""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="不调真实模型，回显验证链路")
    args = parser.parse_args()
    if args.dry_run:
        os.environ["DRY_RUN"] = "1"
    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
