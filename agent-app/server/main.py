"""FastAPI 服务端：/chat 同步、/chat/stream SSE 流式、/sessions 会话管理。

启动：conda activate langchain && cd agent-app && python -m server.main
无 API key 链路验证：DRY_RUN=1 python -m server.main
"""
import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from .agent import get_agent

app = FastAPI(title="LangGraph Agent Server", version="1.1.0")

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


def _pending_approval(session_id: str) -> Optional[Approval]:
    """检查该会话图是否因 interrupt 暂停（等待审批）。"""
    agent = get_agent()
    state = agent.get_state(_cfg(session_id))
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
def chat(req: ChatRequest) -> ChatResponse:
    """同步聊天：一次返回完整回复。"""
    if not req.message.strip():
        raise HTTPException(400, "message 不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:8]
    agent = get_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        _cfg(session_id),
    )
    _sessions[session_id] = time.time()
    resp = _extract(result, session_id)
    resp.pending_approval = _pending_approval(session_id)
    return resp


@app.post("/approvals", response_model=ChatResponse)
def approvals(req: ApprovalRequest) -> ChatResponse:
    """审批接口：approve/reject 后恢复被 interrupt 暂停的图执行。"""
    if req.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision 只能为 approve 或 reject")
    agent = get_agent()
    cfg = _cfg(req.session_id)
    if _pending_approval(req.session_id) is None:
        raise HTTPException(404, "该会话没有待审批的操作")
    result = agent.invoke(Command(resume=req.decision), cfg)
    _sessions[req.session_id] = time.time()
    resp = _extract(result, req.session_id)
    resp.pending_approval = _pending_approval(req.session_id)  # 可能还有下一个待审批
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

        async def sse(event: str, data: str) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # compiled graph 原生支持 astream（sync 节点会自动在线程池执行）
            async for chunk in agent.astream(
                {"messages": [{"role": "user", "content": req.message}]},
                _cfg(session_id),
                stream_mode="messages",
            ):
                # stream_mode="messages" 产出 (message_chunk, metadata) 元组
                if isinstance(chunk, tuple):
                    chunk = chunk[0]
                kind = type(chunk).__name__
                if kind in ("AIMessageChunk", "AIMessage"):
                    if getattr(chunk, "tool_calls", None):
                        for t in chunk.tool_calls:
                            yield await sse("tool", f"{t['name']}({t['args']})")
                    if chunk.content:
                        yield await sse("message", chunk.content)
            # 图结束后检查是否有待审批操作（interrupt 发生在工具节点内）
            pa = _pending_approval(session_id)
            if pa:
                yield await sse("approval", json.dumps(pa.model_dump(), ensure_ascii=False))
            yield await sse("done", session_id)
        except Exception as e:  # noqa: BLE001 —— 统一转成 error 事件下发
            yield await sse("error", str(e))

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/sessions")
def sessions() -> dict:
    return {"sessions": sorted(_sessions, key=_sessions.get, reverse=True)}  # type: ignore[arg-type]


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str, limit: int = 20) -> dict:
    """查询某会话的消息历史（来自 checkpointer 持久化的图状态）。"""
    if session_id not in _sessions:
        raise HTTPException(404, "会话不存在")
    agent = get_agent()
    state = agent.get_state(_cfg(session_id))
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


if __name__ == "__main__":
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
