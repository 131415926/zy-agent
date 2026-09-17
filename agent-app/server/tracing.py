"""可观测性：每个会话的执行 trace——LLM 调用/工具执行的耗时、token 用量、事件序列。

实现：LangGraph 流式回调（服务端 main.py 在流循环里打点）+ 本模块的内存环形存储。
- 一个 session 一个 Trace；一次请求一个 Run；Run 内记录事件（llm/tool/approval/plan）
- token 统计优先取 response_metadata 的 usage；缺失时按字符数估算（~4字符/token）
"""
import threading
import time
from collections import deque
from typing import Any

_LOCK = threading.Lock()
# session_id -> deque[run]，每会话最多保留 20 个 run
_TRACES: dict[str, deque] = {}
_MAX_RUNS = 20


def _now() -> float:
    return time.time()


def _est_tokens(text: str) -> int:
    """粗略估算：中文≈1.5字符/token，英文≈4字符/token，取折中 2.5。"""
    return max(1, round(len(text or "") / 2.5))


class Run:
    """一次请求的执行记录。"""

    def __init__(self, session_id: str, message: str):
        self.session_id = session_id
        self.message = message[:200]
        self.t_start = _now()
        self.t_end: float | None = None
        self.events: list[dict] = []          # {t, type, name?, detail?, ms?}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": False}
        self.reply_preview = ""

    # ---- 事件打点 ----

    def event(self, etype: str, name: str = "", detail: str = "", ms: float | None = None) -> None:
        with _LOCK:
            self.events.append({
                "t": round(_now() - self.t_start, 3),
                "type": etype,
                "name": name,
                "detail": str(detail)[:160],
                **({"ms": round(ms, 1)} if ms is not None else {}),
            })

    # ---- token 统计 ----

    def add_usage(self, usage_metadata: dict | None) -> None:
        if not usage_metadata:
            return
        with _LOCK:
            self.usage["prompt_tokens"] += int(usage_metadata.get("input_tokens", 0))
            self.usage["completion_tokens"] += int(usage_metadata.get("output_tokens", 0))

    def estimate_from_reply(self, reply: str) -> None:
        """回复无 usage 时兜底估算。"""
        with _LOCK:
            if self.usage["prompt_tokens"] == 0 and self.usage["completion_tokens"] == 0:
                self.usage["estimated"] = True
                self.usage["completion_tokens"] = _est_tokens(reply)
                self.usage["prompt_tokens"] = _est_tokens(self.message)

    def finish(self, reply: str) -> None:
        self.t_end = _now()
        self.reply_preview = reply[:300]
        self.estimate_from_reply(reply)

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        total_ms = round(((self.t_end or _now()) - self.t_start) * 1000, 1)
        return {
            "session_id": self.session_id,
            "message": self.message,
            "start": self.t_start,
            "total_ms": total_ms,
            "usage": self.usage,
            "reply_preview": self.reply_preview,
            "events": self.events,
        }


def start_run(session_id: str, message: str) -> Run:
    run = Run(session_id, message)
    with _LOCK:
        _TRACES.setdefault(session_id, deque(maxlen=_MAX_RUNS)).append(run)
    return run


def get_traces(session_id: str, limit: int = 5) -> list[dict]:
    with _LOCK:
        runs = list(_TRACES.get(session_id, []))
    return [r.to_dict() for r in runs[-limit:]]


def summarize(session_id: str) -> dict:
    """会话级汇总：总请求/总耗时/总token/事件类型分布。"""
    with _LOCK:
        runs = list(_TRACES.get(session_id, []))
    total_ms = sum(r.to_dict()["total_ms"] for r in runs)
    ptok = sum(r.usage["prompt_tokens"] for r in runs)
    ctok = sum(r.usage["completion_tokens"] for r in runs)
    kinds: dict[str, int] = {}
    for r in runs:
        for e in r.events:
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    return {
        "session_id": session_id,
        "runs": len(runs),
        "total_ms": round(total_ms, 1),
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "events": kinds,
    }
