"""Agent Server 的 Python SDK：封装 HTTP 调用，普通/流式两种聊天方式。

用法：
    from client.sdk import AgentClient
    c = AgentClient("http://127.0.0.1:8000")
    print(c.chat("s1", "你好"))
    for event, data in c.chat_stream("s1", "你好"):
        ...
"""
import json
from typing import Iterator, Optional

import httpx


class AgentError(Exception):
    """服务端返回错误时抛出。"""


class AgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    # ---- 基础接口 ----

    def health(self) -> dict:
        return self._get("/health")

    def sessions(self) -> list[str]:
        return self._get("/sessions")["sessions"]

    def delete_session(self, session_id: str) -> dict:
        resp = self._http.delete(f"{self.base_url}/sessions/{session_id}")
        return self._check(resp).json()

    # ---- 聊天 ----

    def chat(self, session_id: Optional[str], message: str) -> dict:
        """同步聊天：返回 {"session_id", "reply", "tool_calls"}。"""
        resp = self._http.post(
            f"{self.base_url}/chat",
            json={"session_id": session_id, "message": message},
        )
        return self._check(resp).json()

    def chat_stream(self, session_id: Optional[str], message: str) -> Iterator[tuple[str, str]]:
        """流式聊天：逐个产出 (event, data)。

        event ∈ {"message": 文本增量, "tool": 工具调用提示, "done": 会话id, "error": 错误}
        """
        payload = {"session_id": session_id, "message": message}
        with self._http.stream("POST", f"{self.base_url}/chat/stream", json=payload) as resp:
            if resp.status_code != 200:
                raise AgentError(f"HTTP {resp.status_code}: {resp.read().decode()}")
            event, data = "", ""
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield event, data

    # ---- 内部 ----

    def _get(self, path: str) -> dict:
        return self._check(self._http.get(f"{self.base_url}{path}")).json()

    @staticmethod
    def _check(resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            raise AgentError(f"HTTP {resp.status_code}: {resp.text}")
        return resp

    def close(self) -> None:
        self._http.close()
