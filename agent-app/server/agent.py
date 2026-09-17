"""LangGraph Agent 定义：create_agent + 真实工具集 + checkpointer + 审批门控。

审批机制（human-in-the-loop）：write_file / run_cmd 等敏感工具执行前调用
langgraph 的 interrupt() 暂停图，等待 /approvals 接口 resume（见 main.py）。
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from .tools_system import SYSTEM_TOOLS, sandbox_banner

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ---------------------------------------------------------------------------
# 工具层：新增工具只需在这里定义并加入 TOOLS 列表
# ---------------------------------------------------------------------------


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


@tool
def get_weather(city: str) -> str:
    """查询指定城市今天的天气（演示用假数据）。"""
    return f"{city}：晴，26°C，适合出行。"


@tool
def get_time() -> str:
    """获取当前时间。"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_docs(query: str) -> str:
    """检索本地知识库（项目文档），返回最相关的片段及来源。回答项目结构、
    学习文档内容、项目约定等问题前，先用此工具检索。"""
    from .rag import search_docs as _search

    return _search(query)


# 审批包装：敏感工具执行前 interrupt 暂停，等待人工确认后 resume。
# UX 对齐 codex：只读操作自动放行；审批支持「本会话总是允许」（always 缓存）。
from langgraph.types import interrupt  # noqa: E402

from .tools_system import is_readonly_command  # noqa: E402

# 会话级「总是允许」缓存：thread_id -> {tool_name, ...}
_always_allowed: dict[str, set[str]] = {}


def _session_id() -> str:
    """在图执行上下文中取当前 thread_id（工具节点运行时由 LangGraph 提供）。"""
    try:
        from langgraph.config import get_config

        return (get_config() or {}).get("configurable", {}).get("thread_id", "")
    except Exception:
        return ""


def _with_approval(t: BaseTool) -> BaseTool:
    @tool(t.name, description=t.description, args_schema=t.args_schema)
    def guarded(**kwargs):
        sid = _session_id()
        allowed = _always_allowed.setdefault(sid, set())

        # 1) run_cmd 只读命令自动放行（codex 式：读不烦人）
        if t.name == "run_cmd" and is_readonly_command(kwargs.get("command", "")):
            return t.invoke(kwargs)

        # 2) 本会话已选「总是允许」：直接放行
        if t.name in allowed:
            return t.invoke(kwargs)

        # 3) 弹审批：decision ∈ approve(本次) / always(本会话总是) / reject
        decision = interrupt({
            "type": "approval",
            "tool": t.name,
            "args": kwargs,
            "question": f"是否允许执行 {t.name}？",
            "options": ["approve", "always", "reject"],
        })
        if decision == "always":
            allowed.add(t.name)
            return t.invoke(kwargs)
        if decision == "approve":
            return t.invoke(kwargs)
        return f"[已拒绝] 用户未批准 {t.name} 调用（args={kwargs}）。请勿再尝试该操作，改为询问用户。"

    return guarded


SENSITIVE = {"run_cmd", "write_file"}
TOOLS = [(_with_approval(t) if t.name in SENSITIVE else t) for t in SYSTEM_TOOLS] + [get_time, search_docs]

SYSTEM_PROMPT = f"""你是 zy-agent——一个运行在用户终端里的编程助手（对标 codex / claude code），通过工具帮用户完成真实任务。

工作目录沙箱：{sandbox_banner()}
所有文件路径一律使用相对该目录的路径。

核心准则：
1. 动手前先看：修改/执行前先用 list_dir / read_file / glob_files / grep_files 了解现状，不要凭空猜测文件内容。
2. 计划先行：接到非平凡任务，先用一两句话说明你的步骤，然后逐步执行，每步汇报结果。
3. 最小改动：只做用户要求的事，不顺手重构、不添加未经要求的功能。
4. 敏感操作（写文件、执行命令）会弹出审批，被拒绝时立即停止该操作并询问用户，不要绕过。
5. 如实汇报：命令失败就读错误、修根因，不假装成功；不确定就说不确定。
6. 回答用中文，保持简洁；代码/文件内容保持原格式。"""


def _build_model():
    """按 .env 的 LLM_PROVIDER 构建模型；dry-run 模式返回回显假模型。"""
    if os.getenv("DRY_RUN") == "1":
        return _EchoModel()
    from .providers import build_chat_model

    return build_chat_model()


class _EchoModel:
    """dry-run 假模型：不调外部 API，直接回显（用于链路验证）。"""

    def bind_tools(self, tools, **kwargs):
        return self

    def _echo(self, messages):
        from langchain_core.messages import AIMessage

        text = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        return AIMessage(content=f"[dry-run 回显] {text}")

    def invoke(self, messages, config=None, **kwargs):
        return self._echo(messages)

    async def ainvoke(self, messages, config=None, **kwargs):
        return self._echo(messages)

    def stream(self, messages, config=None, **kwargs):
        yield self._echo(messages)

    async def astream(self, messages, config=None, **kwargs):
        yield self._echo(messages)


# 进程级单例：agent 与 checkpointer
# 会话持久化：AsyncSqliteSaver 落盘到 data/checkpoints.db（同时支持同步/异步图调用）。
# 服务端已全链路使用 async 接口（ainvoke/astream/aget_state），同步方法仅在
# 非事件循环线程可用，因此统一走 a* 方法，避免 InvalidStateError。
import aiosqlite  # noqa: E402

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DB_PATH = os.path.abspath(os.path.join(_DATA_DIR, "checkpoints.db"))

checkpointer: AsyncSqliteSaver | None = None


async def init_agent() -> CompiledStateGraph:
    """FastAPI lifespan 启动时调用：建立 aiosqlite 连接并构建 Agent 图。"""
    global checkpointer, _agent
    if checkpointer is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        _conn = await aiosqlite.connect(_DB_PATH)
        checkpointer = AsyncSqliteSaver(_conn)
    if "_agent" not in globals():
        _agent = create_agent(
            model=_build_model(),
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
    return _agent


def get_agent() -> CompiledStateGraph:
    """取已初始化的 Agent 图（必须在 FastAPI startup 之后调用）。"""
    if "_agent" not in globals():
        raise RuntimeError("agent 未初始化：FastAPI lifespan 未执行 init_agent()")
    return _agent  # type: ignore[name-defined]
