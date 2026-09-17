"""LangGraph Agent 定义：create_agent + 真实工具集 + checkpointer + 审批门控。

审批机制（human-in-the-loop）：write_file / run_cmd 等敏感工具执行前调用
langgraph 的 interrupt() 暂停图，等待 /approvals 接口 resume（见 main.py）。
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
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
    """FastAPI lifespan 启动时调用：建立 aiosqlite 连接并构建 Agent 图。

    图结构（plan-and-execute，对标 codex 的先规划后执行）：
        START -> plan -> executor(create_agent 子图，一次执行一步) -> reflect --\
                          ^                                                       \
                          |_______________________________/  (还有待办步骤)
                                                                          -> END (全部完成)
    """
    global checkpointer, _agent
    if checkpointer is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        _conn = await aiosqlite.connect(_DB_PATH)
        checkpointer = AsyncSqliteSaver(_conn)
    if "_agent" not in globals():
        _agent = _build_outer_graph()
    return _agent


# ---- 外层图状态：消息 + 计划 + 当前步 ----
class PlanState(MessagesState):
    plan: list[dict]          # [{"step": str, "status": "pending|done|failed"}]
    final_reply: str          # executor 最后一次回复，作为最终答案


def _plan_node(state: PlanState) -> dict:
    """规划节点：LLM 拆解任务（平凡任务空计划直通 executor）。"""
    from .planner import make_plan, plan_to_text

    user_msg = state["messages"][-1].content if state["messages"] else ""
    plan = make_plan(str(user_msg))
    plan_dicts = [s.model_dump() for s in plan.steps]
    if plan_dicts:
        notice = SystemMessage(content=f"任务已拆解为计划，逐项执行：\n{plan_to_text(plan_dicts)}")
        return {"plan": plan_dicts, "messages": [notice]}
    return {"plan": []}


def _next_pending(plan: list[dict]) -> int | None:
    """返回第一个 pending 步骤的下标；无则 None。"""
    for i, s in enumerate(plan):
        if s.get("status") == "pending":
            return i
    return None


def _execute_node(state: PlanState, config: RunnableConfig) -> dict:
    """执行节点：把当前待办步骤（或整条消息，若平凡任务）交给 executor 子图。"""
    executor = get_executor()
    plan = state.get("plan") or []
    idx = _next_pending(plan)
    if idx is None:
        task = str(state["messages"][-1].content)
    else:
        task = f"执行计划第 {idx + 1} 步：{plan[idx]['step']}\n完成后只汇报这一步的结果。"

    result = executor.invoke({"messages": [("user", task)]}, config)
    reply = result["messages"][-1].content or ""

    new_plan = [dict(s) for s in plan]
    if idx is not None:
        # executor 跑完该步：结果里含拒绝标记视为失败，否则视为完成
        new_plan[idx]["status"] = "failed" if "[已拒绝]" in reply else "done"
    # 把 executor 回复作为 AIMessage 并入外层状态：/chat 提取与历史查询都依赖它
    from langchain_core.messages import AIMessage as _AIM

    return {"plan": new_plan, "final_reply": str(reply), "messages": [_AIM(content=reply)]}


def _reflect_node(state: PlanState) -> dict:
    """反思节点：追加下一步指令或汇总收尾。"""
    plan = state.get("plan") or []
    if _next_pending(plan) is not None:
        return {"messages": [SystemMessage(content="计划还有待办步骤，继续执行下一步。")]}
    return {"messages": [SystemMessage(
        content="计划已全部执行完，请基于以上各步结果给用户一个简短的最终总结。")]}


def _route_after_reflect(state: PlanState) -> str:
    """reflect 后路由：有待办 -> 继续执行；无 -> 结束。"""
    return "executor" if _next_pending(state.get("plan") or []) is not None else END


def _build_outer_graph() -> CompiledStateGraph:
    """组装外层 plan-and-execute 图（executor 为 create_agent 子图）。"""
    g = StateGraph(PlanState)
    g.add_node("plan", _plan_node)
    g.add_node("executor", _execute_node)
    g.add_node("reflect", _reflect_node)
    g.add_edge(START, "plan")
    # plan 后总是进 executor：平凡任务（空计划）由 executor 整体处理一次，
    # 有计划的任务由 executor 按当前待办步执行
    g.add_edge("plan", "executor")
    g.add_edge("executor", "reflect")
    g.add_conditional_edges("reflect", _route_after_reflect, ["executor", END])
    return g.compile(checkpointer=checkpointer)


def get_executor() -> CompiledStateGraph:
    """executor 子图（create_agent + 全部工具），惰性构建、进程级复用。"""
    global _executor
    if "_executor" not in globals():
        _executor = create_agent(
            model=_build_model(),
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )  # 子图不挂 checkpointer：外层图统一持久化
    return _executor


def get_agent() -> CompiledStateGraph:
    """取已初始化的 Agent 图（必须在 FastAPI startup 之后调用）。"""
    if "_agent" not in globals():
        raise RuntimeError("agent 未初始化：FastAPI lifespan 未执行 init_agent()")
    return _agent  # type: ignore[name-defined]
