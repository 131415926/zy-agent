"""LangGraph Agent 定义：create_agent + 工具 + checkpointer（多轮会话记忆）。"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

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


TOOLS = [add, multiply, get_weather, get_time]

SYSTEM_PROMPT = (
    "你是一个乐于助人的中文助手。涉及计算必须使用工具；"
    "被问到天气、时间等信息时使用对应工具查询。回答保持简洁。"
)


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


# 进程级单例：agent 与 checkpointer（生产环境可换 SqliteSaver/RedisSaver）
checkpointer = InMemorySaver()


def get_agent() -> CompiledStateGraph:
    """构建（或复用）Agent 图。"""
    global _agent
    if "_agent" not in globals():
        _agent = create_agent(
            model=_build_model(),
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
    return _agent
