"""阶段④：create_agent —— 1.x 标准 Agent（含多轮记忆）。

运行：python tutorials/06_create_agent.py（需要 API key）
"""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


def demo_agent() -> None:
    """create_agent 返回 Compiled LangGraph 图，自动完成 工具调用->执行->回传 循环。"""
    agent = create_agent(
        model="openai:gpt-4o-mini",  # 也可传 ChatOpenAI 实例
        tools=[add, multiply],
        system_prompt="你是计算助手，必须用工具完成计算，最后给出简洁结果。",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "(3+5)×12 等于多少？"}]}
    )
    for m in result["messages"]:
        kind = type(m).__name__
        detail = getattr(m, "tool_calls", None) or m.content
        print(f"[{kind}]", detail)


def demo_memory() -> None:
    """checkpointer + thread_id 实现多轮会话记忆。"""
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="你是友好的助手。",
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "user-42"}}

    agent.invoke({"messages": [{"role": "user", "content": "我叫小明，最喜欢的数字是7"}]}, cfg)
    result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？最喜欢的数字是几？"}]}, cfg)
    print("[memory]", result["messages"][-1].content)

    # 换 thread_id = 新会话，不共享记忆
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        {"configurable": {"thread_id": "user-43"}},
    )
    print("[new-thread]", result2["messages"][-1].content)


if __name__ == "__main__":
    demo_agent()
    demo_memory()
