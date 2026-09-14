"""阶段⑤：手写 LangGraph ReAct 循环 —— 理解 create_agent 的内部原理。

运行：python tutorials/07_langgraph_graph.py（需要 API key）
"""
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


@tool
def get_weather(city: str) -> str:
    """查询指定城市今天的天气（演示用假数据）。"""
    return f"{city}：晴，26°C，适合出行。"


tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)


def ask_llm(state: MessagesState):
    """模型节点：返回增量消息（框架用 add_messages reducer 追加合并）。"""
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}


def route(state: MessagesState) -> str:
    """条件边：模型要调工具 -> tools 节点；否则结束。"""
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


def main() -> None:
    g = StateGraph(MessagesState)
    g.add_node("ask_llm", ask_llm)
    g.add_node("tools", ToolNode(tools))  # prebuilt：批量执行 tool_calls 并回传 ToolMessage
    g.add_edge(START, "ask_llm")
    g.add_conditional_edges("ask_llm", route, ["tools", END])
    g.add_edge("tools", "ask_llm")  # 工具结果回到模型，形成 ReAct 循环
    graph = g.compile()

    question = "北京和上海今天天气怎么样？适合跑步吗？"
    print("=== stream_mode=updates：逐步观察状态变化 ===")
    for update in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for node, delta in update.items():
            print(f"[node={node}]", str(delta)[:120], "...\n" if len(str(delta)) > 120 else "")

    final = graph.invoke({"messages": [{"role": "user", "content": question}]})
    print("=== 最终回答 ===\n", final["messages"][-1].content)


if __name__ == "__main__":
    main()
