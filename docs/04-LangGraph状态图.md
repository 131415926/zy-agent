# 阶段⑤：LangGraph 状态图

对应示例：`tutorials/07_langgraph_graph.py`

## 核心心智模型

Agent = 一张**状态图**：节点（函数）处理共享状态，边决定下一步走哪。
`create_agent` 只是一张预制图；当你需要**自定义流程**（多步审批、
并行分支、循环反思、人在回路）时，就要自己画图。

## 1. 状态定义

```python
from typing import Annotated
from langgraph.graph import MessagesState, StateGraph, START, END

# MessagesState 预置了 messages 字段，且自动做"消息追加合并"
# 自定义状态：
class State(MessagesState):
    query: str
    result: str
```

关键：`messages: Annotated[list, add_messages]` —— reducer 决定状态如何合并
（追加而非覆盖）。自定义列表字段想追加也要配 `operator.add` 等 reducer。

## 2. 节点与边

```python
def ask_llm(state: State):
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}          # 返回增量，框架负责合并

def route(state: State):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END   # 条件边：返回下一节点名

g = StateGraph(State)
g.add_node("ask_llm", ask_llm)
g.add_node("tools", ToolNode(tools))     # langgraph.prebuilt 的工具执行节点
g.add_edge(START, "ask_llm")
g.add_conditional_edges("ask_llm", route, ["tools", END])
g.add_edge("tools", "ask_llm")           # 工具结果回到模型，形成循环
graph = g.compile()
```

这就是 ReAct Agent 的手写版 —— 和 `create_agent` 内部等价。

## 3. 必懂概念速查

| 概念 | 说明 |
|---|---|
| `StateGraph(State)` | 以 State 为 schema 建图 |
| `add_node(name, fn)` | 节点：收 state，返回**增量** dict |
| `add_edge(a, b)` | 固定边 |
| `add_conditional_edges(a, fn, [...])` | fn 返回目标节点名 |
| `compile(checkpointer=...)` | 编译；checkpointer 提供持久化 |
| `interrupt_before=["tools"]` | 人在回路：执行工具前暂停等确认 |
| `graph.stream(input, stream_mode="updates")` | 逐节点流式观察状态变化 |

## 4. 子图与多 Agent

- 图可以当节点嵌进另一张图（子图），是"多 Agent 协作"的基本手法
- 常见模式：supervisor（调度节点按条件边路由到各专家子图）、
  plan-and-execute（规划节点 -> 执行循环 -> 反思节点）

## 练习

1. 手写 ReAct 循环图，对照 `tutorials/06` 的 `create_agent` 行为。
2. 在工具节点前加 `interrupt_before`，跑一次人在回路确认。
3. 用 `stream_mode="updates"` 观察每步状态增量。
