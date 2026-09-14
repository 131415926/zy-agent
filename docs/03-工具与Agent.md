# 阶段④：工具调用与 Agent

对应示例：`tutorials/05_tools.py`、`tutorials/06_create_agent.py`

## 1. 定义工具

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""     # docstring 就是给模型看的工具说明，必写
    return a + b

print(add.name, add.args_schema.model_json_schema())
```

要点：
- `@tool` 从函数签名 + 类型注解自动生成 schema
- docstring 决定模型"什么时候用它"，写得越清楚调用越准
- 参数类型用 Pydantic 可校验的简单类型（int/str/bool/List）

## 2. 模型决定调用（tool-calling）

```python
llm_with_tools = llm.bind_tools([add, multiply])
resp = llm_with_tools.invoke("3加5等于几？")
resp.tool_calls   # [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_xxx'}]
```

模型本身不会执行工具，只产出"调用请求"；执行由 Agent 循环完成，
结果用 `ToolMessage` 回传给模型。

## 3. create_agent —— 1.x 的标准 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",          # 或 ChatOpenAI 实例
    tools=[add, multiply, search_weather],
    system_prompt="你是计算助手，必须使用工具完成计算。",
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "(3+5)×12 是多少？"}]}
)
for m in result["messages"]:
    print(type(m).__name__, getattr(m, "tool_calls", "") or m.content)
```

`create_agent` 完整签名（1.4.0 实测）：
`model, tools, system_prompt, middleware, response_format, state_schema,
context_schema, checkpointer, store, interrupt_before, interrupt_after, debug`

- 返回值就是一个 **Compiled LangGraph 图**，所以 `.invoke / .stream` 用法
  与普通图完全一致
- **旧版 `AgentExecutor` / `initialize_agent` 已删除，网上旧教程请跳过**

## 4. 多轮记忆：checkpointer + thread_id

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(model, tools=[...], checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "user-42"}}
agent.invoke({"messages": [...]}, cfg)   # 第 1 轮
agent.invoke({"messages": [...]}, cfg)   # 第 2 轮自动带历史
```

同一 `thread_id` 的消息历史被自动保存/回放，无需手动拼历史列表。
这就是 agent-app 服务端实现多会话的方式。

## 练习

1. 写一个 `get_weather(city)` 假工具（返回固定值），让 agent 连续调用两个工具。
2. 用 `response_format` 让 agent 最终回答结构化为 `{"answer": str, "tools_used": list}`。
3. 加 checkpointer 后，验证 agent 能记住你第一轮说的名字。
