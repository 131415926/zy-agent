"""阶段④：@tool 工具定义与模型 tool-calling。

运行：python tutorials/05_tools.py（需要 API key）
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


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
    return f"{city}：晴，26°C"


def demo_schema() -> None:
    """@tool 自动生成 name/description/schema。"""
    print("[schema]", add.name, add.description)
    print("[schema]", add.args_schema.model_json_schema())


def demo_bind_tools() -> None:
    """模型只产出调用请求，不执行 —— 手动执行并回传 ToolMessage。"""
    llm_with_tools = llm.bind_tools([add, multiply])
    messages = [{"role": "user", "content": "3加5等于几？"}]

    resp = llm_with_tools.invoke(messages)
    messages.append(resp)
    print("[tool_calls]", resp.tool_calls)

    for tc in resp.tool_calls:                       # 手动执行工具
        result = add.invoke(tc["args"]) if tc["name"] == "add" else multiply.invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    final = llm_with_tools.invoke(messages)          # 带着工具结果再问模型
    print("[final]", final.content)


def demo_parallel() -> None:
    """一次请求里模型可能并行发出多个工具调用。"""
    llm_with_tools = llm.bind_tools([get_weather])
    resp = llm_with_tools.invoke("北京和上海今天天气如何？")
    print("[parallel]", [(t["name"], t["args"]) for t in resp.tool_calls])


if __name__ == "__main__":
    demo_schema()
    demo_bind_tools()
    demo_parallel()
