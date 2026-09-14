"""阶段①：ChatModel 基本调用、消息类型、流式输出。

运行：conda activate langchain && python tutorials/01_chat_basics.py
需要 OPENAI_API_KEY（第三方 OpenAI 兼容端点可设 OPENAI_API_BASE / OPENAI_MODEL）。
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()  # 读取项目根目录 .env

# ---- 1. 构建模型 ----
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    # 第三方兼容端点：base_url=os.getenv("OPENAI_API_BASE"),
)


def demo_invoke() -> None:
    """最基础的调用：字符串进，AIMessage 出。"""
    resp = llm.invoke("用一句话介绍 LangChain")
    assert isinstance(resp, AIMessage)
    print("[invoke]", resp.content)
    print("[metadata] model =", resp.response_metadata.get("model_name"))


def demo_messages() -> None:
    """多轮 = 消息列表；SystemMessage 设定行为。"""
    msgs = [
        SystemMessage(content="你是一个极简助手，每句回答不超过10个字。"),
        HumanMessage(content="什么是向量数据库？"),
        AIMessage(content="存向量的库。"),
        HumanMessage(content="再详细一点呢？"),
    ]
    resp = llm.invoke(msgs)
    print("[messages]", resp.content)


def demo_stream() -> None:
    """流式：逐段拿到 AIMessageChunk。"""
    print("[stream] ", end="", flush=True)
    for chunk in llm.stream("用三句话介绍 RAG"):
        print(chunk.content, end="", flush=True)
    print()


def demo_batch() -> None:
    """批量并发。"""
    answers = llm.batch(["1+1=?只回答数字", "2+2=?只回答数字"])
    print("[batch]", [a.content for a in answers])


if __name__ == "__main__":
    demo_invoke()
    demo_messages()
    demo_stream()
    demo_batch()
