"""阶段②：ChatPromptTemplate 与 LCEL 管道。

运行：python tutorials/02_prompt_templates.py（需要 API key）
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def demo_basic_template() -> None:
    """变量模板 + LCEL：prompt | llm | parser。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是{domain}领域的专家，回答保持{style}。"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()  # parser 取出纯字符串

    resp = chain.invoke({
        "domain": "数据库",
        "style": "简短",
        "question": "什么是索引？",
    })
    print("[template]", resp)


def demo_few_shot() -> None:
    """Few-shot：给几个示例让模型模仿格式。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "把中文翻译成英文，只输出译文。"),
        ("human", "你好"),
        ("ai", "Hello"),
        ("human", "谢谢"),
        ("ai", "Thank you"),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    print("[few-shot]", chain.invoke({"text": "今天天气很好"}))


def demo_chat_history() -> None:
    """MessagesPlaceholder：给历史消息留插槽（LangGraph 之前的手工多轮方式）。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是简洁的助手。"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    resp = chain.invoke({
        "history": [("human", "我叫小明"), ("ai", "你好小明")],
        "question": "我叫什么名字？",
    })
    print("[history]", resp)


if __name__ == "__main__":
    demo_basic_template()
    demo_few_shot()
    demo_chat_history()
