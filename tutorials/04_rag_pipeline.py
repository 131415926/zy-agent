"""阶段③：RAG 全流程 —— 切分、FAISS 向量检索、LCEL 组装。

运行：python tutorials/04_rag_pipeline.py（需要 API key：LLM + Embedding）
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small"))

# ---- 0. 模拟知识库文档（实战中用 PyPDFLoader/TextLoader 加载本地文件）----
DOCS = [
    "LangChain 是一个面向大模型应用开发的框架，提供模型调用、提示词管理、工具调用、检索增强等能力。",
    "LangGraph 是基于状态图编排 LLM 应用的库：节点处理共享状态，边决定流转，支持循环、分支和人工介入。",
    "FAISS 是 Facebook 开源的向量相似度检索库，常用于 RAG 场景的本地向量存储，支持持久化到磁盘。",
    "RAG（检索增强生成）先从知识库检索相关片段，再让模型基于片段回答，可缓解幻觉并支持引用来源。",
    "LangChain 1.x 中 create_agent 是标准 Agent 入口，底层由 LangGraph 图驱动，旧 AgentExecutor 已淘汰。",
]


def build_retriever():
    """加载 -> 切分 -> 向量化 -> 检索器。"""
    from langchain_core.documents import Document

    docs = [Document(page_content=t, metadata={"source": f"kb#{i}"}) for i, t in enumerate(DOCS)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    print(f"[split] {len(docs)} docs -> {len(chunks)} chunks")

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local("/tmp/faiss_demo_index")  # 持久化演示
    return vs.as_retriever(search_kwargs={"k": 2})


def main() -> None:
    retriever = build_retriever()
    print("[retrieve]", [d.metadata["source"] for d in retriever.invoke("什么是RAG")])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "仅根据以下上下文回答，不知道就直说不知道：\n{context}"),
        ("human", "{question}"),
    ])

    def fmt(docs):
        return "\n\n".join(f"{d.page_content} [{d.metadata['source']}]" for d in docs)

    rag_chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for q in ["LangGraph 和 LangChain 什么关系？", "公司年假有几天？"]:
        print(f"\n[Q] {q}")
        print("[A]", rag_chain.invoke(q))


if __name__ == "__main__":
    main()
