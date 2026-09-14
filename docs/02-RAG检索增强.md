# 阶段③：RAG 检索增强

对应示例：`tutorials/04_rag_pipeline.py`

## RAG 五步流水线

```
加载 Load -> 切分 Split -> 向量 Embed -> 检索 Retrieve -> 生成 Generate
```

## 1. 切分

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块目标长度
    chunk_overlap=50,    # 相邻块重叠，避免语义被切断
)
chunks = splitter.split_documents(docs)   # docs: List[Document]
```

经验值：中文 300~600 字/块，overlap 取 10%~20%。

## 2. Embedding + 向量库

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")           # 持久化
vs2 = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

检索器（Retriever 是 Runnable，可直接接进 LCEL 链）：

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("什么是索引？")
```

## 3. 组装 RAG 链（LCEL + RunnablePassthrough）

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "仅根据以下上下文回答，不知道就说不知道：\n{context}"),
    ("human", "{question}"),
])

def fmt(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | fmt, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("什么是索引？"))
```

注意 dict 输入的写法：`retriever | fmt` 把 Document 列表拼成字符串，
`RunnablePassthrough()` 把原始问题透传。

## 4. 学习重点

- `Document(page_content=..., metadata={...})` 是全框架统一的文档结构
- metadata 保留来源（文件名/页码），回答时可以引用出处
- 本教程用 FAISS（本地、零依赖）；生产可选 Milvus/pgvector/Elasticsearch
- Embedding 模型同样支持第三方 OpenAI 兼容端点

## 练习

1. 找一篇本地 txt/md，跑通完整 RAG 链，问 3 个只有文档里才有答案的问题。
2. 调整 chunk_size（100 vs 500），对比回答质量。
3. 让回答附上来源：把 `fmt` 改成带 `[来源: metadata]` 的格式。
