"""RAG 知识库：文档加载 → 切分 → 本地向量化(fastembed) → FAISS 检索 → 关键词重排。

Embedding 方案：BAAI/bge-small-zh-v1.5（本地 ONNX，无需外部平台 key，支持中文）。
索引落盘到 data/faiss_index，由 scripts/build_kb.py 构建后供 agent 工具查询。
"""
import os
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(_DIR, "..", "data"))
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


class _FastembedEmbeddings(Embeddings):
    """把 fastembed 包成 LangChain Embeddings 接口（本地运行，零外部依赖）。

    必须继承 langchain_core.embeddings.Embeddings，否则 FAISS 会把它当普通函数调用。
    """

    _model = None  # 类级缓存，避免每次调用重复加载

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from fastembed import TextEmbedding

            cls._model = TextEmbedding(EMBED_MODEL)
        return cls._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._get_model().embed([text]))[0]


def load_project_docs(root: str, patterns: tuple[str, ...] = ("**/*.md",)) -> list[Document]:
    """加载项目文档（默认所有 .md），保留相对路径作为来源 metadata。"""
    import glob as _glob

    docs = []
    for pat in patterns:
        for fp in _glob.glob(os.path.join(root, pat), recursive=True):
            rel = os.path.relpath(fp, root)
            # 跳过构建/依赖目录
            if any(seg in rel for seg in (".git", "node_modules", "data/", ".idea")):
                continue
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": rel}))
    return docs


def split_docs(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 80) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ""],
    )
    return splitter.split_documents(docs)


def build_index(docs: list[Document]) -> int:
    """切分并向量化入库（FAISS 本地索引），返回 chunk 数。"""
    chunks = split_docs(docs)
    if not chunks:
        raise ValueError("没有可入库的文档内容")
    vs = FAISS.from_documents(chunks, _FastembedEmbeddings())
    os.makedirs(DATA_DIR, exist_ok=True)
    vs.save_local(INDEX_PATH)
    return len(chunks)


def _load_index() -> FAISS | None:
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        return None
    return FAISS.load_local(INDEX_PATH, _FastembedEmbeddings(), allow_dangerous_deserialization=True)


# ---- rerank：向量初筛 + 关键词命中重排（轻量方案，无需 rerank 模型）----

_CJK = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


def _keywords(query: str) -> set[str]:
    """极简分词：连续中文片段 + 英数单词。"""
    return set(_CJK.findall(query.lower()))


def _kw_score(query: str, text: str) -> float:
    """关键词覆盖率得分：查询词在文本中的命中比例。"""
    kws = _keywords(query)
    if not kws:
        return 0.0
    text_low = text.lower()
    hit = sum(1 for k in kws if k in text_low)
    return hit / len(kws)


def retrieve(query: str, k: int = 4, fetch_k: int = 12) -> list[dict]:
    """检索：FAISS 向量召回 fetch_k 条 → 关键词得分重排 → 取前 k 条。

    返回 [{"content", "source", "vector_score", "rerank_score"}]。
    """
    vs = _load_index()
    if vs is None:
        return []
    scored = vs.similarity_search_with_score(query, k=fetch_k)  # FAISS 距离，越小越近
    items = []
    for doc, dist in scored:
        vec_score = 1.0 / (1.0 + max(dist, 0))          # 距离归一为 0~1 相似度
        kw_score = _kw_score(query, doc.page_content)
        items.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "?"),
            "vector_score": round(vec_score, 4),
            "rerank_score": round(0.6 * vec_score + 0.4 * kw_score, 4),  # 加权重排
        })
    items.sort(key=lambda x: x["rerank_score"], reverse=True)
    return items[:k]


def search_docs(query: str, k: int = 4) -> str:
    """供 agent 调用的检索入口：返回拼好的上下文文本（带来源标注）。"""
    items = retrieve(query, k=k)
    if not items:
        return "知识库为空或未构建。请先运行 scripts/build_kb.py 构建索引。"
    blocks = [f"[{i+1}] 来源: {it['source']}\n{it['content']}" for i, it in enumerate(items)]
    return "\n\n".join(blocks)
