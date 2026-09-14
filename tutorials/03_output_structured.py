"""阶段②：结构化输出 with_structured_output（Pydantic schema）。

运行：python tutorials/03_output_structured.py（需要 API key）
"""
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ---- 定义输出 schema：字段注释即给模型看的说明 ----
class BookReview(BaseModel):
    title: str = Field(description="书名")
    score: int = Field(description="评分 0-10")
    summary: str = Field(description="一句话总结")


class Sentiment(BaseModel):
    sentiment: str = Field(description="positive / negative / neutral")
    confidence: float = Field(description="置信度 0~1")


def demo_review() -> None:
    structured = llm.with_structured_output(BookReview)
    review = structured.invoke("评价一下刘慈欣的《三体》")
    print("[review]", type(review).__name__, "->", review.model_dump())


def demo_sentiment() -> None:
    """同一模型可绑定不同 schema。"""
    structured = llm.with_structured_output(Sentiment)
    for text in ["这服务太差劲了", "今天下了一场雨"]:
        s = structured.invoke(text)
        print("[sentiment]", s.sentiment, s.confidence, "<-", text)


def demo_list_output() -> None:
    """嵌套结构：列表字段。"""

    class Extract(BaseModel):
        names: list[str] = Field(description="文中出现的所有人名")

    structured = llm.with_structured_output(Extract)
    r = structured.invoke("张三和李四约王五吃饭，赵六买单。")
    print("[extract]", r.names)


if __name__ == "__main__":
    demo_review()
    demo_sentiment()
    demo_list_output()
