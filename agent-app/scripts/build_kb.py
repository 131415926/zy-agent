"""知识库索引构建脚本：把项目文档向量化入 FAISS，供 agent 的 search_docs 工具查询。

用法：
    cd agent-app && python -m scripts.build_kb            # 默认索引 ../docs
    python -m scripts.build_kb --root .. --pattern "**/*.py"   # 自定义范围
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.rag import build_index, load_project_docs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 知识库索引")
    parser.add_argument("--root", default=os.path.join(".."), help="文档根目录（默认项目根）")
    parser.add_argument("--pattern", action="append", default=None,
                        help="glob 模式，可多次传（默认 **/*.md）")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    patterns = tuple(args.pattern) if args.pattern else ("**/*.md",)

    docs = load_project_docs(root, patterns)
    if not docs:
        print(f"在 {root} 下未找到匹配 {patterns} 的文档")
        sys.exit(1)
    n = build_index(docs)
    print(f"✓ 已索引 {len(docs)} 个文档，切分 {n} 个 chunk，写入 data/faiss_index/")


if __name__ == "__main__":
    main()
