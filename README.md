# LangChain / LangGraph 学习仓库

环境：conda `langchain`（Python 3.12）· langchain 1.4.0 · langgraph 1.2.11

- 学习路线：见 [docs/00-学习路线总览.md](docs/00-学习路线总览.md)
- 分阶段文档：`docs/01`~`docs/04`（基础→RAG→工具Agent→LangGraph）
- 综合实战：`agent-app/`（FastAPI 服务端 + SDK 客户端 + CLI），见 [docs/05](docs/05-agent-app架构说明.md)

```bash
conda activate langchain
export OPENAI_API_KEY=sk-xxx        # 第三方兼容端点另设 OPENAI_API_BASE / OPENAI_MODEL
python tutorials/01_chat_basics.py  # 按编号顺序跑
```
# zy-agent
