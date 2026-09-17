"""zy-agent 启动脚本：在项目根目录即可启动 CLI 客户端，免 cd agent-app。

用法：conda activate LangChain && python run_cli.py chat [--server http://127.0.0.1:8000]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-app"))

if __name__ == "__main__":
    from client.cli import app

    app()
