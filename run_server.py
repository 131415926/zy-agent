"""zy-agent 启动脚本：在项目根目录即可启动服务端，免 cd agent-app。

用法：conda activate LangChain && python run_server.py [--port 8000] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-app"))

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-app"))
    from server.main import main

    main()
