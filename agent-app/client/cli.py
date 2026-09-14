"""CLI 交互聊天客户端（typer + rich）。

启动服务端后运行：
    cd agent-app && python -m client.cli chat                    # 交互聊天（流式）
    python -m client.cli chat --server http://127.0.0.1:8000
    python -m client.cli chat --session demo1 --once "你好"       # 单次提问
    python -m client.cli sessions                                # 查看会话
"""
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .sdk import AgentClient, AgentError

app = typer.Typer(help="LangGraph Agent CLI 客户端")
console = Console()


def _client(server: str) -> AgentClient:
    try:
        c = AgentClient(server)
        c.health()
        return c
    except AgentError:
        console.print(f"[red]无法连接服务端 {server}，请先启动：python -m server.main[/red]")
        sys.exit(1)


@app.command()
def chat(
    server: str = typer.Option("http://127.0.0.1:8000", help="服务端地址"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="会话 ID（多轮记忆）"),
    once: Optional[str] = typer.Option(None, "--once", "-o", help="单次提问后退出"),
):
    """与 Agent 聊天（默认进入交互模式，流式输出）。"""
    client = _client(server)
    sid = session

    def ask(text: str) -> None:
        nonlocal sid
        console.print()
        with console.status("[dim]思考中…[/dim]", spinner="dots"):
            chunks, tools = [], []
            try:
                for event, data in client.chat_stream(sid, text):
                    if event == "message":
                        chunks.append(data)
                    elif event == "tool":
                        tools.append(data)
                    elif event == "done":
                        sid = data
                    elif event == "error":
                        console.print(f"[red]服务端错误：{data}[/red]")
                        return
            except AgentError as e:
                console.print(f"[red]请求失败：{e}[/red]")
                return
        if tools:
            console.print(f"[cyan]🔧 工具调用：{'; '.join(tools)}[/cyan]")
        reply = "".join(chunks)
        console.print(Panel(Markdown(reply), title=f"assistant · {sid}", border_style="green"))

    if once is not None:
        ask(once)
        return

    console.print(f"[bold]Agent CLI[/bold] 已连接 {server}（/exit 退出，/new 新会话）")
    while True:
        try:
            text = console.input("[bold blue]你 > [/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/exit", "/quit", "exit", "quit"):
            break
        if text == "/new":
            sid = None
            console.print("[dim]已开启新会话[/dim]")
            continue
        ask(text)
    console.print("[dim]再见！[/dim]")


@app.command()
def sessions(server: str = typer.Option("http://127.0.0.1:8000", help="服务端地址")):
    """列出服务端活跃会话。"""
    client = _client(server)
    ss = client.sessions()
    if not ss:
        console.print("[dim]暂无活跃会话[/dim]")
        return
    for i, s in enumerate(ss, 1):
        console.print(f"{i}. {s}")


if __name__ == "__main__":
    app()
