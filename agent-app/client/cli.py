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


def _render_plan(payload_json: str) -> None:
    """渲染服务端下发的计划清单（JSON 数组 [{step, status}]）。"""
    import json as _json

    steps = _json.loads(payload_json)
    lines = []
    for i, s in enumerate(steps, 1):
        mark = {"done": "[green]✓[/green]", "failed": "[red]✗[/red]", "pending": "[dim]○[/dim]"}.get(
            s.get("status", "pending"), "[dim]○[/dim]")
        lines.append(f"{mark} {i}. {s['step']}")
    console.print(Panel("\n".join(lines), title="📋 执行计划", border_style="blue"))


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
        chunks, tools = [], []
        console.print("[dim]⏳ 处理中…（多工具任务可能需要 1-2 分钟）[/dim]")
        try:
            for event, data in client.chat_stream(sid, text):
                if event == "start":
                    sid = data  # 流一开始就拿到会话 id，审批时不再为空
                elif event == "plan":
                    _render_plan(data)
                elif event == "message":
                    if not chunks:
                        console.print("[dim]✓ 模型开始回复[/dim]")
                    chunks.append(data)
                    print(data, end="", flush=True)
                elif event == "tool":
                    tools.append(data)
                    console.print(f"[cyan]🔧 {data}[/cyan]")
                elif event == "approval":
                    handle_approval(data)
                elif event == "done":
                    sid = data
                elif event == "error":
                    console.print(f"\n[red]服务端错误：{data}[/red]")
                    return
            print()
        except AgentError as e:
            console.print(f"[red]请求失败：{e}[/red]")
            return
        if tools:
            console.print(f"[cyan]🔧 工具调用：{'; '.join(tools)}[/cyan]")

    def handle_approval(payload_json: str) -> None:
        """流式中收到审批请求：展示详情并在终端内 y/n 确认。"""
        import json as _json

        nonlocal sid
        pa = _json.loads(payload_json)
        if not sid:
            console.print("[red]未获取到会话 ID，无法提交审批（请重启 CLI 后重试）[/red]")
            return
        args_str = _json.dumps(pa["args"], ensure_ascii=False, indent=2)
        console.print(f"[yellow]⛔ 需要审批：{pa['tool']}[/yellow]")
        console.print(Panel(args_str, title=pa.get("question", "操作详情"), border_style="yellow"))
        console.print("[dim]  y = 允许本次   a = 本会话总是允许   n = 拒绝[/dim]")
        ans = console.input("[yellow]你的选择 [y/a/n] [/yellow]").strip().lower()
        decision = {"a": "always", "y": "approve"}.get(ans, "reject")
        # 审批走同步接口（resume 后继续跑完本轮）
        result = client.approve(sid, decision)
        sid = result.get("session_id", sid)
        reply = result.get("reply", "")
        if reply:
            console.print(Panel(Markdown(reply), title=f"assistant · {sid}", border_style="green"))
        # 拒绝后模型可能给出新的说明，或又产生下一个待审批
        while result.get("pending_approval"):
            pa2 = result["pending_approval"]
            console.print(f"[yellow]⛔ 又一个待审批：{pa2['tool']} {pa2['args']}[/yellow]")
            ans2 = console.input("[yellow]批准执行? [y/n] [/yellow]").strip().lower()
            result = client.approve(sid, "approve" if ans2 in ("y", "yes") else "reject")
            sid = result.get("session_id", sid)
            if result.get("reply"):
                console.print(Panel(Markdown(result["reply"]), title=f"assistant · {sid}", border_style="green"))

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
        # 清洗粘贴内容里的孤立代理字符（macOS 终端复制 emoji 等常见），否则 httpx 编码崩溃
        text = text.encode("utf-8", errors="replace").decode("utf-8")
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


@app.command()
def logs(
    session_id: str = typer.Argument(..., help="会话 ID"),
    server: str = typer.Option("http://127.0.0.1:8000", help="服务端地址"),
    limit: int = typer.Option(5, "--limit", "-l", help="最近几次请求"),
):
    """查看会话执行 trace：耗时 / token / 事件序列。"""
    client = _client(server)
    data = client.traces(session_id, limit=limit)
    runs = data.get("runs", [])
    if not runs:
        console.print("[dim]该会话暂无 trace 记录（trace 为进程内存态，服务重启后清空）[/dim]")
        return
    for r in runs:
        u = r.get("usage", {})
        tok = f"{u.get('prompt_tokens', 0)}+{u.get('completion_tokens', 0)}tok"
        tok += " (估算)" if u.get("estimated") else ""
        console.print(Panel(
            f"[bold]{r['message']}[/bold]\n"
            f"[dim]耗时 {r['total_ms']}ms · token {tok}[/dim]\n" +
            "\n".join(f"  {e['t']:>6}s {e['type']:<8} {e.get('name','')} {e.get('detail','')[:60]}"
                      for e in r.get("events", [])),
            title=f"run @ {r['start']:.0f}", border_style="magenta"))


@app.command()
def stats(
    session_id: str = typer.Argument(..., help="会话 ID"),
    server: str = typer.Option("http://127.0.0.1:8000", help="服务端地址"),
):
    """会话级汇总：请求数 / 总耗时 / 总 token / 事件分布。"""
    client = _client(server)
    s = client.traces_summary(session_id)
    console.print(Panel(
        f"请求数: {s.get('runs', 0)}\n"
        f"总耗时: {s.get('total_ms', 0)}ms\n"
        f"token: {s.get('prompt_tokens', 0)}+{s.get('completion_tokens', 0)}\n"
        f"事件分布: {s.get('events', {})}",
        title=f"📊 {session_id} 汇总", border_style="magenta"))


if __name__ == "__main__":
    app()
