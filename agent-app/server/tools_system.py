"""真实工具集：文件读写 / 目录浏览 / 命令执行 / 代码搜索。

安全模型（对标 codex 类产品的权限分级）：
- 全部文件操作限制在 WORKSPACE_ROOT 沙箱内（路径白名单校验，拒绝 .. 逃逸与绝对路径越界）
- 命令执行 run_cmd 默认拦截危险命令（rm -rf / mkfs / shutdown 等），执行前需人工审批（见 main.py interrupt 机制）
"""
import fnmatch
import os
import shlex
import subprocess

from langchain_core.tools import tool

# 沙箱根目录：默认为项目所在目录，可用 WORKSPACE_ROOT 环境变量覆盖
WORKSPACE_ROOT = os.path.abspath(os.getenv("WORKSPACE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..")))

MAX_READ_CHARS = 20_000        # 单次读取上限，防止撑爆上下文
MAX_CMD_TIMEOUT = 60           # 命令执行超时（秒）
MAX_OUTPUT_CHARS = 10_000      # 命令输出截断上限

# 危险命令黑名单：命中即拒绝执行（无论是否审批）
DANGEROUS_PATTERNS = (
    "rm -rf /", "mkfs", "shutdown", "reboot", "halt",
    ":(){ :|:& };:", "dd if=", "> /dev/sda", "chmod -R 777 /",
)


def _safe_path(rel_path: str) -> str:
    """把相对路径解析到沙箱内，越界一律拒绝。"""
    if not rel_path or rel_path.startswith("~"):
        raise ValueError(f"路径不合法: {rel_path!r}（请用沙箱内相对路径）")
    full = os.path.abspath(os.path.join(WORKSPACE_ROOT, rel_path))
    if not (full == WORKSPACE_ROOT or full.startswith(WORKSPACE_ROOT + os.sep)):
        raise ValueError(f"路径越界: {rel_path!r}（只能访问 {WORKSPACE_ROOT} 内的文件）")
    return full


@tool
def list_dir(path: str = ".") -> str:
    """列出沙箱内目录内容（名称/类型/大小）。path 为相对路径，默认根目录。"""
    full = _safe_path(path)
    if not os.path.isdir(full):
        return f"不是目录: {path}"
    entries = []
    for name in sorted(os.listdir(full))[:200]:
        p = os.path.join(full, name)
        if os.path.isdir(p):
            entries.append(f"[dir]  {name}/")
        else:
            entries.append(f"[file] {name} ({os.path.getsize(p)}B)")
    return "\n".join(entries) or "(空目录)"


@tool
def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    """读取沙箱内文本文件。offset 起始行(1-based)，limit 最多行数。大文件请分段读。"""
    full = _safe_path(path)
    if not os.path.isfile(full):
        return f"文件不存在: {path}"
    with open(full, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    start = max(offset - 1, 0)
    picked = lines[start:start + limit]
    if not picked:
        return f"(行号超出范围，文件共 {len(lines)} 行)"
    body = "".join(picked)
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + f"\n…(截断，共{len(lines)}行，请用 offset/limit 分段读取)"
    header = f"（{path} 第{start+1}-{start+len(picked)}行/共{len(lines)}行）\n"
    return header + body


@tool
def write_file(path: str, content: str) -> str:
    """写入/覆盖沙箱内文件（写父目录不存在会自动创建）。返回写入结果。"""
    full = _safe_path(path)
    os.makedirs(os.path.dirname(full) or WORKSPACE_ROOT, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}（{len(content)} 字符）"


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """在沙箱内按通配符找文件，如 **/*.py。返回相对路径列表（最多100条）。"""
    import glob as _glob

    base = _safe_path(path)
    matches = _glob.glob(os.path.join(base, pattern), recursive=True)
    rels = [os.path.relpath(m, WORKSPACE_ROOT) for m in matches if os.path.isfile(m)][:100]
    return "\n".join(rels) or "(无匹配)"


@tool
def grep_files(pattern: str, path: str = ".", glob_: str = "*") -> str:
    """在沙箱内按正则搜索文件内容。path 限定目录，glob_ 限定文件名模式（如 *.py）。返回 file:line: 内容（最多50条）。"""
    import re as _re

    base = _safe_path(path)
    rx = _re.compile(pattern)
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules", ".idea")]
        for fn in files:
            if not fnmatch.fnmatch(fn, glob_):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            rel = os.path.relpath(fp, WORKSPACE_ROOT)
                            hits.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                            if len(hits) >= 50:
                                return "\n".join(hits) + "\n(已达50条上限)"
            except OSError:
                continue
    return "\n".join(hits) or "(无匹配)"


@tool
def run_cmd(command: str) -> str:
    """在沙箱根目录执行 shell 命令（白名单外的命令需要人工审批，危险命令直接拒绝）。"""
    cmd = command.strip()
    low = cmd.lower()
    if any(p in low for p in DANGEROUS_PATTERNS):
        return f"已拒绝：命令命中危险黑名单：{cmd}"
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=WORKSPACE_ROOT,
            capture_output=True, text=True, timeout=MAX_CMD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"命令超时（>{MAX_CMD_TIMEOUT}s）: {cmd}"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n…(输出截断)"
    return f"[exit {r.returncode}]\n{out or '(无输出)'}"


SYSTEM_TOOLS = [list_dir, read_file, write_file, glob_files, grep_files, run_cmd]

# 需要人工审批的工具：调用这些工具前图会 interrupt 暂停
APPROVAL_REQUIRED = {"run_cmd", "write_file"}


def sandbox_banner() -> str:
    return f"WORKSPACE_ROOT = {WORKSPACE_ROOT}"


if __name__ == "__main__":
    print(sandbox_banner())
    print(list_dir.invoke({}))
