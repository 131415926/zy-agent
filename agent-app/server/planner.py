"""Planner：LLM 把用户任务拆解为步骤计划（结构化输出）+ 步骤状态管理。

计划格式：[{"step": str, "status": "pending"|"done"|"failed"}]
- 简单任务（is_trivial）不走规划，直接由 executor 处理
- 步骤状态由 executor 节点推进，reflect 节点根据状态决定循环还是结束
"""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from .providers import build_chat_model


class Step(BaseModel):
    step: str = Field(description="这一步要做什么（一句话，可执行）")
    status: Literal["pending", "done", "failed"] = "pending"


class Plan(BaseModel):
    trivial: bool = Field(description="任务是否平凡简单：闲聊/单步可完成=true")
    steps: list[Step] = Field(description="执行步骤列表（trivial=true 时为空）")


TRIVIAL_MARKERS = ("你好", "hi ", "hello", "你是谁", "谢谢", "再见", "介绍一下你自己")


def is_trivial(message: str) -> bool:
    """快速预判：寒暄/单步闲聊直接走 executor，不浪费一次规划调用。"""
    low = message.strip().lower()
    return any(low.startswith(m) or low == m for m in TRIVIAL_MARKERS) and len(message) < 30


_PLANNER_SYSTEM = """你是任务规划器。把用户的任务拆解为 2-6 个可独立执行的步骤。
要求：
1. 每步一句话，具体可执行（含要操作的文件/命令对象），不要空话。
2. 步骤按依赖排序；读文件/查信息类步骤放前面，写文件/执行命令放后面。
3. 如果任务确实平凡简单（寒暄、单句问答、不需要工具），trivial=true 且 steps 为空。
4. 只输出结构化结果，不要额外解释。"""


def make_plan(message: str, history: list | None = None) -> Plan:
    """调用 LLM 生成结构化计划（同步；executor 子图节点在线程池中运行）。

    注意：不用 with_structured_output —— SenseNova 等端点不支持 json_schema
    响应格式（400）。改为提示词要求纯 JSON + 本地解析校验，兼容任何 OpenAI
    兼容平台。
    """
    if is_trivial(message):
        return Plan(trivial=True, steps=[])

    llm = build_chat_model()
    schema_hint = (
        '只输出一个 JSON 对象（不要 markdown 代码块）：\n'
        '{"trivial": false, "steps": [{"step": "第一步要做什么", "status": "pending"}, ...]}'
    )
    msgs = [{"role": "system", "content": f"{_PLANNER_SYSTEM}\n{schema_hint}"}]
    if history:
        brief = "\n".join(f"{m.get('role')}: {str(m.get('content'))[:100]}" for m in history[-4:])
        msgs.append({"role": "system", "content": f"相关历史摘要（供参考，勿重复执行）：\n{brief}"})
    msgs.append({"role": "user", "content": f"任务：{message}"})
    raw = llm.invoke(msgs).content or ""

    plan = _parse_plan(raw)
    if plan is None:
        # 模型没按格式输出：退化为单步计划，交给 executor 直接干
        return Plan(trivial=False, steps=[Step(step=message)])
    # 防御：trivial=true 却带步骤的统一清空
    if plan.trivial:
        plan.steps = []
    return plan


def _parse_plan(raw: str) -> Plan | None:
    """从模型回复中提取并校验计划 JSON；失败返回 None（调用方走兜底）。"""
    text = raw.strip()
    # 容忍 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # 容忍前后多余文本：截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        steps = [Step(step=str(s["step"])).model_dump()
                 for s in data.get("steps", []) if s.get("step")]
        return Plan(trivial=bool(data.get("trivial")), steps=[Step(**s) for s in steps])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return None


def plan_to_text(plan: list[dict]) -> str:
    """渲染计划清单（带状态符号），用于下发前端/CLI 展示。"""
    marks = {"done": "[x]", "failed": "[!]", "pending": "[ ]"}
    return "\n".join(f"{marks.get(s.get('status'), '[ ]')} {i+1}. {s['step']}"
                     for i, s in enumerate(plan))


def parse_plan_json(raw: str) -> list[dict]:
    """从文本恢复计划列表（checkpointer 读回时 state 里的 plan 已是 dict，此函数兜底用）。"""
    try:
        data = json.loads(raw)
        return [Step(**s).model_dump() for s in data.get("steps", [])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
