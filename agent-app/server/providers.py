"""多平台模型 Provider 工厂。

通过 .env 的 LLM_PROVIDER 切换平台，全部走 OpenAI 兼容协议：
    sensenova | openai | <任意自定义前缀，如 deepseek/qwen/...>

新增平台：在 PROVIDERS 里加一项 (env前缀, 默认base_url, 默认model) 即可。
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# 平台注册表：provider名 -> (环境变量前缀, 默认 base_url, 默认 model)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "sensenova": ("SENSENOVA", "https://token.sensenova.cn/v1", "sensenova-6.8-flash-lite"),
    "openai": ("OPENAI", "https://api.openai.com/v1", "gpt-4o-mini"),
}


class ProviderConfig:
    def __init__(self, name: str, api_key: str, base_url: str, model: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def __repr__(self) -> str:  # 避免日志泄漏 key
        return f"Provider({self.name}, model={self.model}, base_url={self.base_url})"


@lru_cache(maxsize=8)
def get_provider(name: str | None = None) -> ProviderConfig:
    """解析 provider 配置；缺 key 时抛出带指引的 RuntimeError。"""
    name = (name or os.getenv("LLM_PROVIDER", "sensenova")).strip().lower()
    if name == "dry-run":
        raise RuntimeError("dry-run 不是真实 provider，请走 agent._build_model 的分支")
    if name not in PROVIDERS:
        # 自定义平台：直接用 <NAME>_API_KEY / <NAME>_BASE_URL / <NAME>_MODEL
        prefix = name.upper()
        PROVIDERS[name] = (prefix, "", "")
    prefix, default_base, default_model = PROVIDERS[name]

    api_key = os.getenv(f"{prefix}_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            f"provider '{name}' 缺少 API key：请在 agent-app/.env 设置 {prefix}_API_KEY"
            f"（可选 {prefix}_BASE_URL / {prefix}_MODEL）"
        )
    return ProviderConfig(
        name=name,
        api_key=api_key,
        base_url=os.getenv(f"{prefix}_BASE_URL", "") or default_base,
        model=os.getenv(f"{prefix}_MODEL", "") or default_model,
    )


def build_chat_model(provider: ProviderConfig | None = None, **overrides):
    """构建 ChatOpenAI（所有兼容平台统一走这个类）。"""
    from langchain_openai import ChatOpenAI

    p = provider or get_provider()
    kwargs = dict(
        model=overrides.pop("model", p.model),
        temperature=overrides.pop("temperature", 0),
        base_url=p.base_url or None,
        api_key=p.api_key,
        timeout=overrides.pop("timeout", 120),
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
