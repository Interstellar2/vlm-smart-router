"""
OpenAI Provider：gpt-4o、gpt-4o-mini 实例
"""

from langchain_openai import ChatOpenAI

from core.llm.config import get_config
from core.llm.model_type import ModelType

_config = get_config()
_openai_cfg = _config.get("providers", {}).get("openai", {})


_OPENAI_API_KEY = _openai_cfg.get("api_key", "")
_OPENAI_BASE_URL = _openai_cfg.get("base_url", "https://api.openai.com/v1")


PROVIDERS: dict[ModelType, ChatOpenAI] = {
    ModelType.OPENAI_GPT_4O: ChatOpenAI(
        model="gpt-4o",
        api_key=_OPENAI_API_KEY or "EMPTY",
        base_url=_OPENAI_BASE_URL,
        temperature=0.2,
        max_retries=1,
    ),
    ModelType.OPENAI_GPT_4O_MINI: ChatOpenAI(
        model="gpt-4o-mini",
        api_key=_OPENAI_API_KEY or "EMPTY",
        base_url=_OPENAI_BASE_URL,
        temperature=0.2,
        max_retries=1,
    ),
}
