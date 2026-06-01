"""
百炼 Provider：qwen3.x / qwen-vl 系列实例
"""

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from core.llm.config import get_config
from core.llm.model_type import ModelType

_config = get_config()
_bailian_cfg = _config.get("providers", {}).get("bailian", {})


_BAILIAN_API_KEY = _bailian_cfg.get("api_key", "")
_BAILIAN_BASE_URL = _bailian_cfg.get(
    "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

_bailian_rate_limiter = InMemoryRateLimiter(requests_per_second=3000 * 0.8 / 60)

_BAILIAN_COMMON_KWARGS = {
    "api_key": _BAILIAN_API_KEY or "EMPTY",
    "base_url": _BAILIAN_BASE_URL,
    "temperature": 0.2,
    "max_retries": 3,
    "rate_limiter": _bailian_rate_limiter,
}


def _with_thinking():
    return {"model_kwargs": {"extra_body": {"enable_thinking": False}}}


PROVIDERS: dict[ModelType, ChatOpenAI] = {
    ModelType.QWEN3_6_PLUS: ChatOpenAI(
        model="qwen3.6-plus",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN3_6_FLASH: ChatOpenAI(
        model="qwen3.6-flash",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN3_6_MAX: ChatOpenAI(
        model="qwen3.6-max",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN3_5_PLUS: ChatOpenAI(
        model="qwen3.5-plus",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN3_5_FLASH: ChatOpenAI(
        model="qwen3.5-flash",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN3_MAX: ChatOpenAI(
        model="qwen3-max",
        **_BAILIAN_COMMON_KWARGS,
        max_tokens=4096,
    ),
    ModelType.QWEN_FLASH: ChatOpenAI(
        model="qwen-flash",
        **_BAILIAN_COMMON_KWARGS,
    ),
    ModelType.QWEN3_VL_PLUS: ChatOpenAI(
        model="qwen3-vl-plus",
        **_BAILIAN_COMMON_KWARGS,
    ),
    ModelType.QWEN3_VL_FLASH: ChatOpenAI(
        model="qwen3-vl-flash",
        **_BAILIAN_COMMON_KWARGS,
        **_with_thinking(),
    ),
    ModelType.QWEN_VL_MAX: ChatOpenAI(
        model="qwen-vl-max",
        **_BAILIAN_COMMON_KWARGS,
    ),
    ModelType.QWEN_VL_PLUS: ChatOpenAI(
        model="qwen-vl-plus",
        **_BAILIAN_COMMON_KWARGS,
    ),
}
