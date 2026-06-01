"""
百炼 Provider：qwen-vl-max、qwen-vl-plus、qwen-vl-turbo 实例
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
    "extra_body": {"enable_thinking": False},
}


PROVIDERS: dict[ModelType, ChatOpenAI] = {
    ModelType.BAILIAN_QWEN_VL_MAX: ChatOpenAI(
        model="qwen-vl-max",
        **_BAILIAN_COMMON_KWARGS,
    ),
    ModelType.BAILIAN_QWEN_VL_PLUS: ChatOpenAI(
        model="qwen-vl-plus",
        **_BAILIAN_COMMON_KWARGS,
    ),
    ModelType.BAILIAN_QWEN_VL_TURBO: ChatOpenAI(
        model="qwen-vl-turbo",
        **_BAILIAN_COMMON_KWARGS,
    ),
}
