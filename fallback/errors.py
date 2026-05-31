"""
错误分类：判断异常是否应该触发模型降级 fallback
提取自 evaluate_platform/model/routing/fallback_errors.py
"""

import asyncio


# 触发降级的 HTTP 状态码
_FALLBACK_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_NO_FALLBACK_STATUS_CODES = frozenset({401, 403})


def _extract_http_status(exc: BaseException) -> int | None:
    """从异常及其属性中提取 HTTP 状态码。"""
    # 常见 SDK 的属性
    for attr in ("status_code", "status", "response_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # 字符串匹配
    msg = str(exc)
    for code in list(_FALLBACK_STATUS_CODES) + list(_NO_FALLBACK_STATUS_CODES) + [400, 404]:
        if f"{code}" in msg:
            return code
    return None


def classify_invoke_error(exc: BaseException) -> tuple[bool, str]:
    """是否应走 API fallback，以及原因码。"""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True, "timeout"

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        status = _extract_http_status(current)
        if status is not None:
            if status in _NO_FALLBACK_STATUS_CODES:
                return False, f"http_{status}"
            if status in _FALLBACK_STATUS_CODES:
                return True, f"http_{status}"
            if status == 404:
                return True, "model_not_found"
            if status == 400:
                body = str(current).lower()
                if any(
                    k in body
                    for k in (
                        "model",
                        "not found",
                        "does not exist",
                        "invalid_model",
                        "model_not_found",
                    )
                ):
                    return True, "model_unavailable"

        name = type(current).__name__
        if name in ("APIConnectionError", "ConnectError", "ConnectTimeout"):
            return True, "connection_error"
        if name in ("APITimeoutError", "ReadTimeout", "WriteTimeout"):
            return True, "timeout"
        if name == "RateLimitError":
            return True, "rate_limit"

        current = current.__cause__ or current.__context__

    return False, "not_fallback_eligible"
