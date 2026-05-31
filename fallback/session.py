"""
Fallback Session：支持 sticky 降级的模型调用会话
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage

from config import fallback_config
from core.llm.factory import llm_factory
from fallback.errors import classify_invoke_error

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class FallbackHop:
    """一次降级跳跃记录。"""

    from_model: str
    to_model: str
    reason: str
    turn: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_model": self.from_model,
            "to_model": self.to_model,
            "reason": self.reason,
            "turn": self.turn,
        }


@dataclass
class FallbackSession:
    """单次评测 case 的模型调用会话（支持 sticky 降级）。"""

    requested_model: str
    agent_id: str = "unknown"
    active_model: str | None = None
    fallback_hops: list[FallbackHop] = field(default_factory=list)
    _chain_index: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.active_model is None:
            self.active_model = self.requested_model

    async def ainvoke(
        self,
        prompt: str,
        *,
        image_urls: list[str] | None = None,
        turn: str | None = None,
    ) -> dict[str, Any]:
        """调用模型，失败时自动 fallback（仅网络/API 错误触发降级）。"""
        async with self._lock:
            fc = fallback_config
            assert self.active_model is not None
            if not fc.sticky_per_case:
                self.active_model = self.requested_model
                self._chain_index.clear()
            hops_this_call = 0

            while True:
                try:
                    model = llm_factory.get_model(self.active_model)
                    messages = [_build_human_message(prompt, image_urls)]
                    response = await model.ainvoke(messages)
                    return _parse_response(response)
                except Exception as e:
                    # parse_error 直接抛出，不走 fallback，由外层 quality_escalate 捕获
                    if isinstance(e, (ValueError, json.JSONDecodeError, TypeError)):
                        raise
                    eligible, reason = classify_invoke_error(e)
                    if not fc.enabled or not eligible:
                        raise
                    next_model = self._next_fallback_model()
                    if next_model is None or hops_this_call >= fc.max_hops:
                        raise
                    hops_this_call += 1
                    self._record_hop(next_model, reason, turn)

    async def structured_ainvoke(
        self,
        prompt: str,
        schema: type[T],
        *,
        image_urls: list[str] | None = None,
        turn: str | None = None,
    ) -> T:
        """调用模型并校验返回结构，parse_error 直接抛出，不走 fallback。"""
        async with self._lock:
            fc = fallback_config
            assert self.active_model is not None
            if not fc.sticky_per_case:
                self.active_model = self.requested_model
                self._chain_index.clear()
            hops_this_call = 0

            while True:
                try:
                    model = llm_factory.get_model(self.active_model)
                    messages = [_build_human_message(prompt, image_urls)]
                    response = await model.ainvoke(messages)
                    parsed = _parse_response(response)
                    validated = schema(**parsed)
                    return validated
                except Exception as e:
                    if isinstance(e, (ValueError, json.JSONDecodeError, TypeError)):
                        raise
                    eligible, reason = classify_invoke_error(e)
                    if not fc.enabled or not eligible:
                        raise
                    next_model = self._next_fallback_model()
                    if next_model is None or hops_this_call >= fc.max_hops:
                        raise
                    hops_this_call += 1
                    self._record_hop(next_model, reason, turn)

    def _record_hop(
        self, next_model: str, reason: str, turn: str | None
    ) -> None:
        assert self.active_model is not None
        hop = FallbackHop(
            from_model=self.active_model,
            to_model=next_model,
            reason=reason,
            turn=turn,
        )
        self.fallback_hops.append(hop)
        logger.warning(
            "[fallback] agent=%s %s -> %s reason=%s",
            self.agent_id, hop.from_model, hop.to_model, hop.reason,
        )
        self.active_model = next_model

    def _next_fallback_model(self) -> str | None:
        assert self.active_model is not None
        fc = fallback_config
        chain = fc.chains.get(self.active_model, [])
        idx = self._chain_index.get(self.active_model, 0)
        if idx >= len(chain):
            return None
        nxt = chain[idx]
        self._chain_index[self.active_model] = idx + 1
        return nxt

    def to_meta(self) -> dict[str, Any]:
        assert self.active_model is not None
        return {
            "requested_model": self.requested_model,
            "actual_model": self.active_model,
            "fallback_hops": [h.to_dict() for h in self.fallback_hops],
            "used_fallback": bool(self.fallback_hops),
        }


def _build_human_message(prompt: str, image_urls: list[str] | None = None) -> HumanMessage:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_urls:
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
    return HumanMessage(content=content)


def _parse_response(response: Any) -> dict[str, Any]:
    """将模型响应解析为 dict；content 为字符串时尝试 JSON，否则包装为 raw。"""
    if isinstance(response, dict):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": content}
    if isinstance(content, list) and content:
        text = content[0] if isinstance(content[0], str) else str(content[0])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return {"raw": str(response)}
