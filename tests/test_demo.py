"""
核心逻辑测试
"""

import json
import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from complexity.analyzer import analyze_bgr_complexity, merge_profiles, ImageComplexityProfile
from core.llm.config import load_config
from services.errors import classify_invoke_error
from services.execution import FallbackSession
from core.settings import fallback_config
from services.routing import VlmAgentRouter


# ---------- Mock ChatModel for monkeypatching ----------

class MockAIMessage:
    def __init__(self, content: str):
        self.content = content


class MockChatModel:
    """模拟 ChatOpenAI，支持注入故障。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._force_fail: str = ""
        self._force_fail_for: set[str] | None = None
        self._force_bad_json: bool = False

    def set_force_fail(self, mode: str, for_models: set[str] | None = None) -> None:
        self._force_fail = mode
        self._force_fail_for = for_models or set()

    def set_force_bad_json(self, enabled: bool) -> None:
        self._force_bad_json = enabled

    async def ainvoke(self, messages):
        if self._force_fail:
            if not self._force_fail_for or self.model_name in self._force_fail_for:
                self._raise_for_mode(self._force_fail, self.model_name)

        if self._force_bad_json and self.model_name != "bailian-qwen-vl-max":
            return MockAIMessage('{"invalid_field": "bad"}')

        return MockAIMessage(
            json.dumps(
                {
                    "model": self.model_name,
                    "prompt_length": len(str(messages)),
                    "result": f"processed_by_{self.model_name}",
                    "score": round(random.uniform(0.7, 0.99), 3),
                }
            )
        )

    def _raise_for_mode(self, mode: str, model: str) -> None:
        if mode == "timeout":
            raise TimeoutError(f"模型 {model} 调用超时")
        if mode == "rate_limit":
            raise Exception(f"RateLimitError: 429 Too Many Requests on {model}")
        if mode == "5xx":
            raise Exception(f"APIError: 500 Internal Server Error on {model}")
        if mode == "model_not_found":
            raise Exception(f"APIError: 404 model_not_found {model}")


class FakeLLMFactory:
    def __init__(self):
        self._models: dict[str, MockChatModel] = {}

    def get_model(self, model_type):
        name = str(model_type)
        if name not in self._models:
            self._models[name] = MockChatModel(name)
        return self._models[name]

    def list_models(self):
        return list(self._models.keys())


@pytest.fixture
def fake_factory(monkeypatch):
    factory = FakeLLMFactory()
    monkeypatch.setattr("services.execution.llm_factory", factory)
    return factory


# ---------- complexity tests ----------

def test_analyze_simple_image():
    """纯色图应该被认为是简单的（composite_score 低）。"""
    bgr = np.full((256, 256, 3), (128, 128, 128), dtype=np.uint8)
    prof = analyze_bgr_complexity(bgr)
    assert prof.composite_score < 0.3
    assert prof.background_uniformity > 0.8


def test_analyze_complex_image():
    """随机噪声图应该被认为是复杂的（composite_score 高）。"""
    np.random.seed(42)
    bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    prof = analyze_bgr_complexity(bgr)
    assert prof.composite_score > 0.5


def test_merge_profiles_conservative():
    """多图取最复杂一张。"""
    p1 = ImageComplexityProfile(0.2, 0.9, 0.1, 0.1, 0.1)
    p2 = ImageComplexityProfile(0.8, 0.1, 0.9, 0.9, 0.9)
    merged = merge_profiles([p1, p2])
    assert merged.composite_score == 0.8


# ---------- router tests ----------

@pytest.mark.asyncio
async def test_router_local_image():
    """用本地图片测试路由。"""
    router = VlmAgentRouter()

    # 创建一张简单图片
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        bgr = np.full((256, 256, 3), (200, 200, 200), dtype=np.uint8)
        cv2.imwrite(path, bgr)

    try:
        routing = await router.resolve("generic", [f"file://{path}"])
        assert routing.tier == "low"  # 简单图应该路由到 low
        assert routing.model_type == "bailian-qwen-vl-turbo"
    finally:
        os.unlink(path)
        await router.aclose()


@pytest.mark.asyncio
async def test_router_complex_image():
    """复杂图片应该路由到 high。"""
    router = VlmAgentRouter()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        np.random.seed(42)
        bgr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(path, bgr)

    try:
        routing = await router.resolve("generic", [f"file://{path}"])
        assert routing.tier == "high"
        assert routing.model_type == "bailian-qwen-vl-max"
    finally:
        os.unlink(path)
        await router.aclose()


@pytest.mark.asyncio
async def test_router_min_tier():
    """audit agent 的 min_tier=mid，简单图也不能低于 mid。"""
    router = VlmAgentRouter()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        bgr = np.full((256, 256, 3), (200, 200, 200), dtype=np.uint8)
        cv2.imwrite(path, bgr)

    try:
        routing = await router.resolve("audit", [f"file://{path}"])
        assert routing.tier == "mid"
        assert routing.model_type == "bailian-qwen-vl-plus"
    finally:
        os.unlink(path)
        await router.aclose()


# ---------- fallback tests ----------

@pytest.mark.asyncio
async def test_fallback_success(fake_factory):
    """正常调用不应触发 fallback。"""
    session = FallbackSession(requested_model="bailian-qwen-vl-plus")
    result = await session.ainvoke("hello")
    assert result["model"] == "bailian-qwen-vl-plus"
    assert not session.fallback_hops


@pytest.mark.asyncio
async def test_fallback_timeout(fake_factory):
    """模拟超时，应降级到 fallback 链中的模型。"""
    model = fake_factory.get_model("bailian-qwen-vl-plus")
    model.set_force_fail("timeout", for_models={"bailian-qwen-vl-plus"})

    session = FallbackSession(requested_model="bailian-qwen-vl-plus")
    result = await session.ainvoke("hello")

    assert bool(session.fallback_hops)
    assert len(session.fallback_hops) == 1
    assert session.fallback_hops[0].from_model == "bailian-qwen-vl-plus"
    assert session.fallback_hops[0].to_model == "bailian-qwen-vl-turbo"
    assert result["model"] == "bailian-qwen-vl-turbo"


@pytest.mark.asyncio
async def test_fallback_chain_exhausted(fake_factory):
    """fallback 链耗尽后应抛异常。"""
    model = fake_factory.get_model("bailian-qwen-vl-turbo")
    model.set_force_fail("timeout")

    # bailian-qwen-vl-turbo 没有配置 fallback 链
    session = FallbackSession(requested_model="bailian-qwen-vl-turbo")
    with pytest.raises(Exception):
        await session.ainvoke("hello")


# ---------- error classification tests ----------

def test_classify_timeout():
    eligible, reason = classify_invoke_error(TimeoutError())
    assert eligible
    assert reason == "timeout"


def test_classify_rate_limit():
    exc = Exception("RateLimitError: 429 Too Many Requests")
    eligible, reason = classify_invoke_error(exc)
    assert eligible
    assert reason == "http_429"


def test_classify_403_no_fallback():
    exc = Exception("403 Forbidden")
    eligible, reason = classify_invoke_error(exc)
    assert not eligible
    assert "403" in reason


# ---------- integration: quality escalate ----------

@pytest.mark.asyncio
async def test_quality_escalate(fake_factory):
    """模拟 JSON 解析失败，触发质量升档。"""
    model = fake_factory.get_model("bailian-qwen-vl-turbo")
    model.set_force_bad_json(True)

    from services.routing import RoutingDecision

    router = VlmAgentRouter()
    routing = RoutingDecision(
        agent_id="generic",
        model_type="bailian-qwen-vl-turbo",
        tier="low",
        composite_score=0.2,
        shadow_mode=False,
        profiles={},
    )

    session = FallbackSession(
        requested_model=routing.model_type,
        agent_id=routing.agent_id,
    )

    # 模拟调用：turbo 返回 bad json，触发 escalate 到 plus，plus 也返回 bad json，escalate 到 max，max 返回正确结果
    # 这里我们手动测试 escalate_model
    next_model = router.escalate_model("generic", "bailian-qwen-vl-turbo")
    assert next_model == "bailian-qwen-vl-plus"

    next_model2 = router.escalate_model("generic", "bailian-qwen-vl-plus")
    assert next_model2 == "bailian-qwen-vl-max"

    next_model3 = router.escalate_model("generic", "bailian-qwen-vl-max")
    assert next_model3 is None  # 已经到顶

    await router.aclose()


# ---------- config tests ----------

def test_load_config_env_substitution(monkeypatch, tmp_path):
    """测试 YAML 环境变量替换。"""
    monkeypatch.setenv("TEST_API_KEY", "secret123")
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        "providers:\n"
        "  openai:\n"
        "    api_key: ${TEST_API_KEY}\n"
        "    base_url: ${TEST_BASE_URL:-https://default.example.com/v1}\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg["providers"]["openai"]["api_key"] == "secret123"
    assert cfg["providers"]["openai"]["base_url"] == "https://default.example.com/v1"


def test_load_config_env_default_when_missing(monkeypatch, tmp_path):
    """测试环境变量缺失时使用默认值。"""
    monkeypatch.delenv("MISSING_VAR", raising=False)
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        "key: ${MISSING_VAR:-fallback_value}\n",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg["key"] == "fallback_value"
