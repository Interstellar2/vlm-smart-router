"""
Mock LLM Provider
用于演示：模拟不同模型的调用，支持通过参数控制失败行为。
"""

import asyncio
import random
from typing import Any


class MockLLMProvider:
    """模拟多模型 LLM 调用，支持注入故障。"""

    def __init__(self) -> None:
        # 每个模型的模拟延迟 (秒)
        self._latencies: dict[str, float] = {
            "qwen-vl-flash": 0.3,
            "qwen-vl-flash-lite": 0.2,
            "qwen-vl-plus": 0.6,
            "qwen-vl-max": 1.0,
            "gpt-4o": 0.8,
        }
        # 全局注入的故障（由请求参数设置）
        self._force_fail: str = ""
        self._force_fail_for: set[str] = set()   # 空集合表示对所有模型生效
        self._force_bad_json: bool = False

    def set_force_fail(self, mode: str, for_models: set[str] | None = None) -> None:
        self._force_fail = mode
        self._force_fail_for = for_models or set()

    def set_force_bad_json(self, enabled: bool) -> None:
        self._force_bad_json = enabled

    async def call(self, model: str, prompt: str) -> dict[str, Any]:
        """模拟模型调用。"""
        latency = self._latencies.get(model, 0.5)
        await asyncio.sleep(latency)

        # 检查故障注入
        if self._force_fail:
            if not self._force_fail_for or model in self._force_fail_for:
                self._raise_for_mode(self._force_fail, model)

        # 模拟返回结果
        if self._force_bad_json and model != "qwen-vl-max":
            # 非最强模型模拟返回错误 JSON，触发 quality escalate
            return {"invalid_field": "bad"}

        return {
            "model": model,
            "prompt_length": len(prompt),
            "result": f"processed_by_{model}",
            "score": round(random.uniform(0.7, 0.99), 3),
        }

    def _raise_for_mode(self, mode: str, model: str) -> None:
        """根据注入模式抛出对应异常。"""
        if mode == "timeout":
            raise TimeoutError(f"模型 {model} 调用超时")
        if mode == "rate_limit":
            raise Exception(f"RateLimitError: 429 Too Many Requests on {model}")
        if mode == "5xx":
            raise Exception(f"APIError: 500 Internal Server Error on {model}")
        if mode == "model_not_found":
            raise Exception(f"APIError: 404 model_not_found {model}")
