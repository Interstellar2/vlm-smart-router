"""
路由配置与降级链配置
"""

from dataclasses import dataclass, field
from typing import Literal

TierName = Literal["low", "mid", "high"]


@dataclass
class AgentRoutingProfile:
    """单个 agent 的 CV 分档与默认模型配置。"""

    routing_enabled: bool = True
    tier_low: str = "bailian-qwen-vl-turbo"
    tier_mid: str = "bailian-qwen-vl-plus"
    tier_high: str = "bailian-qwen-vl-max"
    min_tier: TierName = "low"
    default_model: str = "bailian-qwen-vl-plus"
    quality_escalate: bool = True
    max_cv_images: int = 12


# 各 agent 的路由配置
DEFAULT_AGENT_PROFILES: dict[str, AgentRoutingProfile] = {
    "generic": AgentRoutingProfile(),
    "matting": AgentRoutingProfile(
        tier_low="bailian-qwen-vl-turbo",
        tier_mid="bailian-qwen-vl-plus",
        tier_high="bailian-qwen-vl-max",
        quality_escalate=True,
    ),
    "audit": AgentRoutingProfile(
        routing_enabled=True,
        tier_low="bailian-qwen-vl-plus",      # audit 最低用 plus
        tier_mid="bailian-qwen-vl-plus",
        tier_high="bailian-qwen-vl-max",
        min_tier="mid",
        quality_escalate=True,
    ),
}


@dataclass
class RoutingConfig:
    """VLM 按图像复杂度分档路由。"""

    enabled: bool = True
    shadow_mode: bool = False
    fetch_timeout_sec: int = 60
    analysis_max_side: int = 512
    tier_low_threshold: float = 0.38
    tier_high_threshold: float = 0.62
    force_model: str | None = None
    agents: dict[str, AgentRoutingProfile] = field(
        default_factory=lambda: DEFAULT_AGENT_PROFILES.copy()
    )

    def profile(self, agent_id: str) -> AgentRoutingProfile:
        return self.agents.get(agent_id, AgentRoutingProfile())


# 默认降级链: model_id -> 按顺序尝试的 fallback 模型列表
DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    "openai-gpt-4o": ["bailian-qwen-vl-max", "bailian-qwen-vl-plus"],
    "bailian-qwen-vl-max": ["bailian-qwen-vl-plus", "bailian-qwen-vl-turbo"],
    "bailian-qwen-vl-plus": ["bailian-qwen-vl-turbo"],
    "openai-gpt-4o-mini": ["bailian-qwen-vl-turbo"],
}


@dataclass
class FallbackConfig:
    """VLM API 调用失败时的模型降级链。"""

    enabled: bool = True
    max_hops: int = 2
    sticky_per_case: bool = True
    chains: dict[str, list[str]] = field(
        default_factory=lambda: DEFAULT_FALLBACK_CHAINS.copy()
    )


# 全局配置实例
routing_config = RoutingConfig()
fallback_config = FallbackConfig()
