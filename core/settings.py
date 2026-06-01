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
    tier_low: str = "qwen3.5-flash"
    tier_mid: str = "qwen3.5-plus"
    tier_high: str = "qwen3.6-plus"
    min_tier: TierName = "low"
    default_model: str = "gpt-5.2"
    quality_escalate: bool = True
    max_cv_images: int = 12


# 各 agent 的路由配置
DEFAULT_AGENT_PROFILES: dict[str, AgentRoutingProfile] = {
    "generic": AgentRoutingProfile(),
    "matting": AgentRoutingProfile(
        tier_low="qwen3.5-flash",
        tier_mid="qwen3.5-plus",
        tier_high="qwen3.6-plus",
        quality_escalate=True,
    ),
    "audit": AgentRoutingProfile(
        routing_enabled=True,
        tier_low="qwen3.6-plus",      # audit 最低用 mid
        tier_mid="qwen3.6-plus",
        tier_high="gpt-5.2",
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
    "gpt-5.2": ["qwen3.6-plus", "qwen3.5-plus"],
    "qwen3.6-flash": ["qwen3.5-flash"],
    "qwen3.6-plus": ["qwen3.5-plus", "qwen3.5-flash"],
    "qwen3.6-max": ["qwen3.6-plus", "qwen3.5-plus"],
    "qwen3.5-flash": ["qwen-flash"],
    "qwen3.5-plus": ["qwen3.5-flash"],
    "qwen3-vl-flash": ["qwen3.5-flash"],
    "qwen3-vl-plus": ["qwen3.5-plus"],
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
