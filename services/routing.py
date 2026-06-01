"""
VLM Agent 路由：按图片复杂度分档选模型
提取自 evaluate_platform/model/routing/vlm_agent_router.py
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import httpx

from complexity.analyzer import ImageComplexityProfile, analyze_bgr_complexity, merge_profiles
from core.settings import routing_config, TierName

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """路由决策结果。"""

    agent_id: str
    model_type: str
    tier: str
    composite_score: float
    shadow_mode: bool
    profiles: dict[str, ImageComplexityProfile]

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "model_type": self.model_type,
            "tier": self.tier,
            "composite_score": self.composite_score,
            "shadow_mode": self.shadow_mode,
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
        }


class VlmAgentRouter:
    """按图像复杂度分档路由到不同模型。"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=routing_config.fetch_timeout_sec)

    async def resolve(
        self,
        agent_id: str,
        image_urls: dict[str, str] | list[str],
        role: str = "generic",
    ) -> RoutingDecision:
        rc = routing_config
        profile = rc.profile(agent_id)

        # 1. 路由关闭 -> 固定模型
        if not rc.enabled or not profile.routing_enabled:
            return RoutingDecision(
                agent_id=agent_id,
                model_type=profile.default_model,
                tier="fixed",
                composite_score=0.0,
                shadow_mode=False,
                profiles={},
            )

        # 2. 强制指定模型（调试用）
        forced = (rc.force_model or "").strip()
        if forced:
            return RoutingDecision(
                agent_id=agent_id,
                model_type=forced,
                tier="forced",
                composite_score=0.0,
                shadow_mode=False,
                profiles={},
            )

        # 3. 下载并分析图片
        labeled = self._label_urls(image_urls, profile.max_cv_images)
        profiles_map = await self._analyze_urls(labeled, max_side=rc.analysis_max_side, role=role)
        merged = merge_profiles(list(profiles_map.values()))
        tier, model_type = self._tier_for_score(profile, merged.composite_score)

        # 4. Shadow 模式：只记录决策，实际仍用 default_model
        if rc.shadow_mode:
            logger.info(
                "[shadow] agent=%s would_tier=%s would_model=%s score=%.3f",
                agent_id, tier, model_type, merged.composite_score,
            )
            return RoutingDecision(
                agent_id=agent_id,
                model_type=profile.default_model,
                tier="shadow",
                composite_score=merged.composite_score,
                shadow_mode=True,
                profiles=profiles_map,
            )

        logger.info(
            "[routing] agent=%s tier=%s model=%s score=%.3f",
            agent_id, tier, model_type, merged.composite_score,
        )
        return RoutingDecision(
            agent_id=agent_id,
            model_type=model_type,
            tier=tier,
            composite_score=merged.composite_score,
            shadow_mode=False,
            profiles=profiles_map,
        )

    def escalate_model(self, agent_id: str, current: str) -> str | None:
        """质量升档：返回当前模型的下一档更强模型。"""
        profile = routing_config.profile(agent_id)
        order = [profile.tier_low, profile.tier_mid, profile.tier_high]
        unique: list[str] = []
        for m in order:
            if m not in unique:
                unique.append(m)
        try:
            idx = unique.index(current)
        except ValueError:
            return None
        if idx + 1 >= len(unique):
            return None
        return unique[idx + 1]

    # --- internal ---

    def _label_urls(
        self, image_urls: dict[str, str] | list[str], max_images: int
    ) -> dict[str, str]:
        if isinstance(image_urls, list):
            labeled = {f"img_{i}": url for i, url in enumerate(image_urls)}
        else:
            labeled = dict(image_urls)
        if len(labeled) > max_images:
            labeled = dict(list(labeled.items())[:max_images])
        return labeled

    async def _analyze_urls(
        self, labeled: dict[str, str], max_side: int, role: str
    ) -> dict[str, ImageComplexityProfile]:
        sem = asyncio.Semaphore(8)

        async def _fetch_one(name: str, url: str) -> tuple[str, ImageComplexityProfile]:
            async with sem:
                try:
                    if url.startswith("file://"):
                        path = url[len("file://"):]
                        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                    elif url.startswith("/"):
                        bgr = cv2.imread(url, cv2.IMREAD_COLOR)
                    else:
                        resp = await self._client.get(url)
                        resp.raise_for_status()
                        arr = np.frombuffer(resp.content, np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise ValueError(f"无法读取图片: {url}")
                    profile = await asyncio.to_thread(
                        analyze_bgr_complexity, bgr, max_side=max_side, role=role
                    )
                    return name, profile
                except Exception as e:
                    logger.warning("分析图片失败 %s: %s", url, e)
                    return name, ImageComplexityProfile(0.5, 0.5, 0.5, 0.5, 0.5)

        tasks = [_fetch_one(name, url) for name, url in labeled.items()]
        results = await asyncio.gather(*tasks)
        return {name: prof for name, prof in results}

    def _tier_for_score(
        self, profile, score: float
    ) -> tuple[str, str]:
        rc = routing_config
        if score < rc.tier_low_threshold:
            tier: TierName = "low"
        elif score > rc.tier_high_threshold:
            tier = "high"
        else:
            tier = "mid"
        tier = self._apply_min_tier(tier, profile.min_tier)
        return tier, self._model_for_tier(profile, tier)

    def _apply_min_tier(self, tier: TierName, min_tier: TierName) -> TierName:
        order: list[TierName] = ["low", "mid", "high"]
        if order.index(tier) < order.index(min_tier):
            return min_tier
        return tier

    def _model_for_tier(self, profile, tier: TierName) -> str:
        if tier == "low":
            return profile.tier_low
        if tier == "mid":
            return profile.tier_mid
        return profile.tier_high

    async def aclose(self) -> None:
        await self._client.aclose()
