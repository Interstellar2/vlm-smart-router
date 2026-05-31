"""
质量升档与元数据合并封装
"""

import json
import logging

from config import routing_config
from fallback.session import FallbackSession
from router.vlm_router import RoutingDecision, VlmAgentRouter

logger = logging.getLogger(__name__)


async def run_with_quality_escalate(
    router: VlmAgentRouter,
    agent_id: str,
    routing: RoutingDecision,
    prompt: str,
    image_urls: list[str] | None = None,
) -> tuple[dict, dict]:
    """执行调用；JSON/校验失败时按 agent 配置升一档重试。"""
    profile = routing_config.profile(agent_id)
    primary = routing.model_type

    session = FallbackSession(
        requested_model=routing.model_type,
        agent_id=routing.agent_id,
    )

    try:
        result = await session.ainvoke(prompt, image_urls=image_urls, turn="main")
        meta = _merge_meta(routing, session, escalated=False)
        return result, meta
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        if not profile.quality_escalate:
            raise
        next_model = router.escalate_model(agent_id, primary)
        if next_model is None:
            raise
        logger.warning(
            "[quality_escalate] agent=%s %s -> %s error=%s",
            agent_id, primary, next_model, e,
        )
        esc_routing = RoutingDecision(
            agent_id=agent_id,
            model_type=next_model,
            tier="escalated",
            composite_score=routing.composite_score,
            shadow_mode=False,
            profiles=routing.profiles,
        )
        esc_session = FallbackSession(
            requested_model=next_model,
            agent_id=agent_id,
        )
        result = await esc_session.ainvoke(prompt, image_urls=image_urls, turn="escalated")
        meta = _merge_meta(
            esc_routing,
            esc_session,
            escalated=True,
            escalated_from=primary,
            escalate_reason=str(e),
        )
        meta["escalated_to"] = next_model
        return result, meta


def _merge_meta(
    routing: RoutingDecision,
    session: FallbackSession,
    *,
    escalated: bool = False,
    escalated_from: str | None = None,
    escalate_reason: str | None = None,
) -> dict:
    meta = routing.to_dict()
    meta.update(session.to_meta())
    meta["escalated"] = escalated
    if escalated_from is not None:
        meta["escalated_from"] = escalated_from
    if escalate_reason is not None:
        meta["escalate_reason"] = escalate_reason
    return meta
