"""
FastAPI 入口：演示图片复杂度路由 + 模型降级兜底 + 质量升档
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from complexity.analyzer import ImageComplexityProfile
from config import routing_config, fallback_config
from fallback.escalation import run_with_quality_escalate
from router.vlm_router import RoutingDecision, VlmAgentRouter
from schemas import EvaluateRequest, EvaluateResponse
from utils.image_utils import convert_image_urls_to_vision_urls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_router = VlmAgentRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _router.aclose()


app = FastAPI(title="VLM Demo", lifespan=lifespan)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    """
    演示完整流程：
    1. 分析图片复杂度
    2. 按复杂度路由到对应模型
    3. 调用模型（支持 fallback 降级）
    4. 质量升档兜底
    """
    # 1. 路由决策（使用原始 URL，router 内部处理 file://）
    labeled = {f"img_{i}": url for i, url in enumerate(req.image_urls)}
    routing = await _router.resolve(req.agent_id, labeled, role=req.role)

    # 2. 转换图片 URL 为 vision API 可用格式
    vision_urls = convert_image_urls_to_vision_urls(req.image_urls)

    # 3. 调用模型（自动 fallback + quality_escalate）
    prompt = f"Evaluate images with role={req.role}"
    try:
        result, meta = await run_with_quality_escalate(
            _router, req.agent_id, routing, prompt, image_urls=vision_urls
        )
    except Exception as e:
        logger.error("Evaluate failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    # 4. 构造响应
    complexity = ImageComplexityProfile(
        composite_score=routing.composite_score,
        background_uniformity=0.0,
        edge_density=0.0,
        color_entropy=0.0,
        foreground_ratio=0.0,
    )
    if routing.profiles:
        from complexity.analyzer import merge_profiles
        merged = merge_profiles(list(routing.profiles.values()))
        complexity = ImageComplexityProfile(
            composite_score=merged.composite_score,
            background_uniformity=merged.background_uniformity,
            edge_density=merged.edge_density,
            color_entropy=merged.color_entropy,
            foreground_ratio=merged.foreground_ratio,
        )

    return EvaluateResponse(
        agent_id=req.agent_id,
        routing=RoutingDecision(
            agent_id=routing.agent_id,
            model_type=routing.model_type,
            tier=routing.tier,
            composite_score=routing.composite_score,
            shadow_mode=routing.shadow_mode,
            profiles={k: ImageComplexityProfile(**v.to_dict()) for k, v in routing.profiles.items()},
        ),
        complexity=complexity,
        result=result,
        meta=meta,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def get_config():
    """查看当前路由与降级配置。"""
    return {
        "routing": {
            "enabled": routing_config.enabled,
            "shadow_mode": routing_config.shadow_mode,
            "tier_low_threshold": routing_config.tier_low_threshold,
            "tier_high_threshold": routing_config.tier_high_threshold,
            "force_model": routing_config.force_model,
            "agents": {
                k: {
                    "routing_enabled": v.routing_enabled,
                    "tier_low": v.tier_low,
                    "tier_mid": v.tier_mid,
                    "tier_high": v.tier_high,
                    "min_tier": v.min_tier,
                    "default_model": v.default_model,
                    "quality_escalate": v.quality_escalate,
                }
                for k, v in routing_config.agents.items()
            },
        },
        "fallback": {
            "enabled": fallback_config.enabled,
            "max_hops": fallback_config.max_hops,
            "sticky_per_case": fallback_config.sticky_per_case,
            "chains": fallback_config.chains,
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
