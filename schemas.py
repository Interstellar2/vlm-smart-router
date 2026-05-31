"""
Pydantic 数据模型
"""

from pydantic import BaseModel, Field
from typing import Literal


class ImageComplexityProfile(BaseModel):
    """单张图 CV 复杂度画像。"""

    composite_score: float = Field(..., ge=0.0, le=1.0)
    background_uniformity: float = Field(..., ge=0.0, le=1.0)
    edge_density: float = Field(..., ge=0.0, le=1.0)
    color_entropy: float = Field(..., ge=0.0, le=1.0)
    foreground_ratio: float = Field(..., ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    """路由决策结果。"""

    agent_id: str
    model_type: str
    tier: str
    composite_score: float
    shadow_mode: bool
    profiles: dict[str, ImageComplexityProfile]


class FallbackHop(BaseModel):
    """一次降级跳跃记录。"""

    from_model: str
    to_model: str
    reason: str
    turn: str | None = None


class EvaluateRequest(BaseModel):
    """评测请求。"""

    agent_id: str = "generic"
    image_urls: list[str]
    role: Literal["generic", "cutout"] = "generic"


class EvaluateResponse(BaseModel):
    """评测响应。"""

    agent_id: str
    routing: RoutingDecision
    complexity: ImageComplexityProfile
    result: dict
    meta: dict
