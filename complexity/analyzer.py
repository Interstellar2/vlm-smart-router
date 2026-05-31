"""
图片复杂度 CV 分析
提取自 evaluate_platform/model/routing/image_complexity.py，保留核心逻辑。
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageComplexityProfile:
    """单张图 CV 复杂度画像；composite_score 越大表示越复杂。"""

    composite_score: float
    background_uniformity: float
    edge_density: float
    color_entropy: float
    foreground_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _border_uniformity_score(bgr: np.ndarray, border_ratio: float = 0.08) -> float:
    """边缘区域颜色一致性，越高表示背景越干净（越简单）。"""
    h, w = bgr.shape[:2]
    bh = max(1, int(h * border_ratio))
    bw = max(1, int(w * border_ratio))
    strips = [
        bgr[:bh, :],
        bgr[h - bh :, :],
        bgr[:, :bw],
        bgr[:, w - bw :],
    ]
    lab_strips = [cv2.cvtColor(s, cv2.COLOR_BGR2LAB) for s in strips if s.size > 0]
    if not lab_strips:
        return 0.5
    lab = np.concatenate([x.reshape(-1, 3) for x in lab_strips], axis=0).astype(np.float32)
    std = float(np.mean(np.std(lab, axis=0)))
    return float(np.clip(1.0 - std / 40.0, 0.0, 1.0))


def _edge_density_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.clip(edges.mean() / 64.0, 0.0, 1.0))


def _color_entropy_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = np.log2(32 * 32)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _foreground_ratio_score(bgr: np.ndarray) -> float:
    """粗估前景占比：与四边均值差异大的像素视为前景。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_mask = np.zeros((h, w), dtype=bool)
    bh = max(1, int(h * 0.08))
    bw = max(1, int(w * 0.08))
    border_mask[:bh, :] = True
    border_mask[h - bh :, :] = True
    border_mask[:, :bw] = True
    border_mask[:, w - bw :] = True
    border_mean = lab[border_mask].mean(axis=0)
    dist = np.linalg.norm(lab - border_mean, axis=2)
    fg = dist > 18.0
    return float(np.clip(fg.mean(), 0.0, 1.0))


def analyze_bgr_complexity(
    bgr: np.ndarray,
    *,
    max_side: int = 512,
    role: str = "generic",
) -> ImageComplexityProfile:
    """对 BGR 图像做轻量复杂度分析。"""
    img = _resize_max_side(bgr, max_side)
    bg_uniform = _border_uniformity_score(img)
    edge = _edge_density_score(img)
    entropy = _color_entropy_score(img)
    fg_ratio = _foreground_ratio_score(img)

    bg_complexity = 1.0 - bg_uniform
    if role == "cutout":
        composite = 0.15 * bg_complexity + 0.55 * edge + 0.20 * entropy + 0.10 * fg_ratio
    else:
        composite = 0.45 * bg_complexity + 0.20 * edge + 0.25 * entropy + 0.10 * fg_ratio

    return ImageComplexityProfile(
        composite_score=float(np.clip(composite, 0.0, 1.0)),
        background_uniformity=bg_uniform,
        edge_density=edge,
        color_entropy=entropy,
        foreground_ratio=fg_ratio,
    )


def merge_profiles(
    profiles: list[ImageComplexityProfile],
) -> ImageComplexityProfile:
    """多图取最复杂一张的指标（保守路由）。"""
    if not profiles:
        return ImageComplexityProfile(0.5, 0.5, 0.5, 0.5, 0.5)
    idx = max(range(len(profiles)), key=lambda i: profiles[i].composite_score)
    return profiles[idx]
