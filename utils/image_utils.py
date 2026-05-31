"""
图片 URL 工具函数
"""

import base64
from pathlib import Path


def convert_image_urls_to_vision_urls(urls: list[str]) -> list[str]:
    """将 file:// 和绝对路径转换为 base64 data URI，HTTP URL 保持原样。"""
    result = []
    for url in urls:
        if url.startswith("file://"):
            path = url[len("file://"):]
            ext = Path(path).suffix.lower().lstrip(".") or "jpeg"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            result.append(f"data:image/{ext};base64,{data}")
        elif url.startswith("/"):
            ext = Path(url).suffix.lower().lstrip(".") or "jpeg"
            with open(url, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            result.append(f"data:image/{ext};base64,{data}")
        else:
            result.append(url)
    return result
