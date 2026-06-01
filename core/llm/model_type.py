"""
模型类型枚举
"""

from enum import Enum


class ModelType(str, Enum):
    """支持的 VLM 模型类型。"""

    GPT5_2 = "gpt-5.2"

    QWEN3_6_PLUS = "qwen3.6-plus"
    QWEN3_6_FLASH = "qwen3.6-flash"
    QWEN3_6_MAX = "qwen3.6-max"

    QWEN3_5_PLUS = "qwen3.5-plus"
    QWEN3_5_FLASH = "qwen3.5-flash"

    QWEN3_MAX = "qwen3-max"
    QWEN_FLASH = "qwen-flash"

    QWEN3_VL_PLUS = "qwen3-vl-plus"
    QWEN3_VL_FLASH = "qwen3-vl-flash"

    QWEN_VL_MAX = "qwen-vl-max"
    QWEN_VL_PLUS = "qwen-vl-plus"
