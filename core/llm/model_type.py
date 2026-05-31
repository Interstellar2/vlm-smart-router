"""
模型类型枚举
"""

from enum import Enum


class ModelType(str, Enum):
    """支持的 VLM 模型类型。"""

    OPENAI_GPT_4O = "openai-gpt-4o"
    OPENAI_GPT_4O_MINI = "openai-gpt-4o-mini"

    BAILIAN_QWEN_VL_MAX = "bailian-qwen-vl-max"
    BAILIAN_QWEN_VL_PLUS = "bailian-qwen-vl-plus"
    BAILIAN_QWEN_VL_TURBO = "bailian-qwen-vl-turbo"
