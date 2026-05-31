"""
LLMFactory：统一管理 ModelType -> ChatOpenAI 映射
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI

from core.llm.config import get_config
from core.llm.model_type import ModelType
from core.llm.providers import bailian, openai

logger = logging.getLogger(__name__)


class LLMFactory:
    """模型工厂：按 ModelType 返回预创建的 ChatOpenAI 实例。"""

    def __init__(self) -> None:
        self._registry: dict[ModelType, ChatOpenAI] = {}
        self._load_providers()

    def _load_providers(self) -> None:
        # OpenAI 提供者
        for mt, instance in openai.PROVIDERS.items():
            self._register(mt, instance)
        # Bailian 提供者
        for mt, instance in bailian.PROVIDERS.items():
            self._register(mt, instance)

    def _register(self, model_type: ModelType, instance: ChatOpenAI) -> None:
        self._registry[model_type] = instance
        logger.debug("[llm_factory] registered %s -> %s", model_type, instance.model_name)

    def get_model(self, model_type: ModelType | str) -> ChatOpenAI:
        """获取模型实例；支持传入 ModelType 或字符串。"""
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        if model_type not in self._registry:
            raise KeyError(f"模型 {model_type.value} 未注册")
        return self._registry[model_type]

    def list_models(self) -> list[str]:
        """列出已注册的模型名称。"""
        return [mt.value for mt in self._registry]


# 全局工厂实例
llm_factory = LLMFactory()
