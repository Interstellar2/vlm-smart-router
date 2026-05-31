"""
配置加载器：YAML + ${ENV_VAR} 替换 + .env 自动加载
"""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def _resolve_env_vars(value: str) -> str:
    """替换 ${VAR} 和 ${VAR:-default} 形式的环境变量。"""
    pattern = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.getenv(var_name, default or "")

    return pattern.sub(replacer, value)


def _recursive_resolve(obj):
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _recursive_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_resolve(i) for i in obj]
    return obj


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 配置并自动替换环境变量；文件缺失时返回空字典。"""
    _load_dotenv()
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _recursive_resolve(raw) if raw else {}


# 全局配置实例（延迟加载，避免导入时副作用）
_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config
