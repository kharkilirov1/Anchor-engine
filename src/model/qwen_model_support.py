from __future__ import annotations

from typing import Any


def get_text_config(config: Any) -> Any:
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def get_config_attr(
    config: Any,
    name: str,
    default: Any | None = None,
) -> Any:
    text_config = get_text_config(config)
    if hasattr(text_config, name):
        return getattr(text_config, name)
    if hasattr(config, name):
        return getattr(config, name)
    if default is not None:
        return default
    raise AttributeError(f"config does not define `{name}`")


def is_conditional_generation_config(config: Any) -> bool:
    model_type = str(getattr(config, "model_type", ""))
    architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
    if "ConditionalGeneration" in "".join(architectures):
        return True
    if model_type == "qwen3_5" and getattr(config, "text_config", None) is not None:
        return True
    return False
