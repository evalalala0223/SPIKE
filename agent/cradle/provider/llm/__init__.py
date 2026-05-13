from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "OpenAIProvider": ".openai",
    "LLMFactory": ".llm_factory",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    *_LAZY_EXPORTS.keys(),
]
