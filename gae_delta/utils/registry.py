"""Component registry for dynamic model/data/transform resolution."""
from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    """Simple registry for mapping string names to classes/functions."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """Decorator to register a component."""
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"'{name}' already registered in {self._name} registry."
                )
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name} registry. "
                f"Available: {list(self._registry)}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry


MODEL_REGISTRY = Registry("model")
TRANSFORM_REGISTRY = Registry("transform")
