from typing import Any

from config.settings import models_config, settings
from models.adapters import (
    AnthropicAdapter,
    BaseAdapter,
    OpenAIAdapter,
    TogetherAdapter,
)


class ModelRegistry:
    _instance: "ModelRegistry | None" = None

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}
        self._config = models_config["models"]
        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        provider_map = {
            "openai": (OpenAIAdapter, settings.openai_api_key),
            "anthropic": (AnthropicAdapter, settings.anthropic_api_key),
            "together": (TogetherAdapter, settings.together_api_key),
        }

        for name, cfg in self._config.items():
            provider = cfg["provider"]
            if provider not in provider_map:
                continue

            adapter_cls, api_key = provider_map[provider]
            if not api_key:
                continue

            self._adapters[name] = adapter_cls(
                api_key=api_key,
                model_id=cfg["model_id"],
                cost_per_1k_input=cfg["cost_per_1k_input"],
                cost_per_1k_output=cfg["cost_per_1k_output"],
            )

    def get(self, name: str) -> BaseAdapter:
        if name not in self._adapters:
            raise KeyError(f"Model '{name}' not found or not configured")
        return self._adapters[name]

    def get_config(self, name: str) -> dict[str, Any]:
        if name not in self._config:
            raise KeyError(f"Model config '{name}' not found")
        return self._config[name]

    def list_models(self) -> list[str]:
        return list(self._adapters.keys())

    def list_by_capability(self, capability: str) -> list[str]:
        return [
            name
            for name, cfg in self._config.items()
            if capability in cfg.get("capabilities", []) and name in self._adapters
        ]

    def cheapest_for_capability(self, capability: str) -> str | None:
        candidates = self.list_by_capability(capability)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda n: self._config[n]["cost_per_1k_input"]
            + self._config[n]["cost_per_1k_output"],
        )

    async def health_all(self) -> dict[str, bool]:
        results = {}
        for name, adapter in self._adapters.items():
            results[name] = await adapter.health_check()
        return results


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
