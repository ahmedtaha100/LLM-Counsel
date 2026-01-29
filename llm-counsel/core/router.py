from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.settings import models_config, routing_config

if TYPE_CHECKING:
    from core.shared import SharedEncoder
    from models.registry import ModelRegistry


@dataclass
class RoutingDecision:
    model: str
    complexity_score: float
    category: str
    capabilities: list[str]
    confidence: float
    use_panel: bool
    panel_models: list[str]
    reason: str


class Router:
    HIGH_STAKES_KEYWORDS = frozenset(routing_config["panel_triggers"]["high_stakes_keywords"])

    def __init__(
        self,
        registry: "ModelRegistry | None" = None,
        encoder: "SharedEncoder | None" = None,
    ):
        from core.shared import get_encoder
        from models.classifier import ComplexityClassifier
        from models.registry import get_registry

        self._registry = registry or get_registry()
        self._encoder = encoder or get_encoder()
        self._classifier = ComplexityClassifier(encoder=self._encoder)
        self._routing_rules = routing_config["routing"]
        self._capability_routing = routing_config["capability_routing"]
        self._confidence_threshold = routing_config["panel_triggers"]["confidence_threshold"]
        self._model_configs = models_config.get("models", {})

    def _is_high_stakes(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.HIGH_STAKES_KEYWORDS)

    def _get_model_cost(self, model_name: str) -> float:
        cfg = self._model_configs.get(model_name, {})
        return cfg.get("cost_per_1k_input", 999) + cfg.get("cost_per_1k_output", 999)

    def _get_capable_models(self, caps: list[str], require_all: bool = True) -> list[str]:
        available = self._registry.list_models()
        capable = []
        for model in available:
            cfg = self._model_configs.get(model, {})
            model_caps = set(cfg.get("capabilities", []))
            if require_all:
                if all(c in model_caps for c in caps):
                    capable.append(model)
            else:
                if any(c in model_caps for c in caps):
                    capable.append(model)
        if not capable and require_all:
            return self._get_capable_models(caps, require_all=False)
        return capable if capable else available

    def _select_cheapest_capable(self, caps: list[str]) -> tuple[str, str]:
        capable = self._get_capable_models(caps)
        if not capable:
            available = self._registry.list_models()
            return (available[0] if available else "gpt-4o-mini", "No capable models, using first")

        cheapest = min(capable, key=self._get_model_cost)
        cost = self._get_model_cost(cheapest)
        return cheapest, f"Cheapest capable model (${cost:.4f}/1k tokens)"

    def _model_has_all_caps(self, model: str, caps: list[str]) -> bool:
        cfg = self._model_configs.get(model, {})
        model_caps = set(cfg.get("capabilities", []))
        return all(c in model_caps for c in caps)

    def _select_model_for_capability(
        self, caps: list[str], budget: bool = False
    ) -> tuple[str, str]:
        if not caps:
            caps = ["reasoning"]

        if budget:
            return self._select_cheapest_capable(caps)

        for cap in caps:
            if cap in self._capability_routing:
                candidates = self._capability_routing[cap].get("preferred", [])
                for model in candidates:
                    if model in self._registry.list_models() and self._model_has_all_caps(
                        model, caps
                    ):
                        return model, f"Preferred model for {cap} (has all required: {caps})"

        capable = self._get_capable_models(caps)
        if capable:
            return capable[0], f"First capable model for {caps}"

        available = self._registry.list_models()
        return (available[0] if available else "gpt-4o-mini", "Fallback model")

    def _select_by_complexity(self, category: str) -> tuple[str, str]:
        rule = self._routing_rules.get(category, self._routing_rules["simple"])
        primary = rule["primary"]
        fallback = rule["fallback"]

        if primary in self._registry.list_models():
            return primary, f"Primary model for {category} complexity"
        if fallback in self._registry.list_models():
            return fallback, f"Fallback model for {category} complexity"

        available = self._registry.list_models()
        return (available[0] if available else "gpt-4o-mini", "Default fallback")

    def _get_panel_models(self, complexity_category: str) -> list[str]:
        panels = models_config.get("panels", {})

        if complexity_category == "complex":
            panel_key = "high_stakes"
        elif complexity_category == "moderate":
            panel_key = "balanced"
        else:
            panel_key = "budget"

        panel_models = panels.get(panel_key, [])
        available = self._registry.list_models()
        return [m for m in panel_models if m in available]

    def route(self, query: str, prefer_budget: bool = False) -> RoutingDecision:
        result = self._classifier.classify(query)
        high_stakes = self._is_high_stakes(query)
        low_confidence = result.confidence < self._confidence_threshold

        use_panel = high_stakes or low_confidence or result.category == "complex"

        if prefer_budget:
            model, reason = self._select_cheapest_capable(result.capabilities)
            reason = f"Budget mode: {reason}"
        elif result.capabilities:
            model, reason = self._select_model_for_capability(result.capabilities)
        else:
            model, reason = self._select_by_complexity(result.category)

        if high_stakes:
            reason = "High-stakes query detected, recommending panel"
        elif low_confidence:
            reason = f"Low confidence ({result.confidence:.2f}), recommending panel"

        panel_models = self._get_panel_models(result.category) if use_panel else []

        return RoutingDecision(
            model=model,
            complexity_score=result.score,
            category=result.category,
            capabilities=result.capabilities,
            confidence=result.confidence,
            use_panel=use_panel,
            panel_models=panel_models,
            reason=reason,
        )
