from unittest.mock import MagicMock, patch

import pytest

from core.router import Router, RoutingDecision


@pytest.fixture
def mock_classifier():
    from core import shared

    shared.reset_encoder()

    with patch("models.classifier.ComplexityClassifier") as mock:
        classifier = MagicMock()
        classifier.classify.return_value = MagicMock(
            score=5.0,
            category="moderate",
            capabilities=["reasoning"],
            confidence=0.8,
            estimated_tokens=100,
        )
        mock.return_value = classifier
        yield classifier

    shared.reset_encoder()


@pytest.fixture
def mock_registry():
    with patch("models.registry.get_registry") as mock:
        registry = MagicMock()
        registry.list_models.return_value = ["gpt-4o", "gpt-4o-mini", "claude-haiku"]
        mock.return_value = registry
        yield registry


class TestRouter:
    def test_route_simple_query(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=2.0,
            category="simple",
            capabilities=["reasoning"],
            confidence=0.9,
            estimated_tokens=20,
        )
        router = Router()
        decision = router.route("Hello")

        assert isinstance(decision, RoutingDecision)
        assert decision.category == "simple"
        assert not decision.use_panel

    def test_route_complex_query(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=8.0,
            category="complex",
            capabilities=["reasoning", "code"],
            confidence=0.6,
            estimated_tokens=500,
        )
        router = Router()
        decision = router.route("Design a distributed system")

        assert decision.category == "complex"
        assert decision.use_panel

    def test_route_high_stakes_triggers_panel(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=3.0,
            category="simple",
            capabilities=["reasoning"],
            confidence=0.95,
            estimated_tokens=50,
        )
        router = Router()
        decision = router.route("I need medical advice about my symptoms")

        assert decision.use_panel
        assert "high-stakes" in decision.reason.lower()

    def test_route_low_confidence_triggers_panel(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=4.0,
            category="moderate",
            capabilities=["reasoning"],
            confidence=0.5,
            estimated_tokens=100,
        )
        router = Router()
        decision = router.route("Some ambiguous query")

        assert decision.use_panel
        assert "confidence" in decision.reason.lower()

    def test_route_budget_mode(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=5.0,
            category="moderate",
            capabilities=["code"],
            confidence=0.8,
            estimated_tokens=200,
        )
        router = Router()
        decision = router.route("Write a function", prefer_budget=True)

        assert "budget" in decision.reason.lower()

    def test_route_empty_query(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=1.0,
            category="simple",
            capabilities=["reasoning"],
            confidence=1.0,
            estimated_tokens=10,
        )
        router = Router()
        decision = router.route("")

        assert decision.category == "simple"

    def test_route_very_long_query(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=7.0,
            category="complex",
            capabilities=["reasoning", "analysis"],
            confidence=0.7,
            estimated_tokens=5000,
        )
        router = Router()
        long_query = "Explain this topic " * 1000
        decision = router.route(long_query)

        assert isinstance(decision, RoutingDecision)

    def test_route_non_english(self, mock_classifier, mock_registry):
        mock_classifier.classify.return_value = MagicMock(
            score=3.0,
            category="simple",
            capabilities=["reasoning"],
            confidence=0.7,
            estimated_tokens=50,
        )
        router = Router()
        decision = router.route("你好世界")

        assert isinstance(decision, RoutingDecision)

    def test_routing_decision_fields(self, mock_classifier, mock_registry):
        router = Router()
        decision = router.route("Test query")

        assert hasattr(decision, "model")
        assert hasattr(decision, "complexity_score")
        assert hasattr(decision, "category")
        assert hasattr(decision, "capabilities")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "use_panel")
        assert hasattr(decision, "panel_models")
        assert hasattr(decision, "reason")
