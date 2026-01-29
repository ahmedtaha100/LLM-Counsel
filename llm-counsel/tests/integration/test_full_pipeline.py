from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.cache import SemanticCache
from core.dissent import DissentDetector
from core.judge import AggregationStrategy, Judge
from core.panel import Panel
from core.router import Router
from models.adapters.base import CompletionResult


@pytest.fixture
def mock_encoder():
    from core import shared

    shared.reset_encoder()

    with patch("core.shared.SentenceTransformer") as mock:
        encoder = MagicMock()

        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384).astype(np.float32)

        encoder.encode = MagicMock(side_effect=mock_encode)
        mock.return_value = encoder
        yield encoder

    shared.reset_encoder()


@pytest.fixture
def mock_registry():
    with patch("models.registry.get_registry") as registry_mock:
        registry = MagicMock()
        registry.list_models.return_value = ["gpt-4o-mini", "claude-haiku"]

        def make_adapter(name):
            adapter = MagicMock()
            adapter.complete = AsyncMock(
                return_value=CompletionResult(
                    text=f"Response from {name}",
                    tokens_in=10,
                    tokens_out=20,
                    latency_ms=100.0,
                    cost_usd=0.001,
                    model=name,
                )
            )
            return adapter

        registry.get.side_effect = make_adapter
        registry_mock.return_value = registry
        yield registry


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_simple_query_flow(self, mock_encoder, mock_registry):
        with patch("models.classifier.ComplexityClassifier") as mock_classifier:
            classifier = MagicMock()
            classifier.classify.return_value = MagicMock(
                score=2.0,
                category="simple",
                capabilities=["reasoning"],
                confidence=0.9,
                estimated_tokens=20,
            )
            mock_classifier.return_value = classifier

            router = Router()
            panel = Panel()
            judge = Judge()

            decision = router.route("What is 2+2?")
            assert decision.model in ["gpt-4o-mini", "claude-haiku"]

            response = await panel.single(decision.model, "What is 2+2?")
            assert response.success

            judgment = judge.judge_single(response)
            assert len(judgment.response) > 0

    @pytest.mark.asyncio
    async def test_panel_deliberation_flow(self, mock_encoder, mock_registry):
        with patch("models.classifier.ComplexityClassifier") as mock_classifier:
            classifier = MagicMock()
            classifier.classify.return_value = MagicMock(
                score=7.0,
                category="complex",
                capabilities=["reasoning"],
                confidence=0.6,
                estimated_tokens=200,
            )
            mock_classifier.return_value = classifier

            router = Router()
            panel = Panel()
            judge = Judge()
            dissent = DissentDetector()

            decision = router.route("Complex analysis question")

            models = decision.panel_models or ["gpt-4o-mini", "claude-haiku"]
            result = await panel.deliberate(models, "Complex question")

            assert len(result.successful_models) > 0

            judgment = judge.judge(result, AggregationStrategy.BEST_OF_N)
            assert judgment.selected_model in models

            report = dissent.detect(judgment.all_responses)
            assert report.dissent_level in ["none", "low", "moderate", "high"]

    @pytest.mark.asyncio
    async def test_cache_integration(self, mock_encoder, mock_registry):
        def deterministic_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            results = []
            for text in texts:
                seed = sum(ord(c) for c in text)
                rng = np.random.default_rng(seed)
                emb = rng.random(384).astype(np.float32)
                results.append(emb / np.linalg.norm(emb))
            return np.array(results)

        mock_encoder.encode.side_effect = deterministic_encode

        cache = SemanticCache(similarity_threshold=0.9)
        panel = Panel()
        judge = Judge()

        hit = cache.get("What is 2+2?")
        assert hit is None

        response = await panel.single("gpt-4o-mini", "What is 2+2?")
        judgment = judge.judge_single(response)

        cache.put("What is 2+2?", judgment.response, judgment.selected_model)

        hit = cache.get("What is 2+2?")
        assert hit is not None
        assert hit.response == judgment.response

    @pytest.mark.asyncio
    async def test_routing_respects_complexity(self, mock_encoder, mock_registry):
        with patch("models.classifier.ComplexityClassifier") as mock_classifier:
            classifier = MagicMock()

            call_count = [0]

            def classify_with_different_results(query):
                call_count[0] += 1
                if call_count[0] == 1:
                    return MagicMock(
                        score=2.0,
                        category="simple",
                        capabilities=["reasoning"],
                        confidence=0.9,
                        estimated_tokens=20,
                    )
                else:
                    return MagicMock(
                        score=8.0,
                        category="complex",
                        capabilities=["reasoning", "code"],
                        confidence=0.6,
                        estimated_tokens=500,
                    )

            classifier.classify.side_effect = classify_with_different_results
            mock_classifier.return_value = classifier

            router = Router()

            simple = router.route("Hello")
            complex_q = router.route("Design a distributed system for real-time data processing")

            assert simple.complexity_score <= complex_q.complexity_score

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, mock_encoder, mock_registry):
        mock_registry.get.side_effect = KeyError("Model not found")

        panel = Panel()
        result = await panel.single("nonexistent-model", "Test query")

        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_partial_panel_failure(self, mock_encoder, mock_registry):
        def flaky_adapter(name):
            adapter = MagicMock()
            if name == "failing-model":
                adapter.complete = AsyncMock(side_effect=Exception("API Error"))
            else:
                adapter.complete = AsyncMock(
                    return_value=CompletionResult(
                        text=f"Response from {name}",
                        tokens_in=10,
                        tokens_out=20,
                        latency_ms=100.0,
                        cost_usd=0.001,
                        model=name,
                    )
                )
            return adapter

        mock_registry.get.side_effect = flaky_adapter
        mock_registry.list_models.return_value = ["good-model", "failing-model"]

        panel = Panel()
        judge = Judge()

        result = await panel.deliberate(["good-model", "failing-model"], "Test")

        assert "good-model" in result.successful_models
        assert "failing-model" in result.failed_models

        judgment = judge.judge(result, AggregationStrategy.BEST_OF_N)
        assert judgment.selected_model == "good-model"
