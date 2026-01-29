from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.judge import AggregationStrategy, Judge, JudgmentResult
from core.panel import PanelResponse, PanelResult
from models.adapters.base import CompletionResult


def make_response(model: str, text: str, latency: float = 100.0) -> PanelResponse:
    return PanelResponse(
        model=model,
        result=CompletionResult(
            text=text,
            tokens_in=10,
            tokens_out=len(text.split()),
            latency_ms=latency,
            cost_usd=0.001,
            model=model,
        ),
    )


def make_panel_result(responses: list[PanelResponse]) -> PanelResult:
    successful = [r.model for r in responses if r.success]
    failed = [r.model for r in responses if not r.success]
    total_cost = sum(r.result.cost_usd for r in responses if r.result)
    return PanelResult(
        responses=responses,
        total_latency_ms=max((r.result.latency_ms for r in responses if r.result), default=0),
        total_cost_usd=total_cost,
        successful_models=successful,
        failed_models=failed,
    )


@pytest.fixture
def mock_encoder():
    from core import shared

    shared.reset_encoder()

    with patch("core.shared.SentenceTransformer") as mock:
        encoder = MagicMock()

        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            return np.random.rand(n, 384).astype(np.float32)

        encoder.encode = MagicMock(side_effect=mock_encode)
        mock.return_value = encoder
        yield encoder

    shared.reset_encoder()


class TestJudge:
    def test_judge_single_response(self, mock_encoder):
        judge = Judge()
        response = make_response("model-a", "This is a detailed response with structure.")
        result = judge.judge_single(response)

        assert isinstance(result, JudgmentResult)
        assert result.selected_model == "model-a"
        assert result.strategy == "single"

    def test_judge_single_failed(self, mock_encoder):
        judge = Judge()
        response = PanelResponse(
            model="failed-model",
            result=None,
            error="API Error",
            success=False,
        )
        result = judge.judge_single(response)

        assert result.confidence == 0.0
        assert "failed" in result.reasoning.lower()

    def test_judge_best_of_n(self, mock_encoder):
        judge = Judge()
        responses = [
            make_response("model-a", "Short answer"),
            make_response(
                "model-b",
                "This is a much more detailed and structured response. First, we have the introduction. Second, the main content. Finally, a conclusion.",
            ),
            make_response("model-c", "Medium length response here with some detail"),
        ]
        panel_result = make_panel_result(responses)
        result = judge.judge(panel_result, AggregationStrategy.BEST_OF_N)

        assert isinstance(result, JudgmentResult)
        assert result.strategy == "best_of_n"

    def test_judge_majority_vote(self, mock_encoder):
        def mock_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            return np.array(
                [
                    [1.0, 0.0, 0.0] + [0.0] * 381,
                    [0.99, 0.01, 0.0] + [0.0] * 381,
                    [0.5, 0.5, 0.0] + [0.0] * 381,
                ][: len(texts)],
                dtype=np.float32,
            )

        mock_encoder.encode.side_effect = mock_encode

        judge = Judge()
        responses = [
            make_response("model-a", "Paris is the capital"),
            make_response("model-b", "The capital is Paris"),
            make_response("model-c", "London is the capital"),
        ]
        panel_result = make_panel_result(responses)
        result = judge.judge(panel_result, AggregationStrategy.MAJORITY_VOTE)

        assert result.strategy == "majority_vote"

    def test_judge_confidence_weighted(self, mock_encoder):
        judge = Judge()
        responses = [
            make_response("model-a", "Fast but short", latency=50.0),
            make_response(
                "model-b", "Slow but detailed response with good structure", latency=500.0
            ),
        ]
        panel_result = make_panel_result(responses)
        result = judge.judge(panel_result, AggregationStrategy.CONFIDENCE_WEIGHTED)

        assert result.strategy == "confidence_weighted"
        assert result.confidence > 0

    def test_judge_empty_panel(self, mock_encoder):
        judge = Judge()
        panel_result = PanelResult(
            responses=[],
            total_latency_ms=0,
            total_cost_usd=0,
            successful_models=[],
            failed_models=[],
        )
        result = judge.judge(panel_result, AggregationStrategy.BEST_OF_N)

        assert "no successful" in result.response.lower()
        assert result.confidence == 0.0

    def test_judge_all_failed(self, mock_encoder):
        judge = Judge()
        responses = [
            PanelResponse(model="m1", result=None, error="Error", success=False),
            PanelResponse(model="m2", result=None, error="Error", success=False),
        ]
        panel_result = make_panel_result(responses)
        result = judge.judge(panel_result, AggregationStrategy.BEST_OF_N)

        assert result.confidence == 0.0
        assert result.all_responses == {}

    def test_score_response_length(self, mock_encoder):
        judge = Judge()
        short = judge._compute_response_length_score("Hi")
        medium = judge._compute_response_length_score(
            "This is a medium length response with about fifty words " * 2
        )
        long = judge._compute_response_length_score("Word " * 200)

        assert short < medium
        assert medium <= long

    def test_score_response_structure(self, mock_encoder):
        judge = Judge()
        flat = judge._compute_structure_score("Just a plain text response")
        structured = judge._compute_structure_score(
            "First, this.\nSecond, that.\nFinally, ```code```"
        )

        assert flat < structured

    @pytest.mark.asyncio
    async def test_judge_model_strategy_success(self, mock_encoder):
        with patch("models.registry.get_registry") as mock_registry:
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(
                return_value=CompletionResult(
                    text="1",
                    tokens_in=10,
                    tokens_out=1,
                    latency_ms=50,
                    cost_usd=0.001,
                    model="gpt-4o",
                )
            )
            registry = MagicMock()
            registry.get.return_value = mock_adapter
            mock_registry.return_value = registry

            judge = Judge()
            responses = [
                make_response("model-a", "First response"),
                make_response("model-b", "Second response"),
            ]
            panel_result = make_panel_result(responses)
            result = await judge.judge_async(
                panel_result, AggregationStrategy.JUDGE_MODEL, question="Test?"
            )

            assert result.strategy == "judge_model"
            assert result.selected_model == "model-a"
            assert "gpt-4o" in result.reasoning

    @pytest.mark.asyncio
    async def test_judge_model_strategy_fallback(self, mock_encoder):
        with patch("models.registry.get_registry") as mock_registry:
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(side_effect=Exception("API Error"))
            registry = MagicMock()
            registry.get.return_value = mock_adapter
            mock_registry.return_value = registry

            judge = Judge()
            responses = [
                make_response("model-a", "Short"),
                make_response(
                    "model-b", "This is a longer and better structured response with detail"
                ),
            ]
            panel_result = make_panel_result(responses)
            result = await judge.judge_async(
                panel_result, AggregationStrategy.JUDGE_MODEL, question="Test?"
            )

            assert result.strategy == "judge_model"
            assert "Fallback" in result.reasoning

    @pytest.mark.asyncio
    async def test_judge_model_parsing_exact_match(self, mock_encoder):
        with patch("models.registry.get_registry") as mock_registry:
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(
                return_value=CompletionResult(
                    text="2 is the best response",
                    tokens_in=10,
                    tokens_out=5,
                    latency_ms=50,
                    cost_usd=0.001,
                    model="gpt-4o",
                )
            )
            registry = MagicMock()
            registry.get.return_value = mock_adapter
            mock_registry.return_value = registry

            judge = Judge()
            responses = [
                make_response("model-a", "First response"),
                make_response("model-b", "Second response"),
                make_response("model-c", "Third response"),
            ]
            panel_result = make_panel_result(responses)
            result = await judge.judge_async(
                panel_result, AggregationStrategy.JUDGE_MODEL, question="Test?"
            )

            assert result.selected_model == "model-b"

    @pytest.mark.asyncio
    async def test_judge_model_invalid_choice_fallback(self, mock_encoder):
        with patch("models.registry.get_registry") as mock_registry:
            mock_adapter = MagicMock()
            mock_adapter.complete = AsyncMock(
                return_value=CompletionResult(
                    text="The best one is clearly number five",
                    tokens_in=10,
                    tokens_out=10,
                    latency_ms=50,
                    cost_usd=0.001,
                    model="gpt-4o",
                )
            )
            registry = MagicMock()
            registry.get.return_value = mock_adapter
            mock_registry.return_value = registry

            judge = Judge()
            responses = [
                make_response("model-a", "Short"),
                make_response("model-b", "Also short"),
            ]
            panel_result = make_panel_result(responses)
            result = await judge.judge_async(
                panel_result, AggregationStrategy.JUDGE_MODEL, question="Test?"
            )

            assert "Fallback" in result.reasoning

    def test_sync_judge_model_falls_back_to_best_of_n(self, mock_encoder):
        judge = Judge()
        responses = [
            make_response("model-a", "Short"),
            make_response(
                "model-b", "This is a longer and better structured response with detail"
            ),
        ]
        panel_result = make_panel_result(responses)
        result = judge.judge(
            panel_result, AggregationStrategy.JUDGE_MODEL, question="Test?"
        )

        assert result.strategy == "judge_model"
        assert "Fallback" in result.reasoning
        assert "judge_async" in result.reasoning
