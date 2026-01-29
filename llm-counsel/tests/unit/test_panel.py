import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.panel import Panel, PanelResponse, PanelResult
from models.adapters.base import CompletionResult


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.complete = AsyncMock(
        return_value=CompletionResult(
            text="Mock response",
            tokens_in=10,
            tokens_out=20,
            latency_ms=100.0,
            cost_usd=0.001,
            model="mock-model",
        )
    )
    return adapter


@pytest.fixture
def mock_registry(mock_adapter):
    with patch("models.registry.get_registry") as mock:
        registry = MagicMock()
        registry.get.return_value = mock_adapter
        mock.return_value = registry
        yield registry


class TestPanel:
    @pytest.mark.asyncio
    async def test_deliberate_single_model(self, mock_registry):
        panel = Panel()
        result = await panel.deliberate(["model-a"], "Test prompt")

        assert isinstance(result, PanelResult)
        assert len(result.responses) == 1
        assert result.successful_models == ["model-a"]

    @pytest.mark.asyncio
    async def test_deliberate_multiple_models(self, mock_registry):
        panel = Panel()
        result = await panel.deliberate(
            ["model-a", "model-b", "model-c"],
            "Test prompt",
        )

        assert len(result.responses) == 3
        assert len(result.successful_models) == 3
        assert result.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_deliberate_empty_models(self, mock_registry):
        panel = Panel()
        result = await panel.deliberate([], "Test prompt")

        assert len(result.responses) == 0
        assert result.total_cost_usd == 0

    @pytest.mark.asyncio
    async def test_deliberate_with_timeout(self, mock_registry, mock_adapter):
        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(10)
            return CompletionResult(
                text="Response",
                tokens_in=10,
                tokens_out=20,
                latency_ms=10000,
                cost_usd=0.001,
                model="slow-model",
            )

        mock_adapter.complete = slow_complete
        panel = Panel(timeout_seconds=0.1)
        result = await panel.deliberate(["slow-model"], "Test prompt")

        assert len(result.failed_models) == 1
        assert "slow-model" in result.failed_models

    @pytest.mark.asyncio
    async def test_deliberate_partial_failure(self, mock_registry):
        def get_adapter(name):
            adapter = MagicMock()
            if name == "failing-model":
                adapter.complete = AsyncMock(side_effect=Exception("API Error"))
            else:
                adapter.complete = AsyncMock(
                    return_value=CompletionResult(
                        text="Success",
                        tokens_in=10,
                        tokens_out=20,
                        latency_ms=100,
                        cost_usd=0.001,
                        model=name,
                    )
                )
            return adapter

        mock_registry.get.side_effect = get_adapter
        panel = Panel()
        result = await panel.deliberate(
            ["good-model", "failing-model"],
            "Test prompt",
        )

        assert "good-model" in result.successful_models
        assert "failing-model" in result.failed_models

    @pytest.mark.asyncio
    async def test_single_success(self, mock_registry):
        panel = Panel()
        result = await panel.single("test-model", "Test prompt")

        assert isinstance(result, PanelResponse)
        assert result.success
        assert result.result is not None

    @pytest.mark.asyncio
    async def test_single_failure(self, mock_registry, mock_adapter):
        mock_adapter.complete = AsyncMock(side_effect=Exception("Error"))
        panel = Panel()
        result = await panel.single("test-model", "Test prompt")

        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, mock_registry, mock_adapter):
        call_times = []

        async def tracked_complete(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return CompletionResult(
                text="Response",
                tokens_in=10,
                tokens_out=20,
                latency_ms=50,
                cost_usd=0.001,
                model="test",
            )

        mock_adapter.complete = tracked_complete
        panel = Panel()
        await panel.deliberate(["m1", "m2", "m3"], "Test prompt")

        if len(call_times) >= 2:
            time_diff = max(call_times) - min(call_times)
            assert time_diff < 0.1
