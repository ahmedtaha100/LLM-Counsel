from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.adapters.anthropic import AnthropicAdapter
from models.adapters.base import BaseAdapter, CompletionResult
from models.adapters.openai import OpenAIAdapter
from models.adapters.together import TogetherAdapter


class TestCompletionResult:
    def test_fields(self):
        result = CompletionResult(
            text="Hello",
            tokens_in=10,
            tokens_out=5,
            latency_ms=100.0,
            cost_usd=0.001,
            model="test-model",
        )

        assert result.text == "Hello"
        assert result.tokens_in == 10
        assert result.tokens_out == 5
        assert result.latency_ms == 100.0
        assert result.cost_usd == 0.001
        assert result.model == "test-model"


class TestOpenAIAdapter:
    @pytest.fixture
    def mock_client(self):
        with patch("models.adapters.openai.AsyncOpenAI") as mock:
            client = MagicMock()
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            client.chat.completions.create = AsyncMock(return_value=response)
            client.models.list = AsyncMock(return_value=[])
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_complete(self, mock_client):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )

        result = await adapter.complete("Hello")

        assert isinstance(result, CompletionResult)
        assert result.text == "Test response"
        assert result.tokens_in == 10
        assert result.tokens_out == 20

    @pytest.mark.asyncio
    async def test_complete_with_system(self, mock_client):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )

        await adapter.complete("Hello", system="Be helpful")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_cost_calculation(self, mock_client):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )

        result = await adapter.complete("Hello")
        expected_cost = (10 / 1000) * 0.005 + (20 / 1000) * 0.015
        assert abs(result.cost_usd - expected_cost) < 0.0001

    @pytest.mark.asyncio
    async def test_health_check(self, mock_client):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )

        result = await adapter.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_client):
        mock_client.models.list = AsyncMock(side_effect=Exception("API Error"))
        adapter = OpenAIAdapter(
            api_key="test-key",
            model_id="gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )

        result = await adapter.health_check()
        assert result is False


class TestAnthropicAdapter:
    @pytest.fixture
    def mock_client(self):
        with patch("models.adapters.anthropic.AsyncAnthropic") as mock:
            client = MagicMock()
            response = MagicMock()
            response.content = [MagicMock(text="Test response")]
            response.usage = MagicMock(input_tokens=10, output_tokens=20)
            client.messages.create = AsyncMock(return_value=response)
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_complete(self, mock_client):
        adapter = AnthropicAdapter(
            api_key="test-key",
            model_id="claude-sonnet",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

        result = await adapter.complete("Hello")

        assert result.text == "Test response"
        assert result.tokens_in == 10
        assert result.tokens_out == 20

    @pytest.mark.asyncio
    async def test_default_system_prompt(self, mock_client):
        adapter = AnthropicAdapter(
            api_key="test-key",
            model_id="claude-sonnet",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

        await adapter.complete("Hello")

        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs


class TestTogetherAdapter:
    @pytest.fixture
    def mock_client(self):
        with patch("models.adapters.together.AsyncTogether") as mock:
            client = MagicMock()
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="Test response"))]
            response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            client.chat.completions.create = AsyncMock(return_value=response)
            client.models.list = AsyncMock(return_value=[])
            mock.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_complete(self, mock_client):
        adapter = TogetherAdapter(
            api_key="test-key",
            model_id="llama-3-70b",
            cost_per_1k_input=0.0009,
            cost_per_1k_output=0.0009,
        )

        result = await adapter.complete("Hello")

        assert result.text == "Test response"
        assert result.model == "llama-3-70b"


class TestBaseAdapterCostCalculation:
    def test_calculate_cost(self):
        class TestAdapter(BaseAdapter):
            async def complete(self, prompt, max_tokens=4096, temperature=0.7, system=None):
                pass

            async def health_check(self):
                pass

        adapter = TestAdapter(
            api_key="test",
            model_id="test",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
        )

        cost = adapter.calculate_cost(1000, 500)
        expected = 0.01 + 0.01
        assert abs(cost - expected) < 0.0001
