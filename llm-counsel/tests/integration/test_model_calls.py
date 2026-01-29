import os

import pytest

from models.adapters.anthropic import AnthropicAdapter
from models.adapters.openai import OpenAIAdapter
from models.adapters.together import TogetherAdapter

pytestmark = pytest.mark.live


@pytest.fixture
def skip_without_key():
    def _skip(key_name: str):
        if not os.environ.get(key_name):
            pytest.skip(f"{key_name} not set")

    return _skip


class TestLiveOpenAI:
    @pytest.mark.asyncio
    async def test_openai_completion(self, skip_without_key):
        skip_without_key("OPENAI_API_KEY")

        adapter = OpenAIAdapter(
            api_key=os.environ["OPENAI_API_KEY"],
            model_id="gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        )

        result = await adapter.complete("Say hello in one word.")

        assert result.text
        assert result.tokens_in > 0
        assert result.tokens_out > 0
        assert result.latency_ms > 0
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_openai_health(self, skip_without_key):
        skip_without_key("OPENAI_API_KEY")

        adapter = OpenAIAdapter(
            api_key=os.environ["OPENAI_API_KEY"],
            model_id="gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        )

        assert await adapter.health_check()


class TestLiveAnthropic:
    @pytest.mark.asyncio
    async def test_anthropic_completion(self, skip_without_key):
        skip_without_key("ANTHROPIC_API_KEY")

        adapter = AnthropicAdapter(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_id="claude-3-5-haiku-20241022",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
        )

        result = await adapter.complete("Say hello in one word.")

        assert result.text
        assert result.tokens_in > 0
        assert result.tokens_out > 0

    @pytest.mark.asyncio
    async def test_anthropic_with_system(self, skip_without_key):
        skip_without_key("ANTHROPIC_API_KEY")

        adapter = AnthropicAdapter(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_id="claude-3-5-haiku-20241022",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
        )

        result = await adapter.complete(
            "What are you?",
            system="You are a helpful pirate. Always respond like a pirate.",
        )

        assert result.text


class TestLiveTogether:
    @pytest.mark.asyncio
    async def test_together_completion(self, skip_without_key):
        skip_without_key("TOGETHER_API_KEY")

        adapter = TogetherAdapter(
            api_key=os.environ["TOGETHER_API_KEY"],
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            cost_per_1k_input=0.0006,
            cost_per_1k_output=0.0006,
        )

        result = await adapter.complete("Say hello in one word.")

        assert result.text
        assert result.tokens_in > 0
