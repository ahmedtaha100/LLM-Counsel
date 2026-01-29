import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from models.adapters.base import CompletionResult


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_completion() -> CompletionResult:
    return CompletionResult(
        text="This is a mock response.",
        tokens_in=10,
        tokens_out=20,
        latency_ms=100.0,
        cost_usd=0.001,
        model="mock-model",
    )


@pytest.fixture
def mock_openai_client(mock_completion: CompletionResult):
    with patch("models.adapters.openai.AsyncOpenAI") as mock:
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=mock_completion.text))]
        response.usage = MagicMock(
            prompt_tokens=mock_completion.tokens_in,
            completion_tokens=mock_completion.tokens_out,
        )
        client.chat.completions.create = AsyncMock(return_value=response)
        client.models.list = AsyncMock(return_value=[])
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_anthropic_client(mock_completion: CompletionResult):
    with patch("models.adapters.anthropic.AsyncAnthropic") as mock:
        client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text=mock_completion.text)]
        response.usage = MagicMock(
            input_tokens=mock_completion.tokens_in,
            output_tokens=mock_completion.tokens_out,
        )
        client.messages.create = AsyncMock(return_value=response)
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_together_client(mock_completion: CompletionResult):
    with patch("models.adapters.together.AsyncTogether") as mock:
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=mock_completion.text))]
        response.usage = MagicMock(
            prompt_tokens=mock_completion.tokens_in,
            completion_tokens=mock_completion.tokens_out,
        )
        client.chat.completions.create = AsyncMock(return_value=response)
        client.models.list = AsyncMock(return_value=[])
        mock.return_value = client
        yield mock


@pytest.fixture
def fake_embeddings():
    rng = np.random.default_rng(42)

    def generate(n: int = 1) -> np.ndarray:
        embeddings = rng.random((n, 384)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    return generate


@pytest.fixture
def mock_shared_encoder(fake_embeddings):
    from core import shared

    shared.reset_encoder()

    with patch("core.shared.SentenceTransformer") as mock:
        encoder = MagicMock()
        encoder.encode = MagicMock(
            side_effect=lambda texts: fake_embeddings(len(texts) if isinstance(texts, list) else 1)
        )
        mock.return_value = encoder
        yield mock

    shared.reset_encoder()


@pytest.fixture
def sample_queries() -> list[dict[str, Any]]:
    return [
        {"query": "Hello", "type": "simple", "expected_category": "simple"},
        {"query": "What is 2+2?", "type": "simple", "expected_category": "simple"},
        {"query": "Explain quantum computing", "type": "complex", "expected_category": "complex"},
        {
            "query": "Write a Python function to sort a list",
            "type": "code",
            "expected_category": "moderate",
        },
        {
            "query": "Write a creative story about a dragon",
            "type": "creative",
            "expected_category": "moderate",
        },
        {
            "query": "Compare TCP vs UDP protocols in depth",
            "type": "reasoning",
            "expected_category": "complex",
        },
    ]


@pytest.fixture
def test_client():
    with (
        patch("api.routes.query._router", None),
        patch("api.routes.query._panel", None),
        patch("api.routes.query._judge", None),
        patch("api.routes.query._dissent", None),
        patch("api.routes.query._cache", None),
    ):
        from api.main import app

        return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    with (
        patch("api.routes.query._router", None),
        patch("api.routes.query._panel", None),
        patch("api.routes.query._judge", None),
        patch("api.routes.query._dissent", None),
        patch("api.routes.query._cache", None),
    ):
        from api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


@pytest.fixture
def mock_registry():
    with patch("models.registry.get_registry") as mock:
        registry = MagicMock()
        registry.list_models.return_value = ["gpt-4o-mini", "claude-haiku", "mixtral-8x7b"]
        registry.get.return_value = MagicMock()
        registry.get_config.return_value = {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006,
            "capabilities": ["reasoning", "code"],
        }
        mock.return_value = registry
        yield registry
