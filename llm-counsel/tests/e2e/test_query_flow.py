from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_all():
    from core import shared

    shared.reset_encoder()

    with (
        patch("models.adapters.openai.AsyncOpenAI") as openai_mock,
        patch("models.adapters.anthropic.AsyncAnthropic") as anthropic_mock,
        patch("models.adapters.together.AsyncTogether") as together_mock,
        patch("core.shared.SentenceTransformer") as encoder_mock,
    ):

        def make_client(text):
            client = MagicMock()
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content=text))]
            response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
            response.content = [MagicMock(text=text)]
            client.chat.completions.create = AsyncMock(return_value=response)
            client.messages.create = AsyncMock(return_value=response)
            client.models.list = AsyncMock(return_value=[])
            return client

        openai_mock.return_value = make_client("OpenAI response")
        anthropic_mock.return_value = make_client("Anthropic response")
        together_mock.return_value = make_client("Together response")

        def make_encoder():
            encoder = MagicMock()

            def mock_encode(texts):
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return np.random.rand(n, 384).astype(np.float32)

            encoder.encode = MagicMock(side_effect=mock_encode)
            return encoder

        encoder_mock.return_value = make_encoder()

        yield

    shared.reset_encoder()


@pytest.fixture
async def client(mock_all):
    with (
        patch("api.routes.query._router", None),
        patch("api.routes.query._panel", None),
        patch("api.routes.query._judge", None),
        patch("api.routes.query._dissent", None),
        patch("api.routes.query._cache", None),
        patch("models.registry._registry", None),
        patch("core.shared._encoder", None),
    ):
        from api.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac


class TestE2EQueryFlow:
    @pytest.mark.asyncio
    async def test_simple_query_e2e(self, client):
        response = await client.post(
            "/query",
            json={
                "query": "What is 2+2?",
                "mode": "single",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["model_used"]
        assert data["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_panel_query_e2e(self, client):
        response = await client.post(
            "/query",
            json={
                "query": "Explain quantum computing in detail",
                "mode": "panel",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"]

    @pytest.mark.asyncio
    async def test_auto_mode_e2e(self, client):
        response = await client.post(
            "/query",
            json={
                "query": "Design a scalable microservices architecture",
                "mode": "auto",
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_behavior_e2e(self, client):
        query = {"query": "What is the capital of France?", "mode": "single"}

        resp1 = await client.post("/query", json=query)
        assert resp1.status_code == 200
        data1 = resp1.json()

        resp2 = await client.post("/query", json=query)
        assert resp2.status_code == 200
        data2 = resp2.json()

        if data2.get("cache_hit"):
            assert data2["latency_ms"] < data1["latency_ms"]

    @pytest.mark.asyncio
    async def test_analytics_after_queries(self, client):
        await client.post("/query", json={"query": "Test query 1", "mode": "single"})
        await client.post("/query", json={"query": "Test query 2", "mode": "single"})

        response = await client.get("/analytics")
        assert response.status_code == 200
        data = response.json()

        assert data["total_queries"] >= 2

    @pytest.mark.asyncio
    async def test_health_endpoints_e2e(self, client):
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_full_workflow(self, client):
        health = await client.get("/health")
        assert health.status_code == 200

        query_resp = await client.post(
            "/query",
            json={
                "query": "What is machine learning?",
                "mode": "auto",
                "max_tokens": 500,
                "temperature": 0.7,
            },
        )
        assert query_resp.status_code == 200
        assert query_resp.json()["response"]

        analytics = await client.get("/analytics")
        assert analytics.status_code == 200

    @pytest.mark.asyncio
    async def test_budget_mode_e2e(self, client):
        response = await client.post(
            "/query",
            json={
                "query": "Explain relativity",
                "budget_mode": True,
            },
        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_validation_errors(self, client):
        response = await client.post("/query", json={"query": ""})
        assert response.status_code == 422

        response = await client.post(
            "/query",
            json={
                "query": "Test",
                "max_tokens": -1,
            },
        )
        assert response.status_code == 422

        response = await client.post(
            "/query",
            json={
                "query": "Test",
                "temperature": 5.0,
            },
        )
        assert response.status_code == 422
