from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_all_adapters():
    from core import shared

    shared.reset_encoder()

    with (
        patch("models.adapters.openai.AsyncOpenAI") as openai_mock,
        patch("models.adapters.anthropic.AsyncAnthropic") as anthropic_mock,
        patch("models.adapters.together.AsyncTogether") as together_mock,
        patch("core.shared.SentenceTransformer") as encoder_mock,
        patch("config.settings.settings.openai_api_key", "fake-openai-key"),
        patch("config.settings.settings.anthropic_api_key", "fake-anthropic-key"),
        patch("config.settings.settings.together_api_key", "fake-together-key"),
    ):

        def make_client(text):
            client = MagicMock()
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content=text))]
            response.usage = MagicMock(
                prompt_tokens=10,
                completion_tokens=20,
                input_tokens=10,
                output_tokens=20,
            )
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
                return np.random.rand(len(texts), 384).astype(np.float32)

            encoder.encode = MagicMock(side_effect=mock_encode)
            return encoder

        encoder_mock.return_value = make_encoder()

        yield

    shared.reset_encoder()


@pytest.fixture
def client(mock_all_adapters):
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

        return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "models_available" in data


class TestQueryEndpoint:
    def test_query_basic(self, client):
        response = client.post("/query", json={"query": "Hello"})
        assert response.status_code == 200

    def test_query_response_schema(self, client):
        response = client.post("/query", json={"query": "What is 2+2?"})
        data = response.json()

        assert "response" in data
        assert "model_used" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "cost_usd" in data

    def test_query_single_mode(self, client):
        response = client.post(
            "/query",
            json={
                "query": "Hello",
                "mode": "single",
            },
        )
        assert response.status_code == 200

    def test_query_panel_mode(self, client):
        response = client.post(
            "/query",
            json={
                "query": "Complex question",
                "mode": "panel",
            },
        )
        assert response.status_code == 200

    def test_query_with_parameters(self, client):
        response = client.post(
            "/query",
            json={
                "query": "Hello",
                "max_tokens": 1000,
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200

    def test_query_empty_fails(self, client):
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_query_budget_mode(self, client):
        response = client.post(
            "/query",
            json={
                "query": "Hello",
                "budget_mode": True,
            },
        )
        assert response.status_code == 200


class TestAnalyticsEndpoint:
    def test_analytics_returns_200(self, client):
        response = client.get("/analytics")
        assert response.status_code == 200

    def test_analytics_schema(self, client):
        response = client.get("/analytics")
        data = response.json()

        assert "cache" in data
        assert "total_queries" in data
        assert "latency" in data
        assert "cost" in data
        assert "queries_by_model" in data
        assert "selections_by_model" in data
        assert "panel_usage_rate" in data

    def test_analytics_latency_fields(self, client):
        response = client.get("/analytics")
        data = response.json()
        latency = data["latency"]

        assert "avg_ms" in latency
        assert "p50_ms" in latency
        assert "p95_ms" in latency
        assert "p99_ms" in latency

    def test_analytics_cost_fields(self, client):
        response = client.get("/analytics")
        data = response.json()
        cost = data["cost"]

        assert "total_cost" in cost
        assert "cost_by_model" in cost
        assert "estimated_gpt4o_cost" in cost
        assert "cost_savings" in cost

    def test_cache_stats(self, client):
        response = client.get("/analytics/cache")
        data = response.json()

        assert "hits" in data
        assert "misses" in data
        assert "entries" in data
        assert "hit_rate" in data

    def test_cache_clear(self, client):
        response = client.post("/analytics/cache/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"

    def test_metrics_clear(self, client):
        response = client.post("/analytics/metrics/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"

    def test_recent_queries_endpoint(self, client):
        client.post("/query", json={"query": "Test query 1"})
        client.post("/query", json={"query": "Test query 2"})

        response = client.get("/analytics/recent?n=10")
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) >= 2

    def test_recent_queries_schema(self, client):
        client.post("/query", json={"query": "Schema test query"})

        response = client.get("/analytics/recent?n=1")
        data = response.json()

        assert len(data) >= 1
        recent = data[-1]
        assert "query" in recent
        assert "models" in recent
        assert "selected_model" in recent
        assert "latency_ms" in recent
        assert "total_cost_usd" in recent
        assert "cost_by_model" in recent
        assert "cache_hit" in recent
        assert "panel_mode" in recent

    def test_selections_by_model_tracked(self, client):
        client.post("/analytics/metrics/clear")

        client.post("/query", json={"query": "First test"})
        client.post("/query", json={"query": "Second test"})

        response = client.get("/analytics")
        data = response.json()

        assert data["total_queries"] >= 2
        assert len(data["selections_by_model"]) >= 1

    def test_failures_by_model_in_schema(self, client):
        response = client.get("/analytics")
        data = response.json()

        assert "failures_by_model" in data

    def test_recent_queries_include_failed_models(self, client):
        client.post("/analytics/metrics/clear")
        client.post("/query", json={"query": "Test query"})

        response = client.get("/analytics/recent?n=10")
        data = response.json()

        assert len(data) >= 1
        assert "failed_models" in data[0]


@pytest.fixture
def failing_client():
    from core import shared

    shared.reset_encoder()

    with (
        patch("models.adapters.openai.AsyncOpenAI") as openai_mock,
        patch("models.adapters.anthropic.AsyncAnthropic") as anthropic_mock,
        patch("models.adapters.together.AsyncTogether") as together_mock,
        patch("core.shared.SentenceTransformer") as encoder_mock,
        patch("config.settings.settings.openai_api_key", "fake-openai-key"),
        patch("config.settings.settings.anthropic_api_key", "fake-anthropic-key"),
        patch("config.settings.settings.together_api_key", "fake-together-key"),
    ):

        def make_failing_client():
            client = MagicMock()
            client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )
            client.messages.create = AsyncMock(side_effect=Exception("API Error"))
            client.models.list = AsyncMock(return_value=[])
            return client

        openai_mock.return_value = make_failing_client()
        anthropic_mock.return_value = make_failing_client()
        together_mock.return_value = make_failing_client()

        def make_encoder():
            encoder = MagicMock()

            def mock_encode(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.rand(len(texts), 384).astype(np.float32)

            encoder.encode = MagicMock(side_effect=mock_encode)
            return encoder

        encoder_mock.return_value = make_encoder()

        with (
            patch("api.routes.query._router", None),
            patch("api.routes.query._panel", None),
            patch("api.routes.query._judge", None),
            patch("api.routes.query._dissent", None),
            patch("api.routes.query._cache", None),
            patch("models.registry._registry", None),
            patch("core.shared._encoder", None),
            patch("core.metrics._metrics", None),
        ):
            from api.main import app

            yield TestClient(app)

    shared.reset_encoder()


class TestFailureTracking:
    def test_failures_tracked_in_analytics(self, failing_client):
        failing_client.post("/analytics/metrics/clear")

        failing_client.post("/query", json={"query": "This will fail"})

        response = failing_client.get("/analytics")
        data = response.json()

        assert data["total_queries"] == 1
        assert len(data["failures_by_model"]) >= 1
        assert len(data["selections_by_model"]) == 0

    def test_failed_models_in_recent(self, failing_client):
        failing_client.post("/analytics/metrics/clear")

        failing_client.post("/query", json={"query": "Failing query"})

        response = failing_client.get("/analytics/recent?n=10")
        data = response.json()

        assert len(data) == 1
        assert len(data[0]["failed_models"]) >= 1

    def test_failures_not_cached(self, failing_client):
        failing_client.post("/analytics/cache/clear")

        failing_client.post("/query", json={"query": "Cache test fail"})
        failing_client.post("/query", json={"query": "Cache test fail"})

        response = failing_client.get("/analytics/cache")
        data = response.json()

        assert data["hits"] == 0
