import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.cache import CacheStats, SemanticCache


@pytest.fixture
def mock_encoder():
    from core import shared

    shared.reset_encoder()

    with patch("core.shared.SentenceTransformer") as mock:
        encoder = MagicMock()
        call_count = [0]

        def deterministic_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            result = []
            for text in texts:
                call_count[0] += 1
                seed = sum(ord(c) for c in text)
                rng = np.random.default_rng(seed)
                emb = rng.random(384).astype(np.float32)
                result.append(emb / np.linalg.norm(emb))
            return np.array(result)

        encoder.encode = MagicMock(side_effect=deterministic_encode)
        mock.return_value = encoder
        yield encoder

    shared.reset_encoder()


class TestSemanticCache:
    def test_put_and_get_exact(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("What is 2+2?", "4", "gpt-4")
        hit = cache.get("What is 2+2?")

        assert hit is not None
        assert hit.response == "4"
        assert hit.model == "gpt-4"
        assert hit.similarity >= 0.95

    def test_cache_miss(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("What is 2+2?", "4", "gpt-4")
        hit = cache.get("Explain quantum physics")

        assert hit is None

    def test_similar_query_hit(self, mock_encoder):
        def similar_encode(texts):
            if isinstance(texts, str):
                texts = [texts]
            result = []
            for _ in texts:
                base = np.array([1.0, 0.1, 0.05] + [0.0] * 381, dtype=np.float32)
                noise = np.random.rand(384).astype(np.float32) * 0.01
                emb = base + noise
                result.append(emb / np.linalg.norm(emb))
            return np.array(result)

        mock_encoder.encode.side_effect = similar_encode

        cache = SemanticCache(similarity_threshold=0.9)
        cache.put("What is the capital of France?", "Paris", "model-a")
        hit = cache.get("What's France's capital?")

        assert hit is not None

    def test_ttl_expiration(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.9, ttl_seconds=1)
        cache.put("test query", "test response", "model-a")

        hit1 = cache.get("test query")
        assert hit1 is not None

        time.sleep(1.1)
        hit2 = cache.get("test query")
        assert hit2 is None

    def test_stats_tracking(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("query1", "response1", "model")

        cache.get("query1")
        cache.get("query2")
        cache.get("query3")

        stats = cache.stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits + stats.misses == 3
        assert stats.entries == 1

    def test_hit_rate_calculation(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("q1", "r1", "m1")

        for _ in range(3):
            cache.get("q1")
        for _ in range(7):
            cache.get("different query")

        stats = cache.stats()
        assert 0.25 <= stats.hit_rate <= 0.35

    def test_clear(self, mock_encoder):
        cache = SemanticCache()
        cache.put("q1", "r1", "m1")
        cache.put("q2", "r2", "m2")
        cache.get("q1")

        cache.clear()
        stats = cache.stats()

        assert stats.entries == 0
        assert stats.hits == 0
        assert stats.misses == 0

    def test_max_entries_eviction(self, mock_encoder):
        cache = SemanticCache(max_entries=10)

        for i in range(15):
            cache.put(f"query-{i}", f"response-{i}", "model")

        stats = cache.stats()
        assert stats.entries <= 10

    def test_metadata_stored(self, mock_encoder):
        cache = SemanticCache()
        cache.put("query", "response", "model", metadata={"key": "value"})

        assert len(cache._entries) == 1
        assert cache._entries[0].metadata == {"key": "value"}

    def test_cache_hit_age(self, mock_encoder):
        cache = SemanticCache(similarity_threshold=0.9)
        cache.put("query", "response", "model")
        time.sleep(0.1)
        hit = cache.get("query")

        assert hit is not None
        assert hit.age_seconds >= 0.1
