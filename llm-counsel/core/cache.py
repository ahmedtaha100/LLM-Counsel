import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from config.settings import settings

if TYPE_CHECKING:
    from core.shared import SharedEncoder

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class CacheEntry:
    query: str
    response: str
    model: str
    embedding: np.ndarray
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheHit:
    query: str
    response: str
    model: str
    similarity: float
    age_seconds: float


@dataclass
class CacheStats:
    hits: int
    misses: int
    entries: int
    hit_rate: float


class SemanticCache:
    def __init__(
        self,
        encoder: "SharedEncoder | None" = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_entries: int = 10000,
    ):
        from core.shared import get_encoder

        self._encoder = encoder or get_encoder()
        self._threshold = similarity_threshold or settings.cache_similarity_threshold
        self._ttl = ttl_seconds or settings.cache_ttl_seconds
        self._max_entries = max_entries
        self._entries: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0
        self._index: Any = None
        self._dimension: int | None = None

    def _ensure_index(self) -> None:
        if FAISS_AVAILABLE and self._index is None:
            self._dimension = self._encoder.dimension
            self._index = faiss.IndexFlatIP(self._dimension)

    def _prune_expired(self) -> None:
        now = time.time()
        valid = []
        for entry in self._entries:
            if now - entry.timestamp < self._ttl:
                valid.append(entry)

        if len(valid) != len(self._entries):
            self._entries = valid
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        if not FAISS_AVAILABLE or self._index is None:
            return

        self._index.reset()
        if self._entries:
            embeddings = np.array([e.embedding for e in self._entries], dtype=np.float32)
            faiss.normalize_L2(embeddings)
            self._index.add(embeddings)

    def _search_linear(self, embedding: np.ndarray) -> tuple[int, float]:
        if not self._entries:
            return -1, 0.0

        best_idx = -1
        best_sim = 0.0
        norm_query = embedding / (np.linalg.norm(embedding) + 1e-9)

        for i, entry in enumerate(self._entries):
            norm_entry = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-9)
            sim = float(np.dot(norm_query, norm_entry))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return best_idx, best_sim

    def get(self, query: str) -> CacheHit | None:
        self._prune_expired()
        self._ensure_index()

        if not self._entries:
            self._misses += 1
            return None

        embedding = self._encoder.encode(query)

        if FAISS_AVAILABLE and self._index is not None and self._index.ntotal > 0:
            query_vec = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vec)
            distances, indices = self._index.search(query_vec, 1)
            best_idx = int(indices[0][0])
            best_sim = float(distances[0][0])
        else:
            best_idx, best_sim = self._search_linear(embedding)

        if best_idx < 0 or best_sim < self._threshold:
            self._misses += 1
            return None

        entry = self._entries[best_idx]
        self._hits += 1

        return CacheHit(
            query=entry.query,
            response=entry.response,
            model=entry.model,
            similarity=best_sim,
            age_seconds=time.time() - entry.timestamp,
        )

    def put(
        self,
        query: str,
        response: str,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_index()

        if len(self._entries) >= self._max_entries:
            self._entries.sort(key=lambda e: e.timestamp)
            self._entries = self._entries[len(self._entries) // 10 :]
            self._rebuild_index()

        embedding = self._encoder.encode(query)

        entry = CacheEntry(
            query=query,
            response=response,
            model=model,
            embedding=embedding,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._entries.append(entry)

        if FAISS_AVAILABLE and self._index is not None:
            vec = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            self._index.add(vec)

    def stats(self) -> CacheStats:
        total = self._hits + self._misses
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._entries),
            hit_rate=self._hits / total if total > 0 else 0.0,
        )

    def clear(self) -> None:
        self._entries = []
        self._hits = 0
        self._misses = 0
        if FAISS_AVAILABLE and self._index is not None:
            self._index.reset()
