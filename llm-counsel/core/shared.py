import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings


class SharedEncoder:
    _instance: "SharedEncoder | None" = None

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._encoder: SentenceTransformer | None = None
        self._dimension: int | None = None

    def _load(self) -> None:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name)
            test_emb = self._encoder.encode(["test"])
            self._dimension = test_emb.shape[1]

    @property
    def dimension(self) -> int:
        self._load()
        return self._dimension or 384

    def encode(self, text: str) -> np.ndarray:
        self._load()
        return self._encoder.encode([text])[0].astype(np.float32)

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        return self._encoder.encode(texts).astype(np.float32)


_encoder: SharedEncoder | None = None


def get_encoder() -> SharedEncoder:
    global _encoder
    if _encoder is None:
        _encoder = SharedEncoder()
    return _encoder


def reset_encoder() -> None:
    global _encoder
    _encoder = None
