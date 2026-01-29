import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestClassifier

if TYPE_CHECKING:
    from core.shared import SharedEncoder


@dataclass
class ComplexityResult:
    score: float
    category: str
    capabilities: list[str]
    confidence: float
    estimated_tokens: int


class ComplexityClassifier:
    CAPABILITY_KEYWORDS = {
        "code": [
            "code",
            "program",
            "function",
            "debug",
            "implement",
            "algorithm",
            "python",
            "javascript",
        ],
        "reasoning": ["explain", "why", "analyze", "compare", "evaluate", "reason", "logic"],
        "creative": ["write", "story", "poem", "creative", "imagine", "design", "compose"],
        "analysis": ["data", "analyze", "trend", "statistics", "research", "study", "investigate"],
    }

    def __init__(self, encoder: "SharedEncoder | None" = None, model_path: Path | None = None):
        from core.shared import get_encoder

        self._encoder = encoder or get_encoder()
        self._classifier: RandomForestClassifier | None = None
        if model_path and model_path.exists():
            with open(model_path, "rb") as f:
                self._classifier = pickle.load(f)

    def _heuristic_complexity(self, query: str) -> float:
        words = query.split()
        word_count = len(words)
        has_question = "?" in query
        has_code_markers = any(m in query for m in ["```", "def ", "function ", "class "])
        has_numbers = any(c.isdigit() for c in query)

        base = min(word_count / 20, 1.0) * 4
        if has_question:
            base += 1
        if has_code_markers:
            base += 2
        if has_numbers:
            base += 0.5

        unique_ratio = len(set(words)) / max(len(words), 1)
        base += unique_ratio * 2

        return min(max(base, 1), 10)

    def _detect_capabilities(self, query: str) -> list[str]:
        query_lower = query.lower()
        detected = []
        for cap, keywords in self.CAPABILITY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(cap)
        return detected if detected else ["reasoning"]

    def _estimate_tokens(self, query: str) -> int:
        return int(len(query.split()) * 1.3 + 100)

    def classify(self, query: str) -> ComplexityResult:
        if not query.strip():
            return ComplexityResult(
                score=1.0,
                category="simple",
                capabilities=["reasoning"],
                confidence=1.0,
                estimated_tokens=10,
            )

        embedding = self._encoder.encode(query)

        if self._classifier is not None:
            proba = self._classifier.predict_proba([embedding])[0]
            pred_idx = int(np.argmax(proba))
            score = float(pred_idx * 3 + 1.5)
            confidence = float(proba[pred_idx])
        else:
            score = self._heuristic_complexity(query)
            confidence = 0.7

        if score <= 3:
            category = "simple"
        elif score <= 6:
            category = "moderate"
        else:
            category = "complex"

        return ComplexityResult(
            score=score,
            category=category,
            capabilities=self._detect_capabilities(query),
            confidence=confidence,
            estimated_tokens=self._estimate_tokens(query),
        )

    def save(self, path: Path) -> None:
        if self._classifier is not None:
            with open(path, "wb") as f:
                pickle.dump(self._classifier, f)
