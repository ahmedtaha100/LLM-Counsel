from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config.settings import routing_config

if TYPE_CHECKING:
    from core.shared import SharedEncoder


@dataclass
class DissentPair:
    model_a: str
    model_b: str
    similarity: float
    divergent_claims_a: list[str]
    divergent_claims_b: list[str]


@dataclass
class DissentReport:
    has_dissent: bool
    dissent_level: str
    pairs: list[DissentPair]
    consensus_claims: list[str]
    unique_claims: dict[str, list[str]]
    summary: str


class DissentDetector:
    def __init__(
        self,
        encoder: "SharedEncoder | None" = None,
        similarity_threshold: float | None = None,
    ):
        from core.shared import get_encoder

        self._encoder = encoder or get_encoder()
        self._threshold = similarity_threshold or routing_config["dissent"]["similarity_threshold"]

    def _extract_claims(self, text: str) -> list[str]:
        sentences = []
        for sent in text.replace("\n", " ").split("."):
            sent = sent.strip()
            if len(sent) > 20:
                sentences.append(sent)
        return sentences[:20]

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self._encoder.encode(text_a)
        emb_b = self._encoder.encode(text_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    def _find_divergent_claims(
        self, claims_a: list[str], claims_b: list[str]
    ) -> tuple[list[str], list[str]]:
        if not claims_a or not claims_b:
            return claims_a, claims_b

        emb_a = self._encoder.batch_encode(claims_a)
        emb_b = self._encoder.batch_encode(claims_b)

        divergent_a = []
        for i, claim in enumerate(claims_a):
            max_sim = 0.0
            for j in range(len(claims_b)):
                sim = np.dot(emb_a[i], emb_b[j]) / (
                    np.linalg.norm(emb_a[i]) * np.linalg.norm(emb_b[j]) + 1e-9
                )
                max_sim = max(max_sim, sim)
            if max_sim < self._threshold:
                divergent_a.append(claim)

        divergent_b = []
        for j, claim in enumerate(claims_b):
            max_sim = 0.0
            for i in range(len(claims_a)):
                sim = np.dot(emb_a[i], emb_b[j]) / (
                    np.linalg.norm(emb_a[i]) * np.linalg.norm(emb_b[j]) + 1e-9
                )
                max_sim = max(max_sim, sim)
            if max_sim < self._threshold:
                divergent_b.append(claim)

        return divergent_a, divergent_b

    def _find_consensus_claims(self, all_claims: dict[str, list[str]]) -> list[str]:
        if len(all_claims) < 2:
            return []

        models = list(all_claims.keys())
        first_claims = all_claims[models[0]]

        if not first_claims:
            return []

        emb_first = self._encoder.batch_encode(first_claims)
        consensus = []

        for i, claim in enumerate(first_claims):
            in_all = True
            for other in models[1:]:
                other_claims = all_claims[other]
                if not other_claims:
                    in_all = False
                    break

                emb_other = self._encoder.batch_encode(other_claims)
                max_sim = 0.0
                for j in range(len(other_claims)):
                    sim = np.dot(emb_first[i], emb_other[j]) / (
                        np.linalg.norm(emb_first[i]) * np.linalg.norm(emb_other[j]) + 1e-9
                    )
                    max_sim = max(max_sim, sim)

                if max_sim < self._threshold:
                    in_all = False
                    break

            if in_all:
                consensus.append(claim)

        return consensus

    def detect(self, responses: dict[str, str]) -> DissentReport:
        if len(responses) < 2:
            return DissentReport(
                has_dissent=False,
                dissent_level="none",
                pairs=[],
                consensus_claims=[],
                unique_claims={},
                summary="Single response, no comparison possible",
            )

        models = list(responses.keys())
        pairs: list[DissentPair] = []
        all_claims = {m: self._extract_claims(r) for m, r in responses.items()}

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a, m_b = models[i], models[j]
                sim = self._compute_similarity(responses[m_a], responses[m_b])

                div_a, div_b = self._find_divergent_claims(all_claims[m_a], all_claims[m_b])

                pairs.append(
                    DissentPair(
                        model_a=m_a,
                        model_b=m_b,
                        similarity=sim,
                        divergent_claims_a=div_a[:5],
                        divergent_claims_b=div_b[:5],
                    )
                )

        min_sim = min(p.similarity for p in pairs) if pairs else 1.0
        has_dissent = min_sim < self._threshold

        if min_sim >= 0.85:
            dissent_level = "none"
        elif min_sim >= self._threshold:
            dissent_level = "low"
        elif min_sim >= 0.5:
            dissent_level = "moderate"
        else:
            dissent_level = "high"

        consensus = self._find_consensus_claims(all_claims)

        unique_claims = {}
        for model, claims in all_claims.items():
            unique = []
            other_claims = []
            for m, c in all_claims.items():
                if m != model:
                    other_claims.extend(c)

            if claims and other_claims:
                emb_mine = self._encoder.batch_encode(claims)
                emb_others = self._encoder.batch_encode(other_claims)

                for i, claim in enumerate(claims):
                    max_sim = 0.0
                    for j in range(len(other_claims)):
                        sim = np.dot(emb_mine[i], emb_others[j]) / (
                            np.linalg.norm(emb_mine[i]) * np.linalg.norm(emb_others[j]) + 1e-9
                        )
                        max_sim = max(max_sim, sim)
                    if max_sim < 0.6:
                        unique.append(claim)

            unique_claims[model] = unique[:3]

        summary = self._generate_summary(has_dissent, dissent_level, pairs, unique_claims)

        return DissentReport(
            has_dissent=has_dissent,
            dissent_level=dissent_level,
            pairs=pairs,
            consensus_claims=consensus[:5],
            unique_claims=unique_claims,
            summary=summary,
        )

    def _generate_summary(
        self,
        has_dissent: bool,
        level: str,
        pairs: list[DissentPair],
        unique: dict[str, list[str]],
    ) -> str:
        if not has_dissent:
            return "All models produced similar responses with high agreement."

        most_divergent = min(pairs, key=lambda p: p.similarity) if pairs else None

        parts = [f"Dissent level: {level}."]

        if most_divergent:
            parts.append(
                f"Greatest disagreement between {most_divergent.model_a} and "
                f"{most_divergent.model_b} (similarity: {most_divergent.similarity:.2f})."
            )

        models_with_unique = [m for m, c in unique.items() if c]
        if models_with_unique:
            parts.append(f"Unique claims found from: {', '.join(models_with_unique)}.")

        return " ".join(parts)
