from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from core.panel import PanelResponse, PanelResult

if TYPE_CHECKING:
    from core.shared import SharedEncoder


class AggregationStrategy(Enum):
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BEST_OF_N = "best_of_n"
    JUDGE_MODEL = "judge_model"


@dataclass
class JudgmentResult:
    response: str
    selected_model: str
    confidence: float
    strategy: str
    reasoning: str
    all_responses: dict[str, str]


class Judge:
    JUDGE_PROMPT = """You are evaluating multiple AI responses to select the best one.

Question: {question}

Responses:
{responses}

Evaluate each response for accuracy, completeness, and clarity.
Reply with ONLY the number (1, 2, 3, etc.) of the best response."""

    def __init__(self, encoder: "SharedEncoder | None" = None, judge_model: str = "gpt-4o"):
        from core.shared import get_encoder

        self._encoder = encoder or get_encoder()
        self._judge_model = judge_model

    def _compute_response_length_score(self, text: str) -> float:
        words = len(text.split())
        if words < 20:
            return 0.3
        if words < 100:
            return 0.7
        if words < 500:
            return 1.0
        return 0.8

    def _compute_structure_score(self, text: str) -> float:
        score = 0.5
        if "\n" in text:
            score += 0.1
        if any(text.startswith(str(i)) for i in range(1, 10)):
            score += 0.1
        if "```" in text:
            score += 0.15
        if any(marker in text for marker in ["First", "Second", "Finally", "However"]):
            score += 0.15
        return min(score, 1.0)

    def _score_response(self, text: str) -> float:
        length_score = self._compute_response_length_score(text)
        structure_score = self._compute_structure_score(text)
        return (length_score * 0.4) + (structure_score * 0.6)

    def _majority_vote(self, responses: dict[str, str]) -> tuple[str, str, float]:
        if not responses:
            return "", "", 0.0

        embeddings = self._encoder.batch_encode(list(responses.values()))
        n = len(embeddings)

        if n == 1:
            model = list(responses.keys())[0]
            return responses[model], model, 0.8

        similarity_sums = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i != j:
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                    )
                    sims.append(sim)
            similarity_sums.append(np.mean(sims) if sims else 0)

        best_idx = int(np.argmax(similarity_sums))
        best_model = list(responses.keys())[best_idx]
        confidence = float(similarity_sums[best_idx])

        return responses[best_model], best_model, confidence

    def _best_of_n(self, responses: dict[str, str]) -> tuple[str, str, float]:
        if not responses:
            return "", "", 0.0

        scores = {m: self._score_response(r) for m, r in responses.items()}
        best_model = max(scores, key=scores.get)
        return responses[best_model], best_model, scores[best_model]

    def _confidence_weighted(
        self, responses: dict[str, str], latencies: dict[str, float]
    ) -> tuple[str, str, float]:
        if not responses:
            return "", "", 0.0

        scores = {}
        for model, resp in responses.items():
            quality = self._score_response(resp)
            latency_factor = 1.0 / (1.0 + latencies.get(model, 1000) / 1000)
            scores[model] = quality * 0.7 + latency_factor * 0.3

        best_model = max(scores, key=scores.get)
        return responses[best_model], best_model, scores[best_model]

    async def _judge_model_select(
        self, responses: dict[str, str], question: str
    ) -> tuple[str, str, float, bool]:
        if not responses:
            return "", "", 0.0, False

        if len(responses) == 1:
            model = list(responses.keys())[0]
            return responses[model], model, 0.9, True

        from models.registry import get_registry

        registry = get_registry()
        models_list = list(responses.keys())

        response_text = "\n\n".join(
            f"Response {i + 1} ({m}):\n{responses[m]}" for i, m in enumerate(models_list)
        )

        prompt = self.JUDGE_PROMPT.format(question=question, responses=response_text)

        try:
            adapter = registry.get(self._judge_model)
            result = await adapter.complete(prompt, max_tokens=10, temperature=0)
            choice = result.text.strip()

            import re

            match = re.match(r"^(\d+)", choice)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(models_list):
                    model = models_list[idx]
                    return responses[model], model, 0.95, True

            text, model, conf = self._best_of_n(responses)
            return text, model, conf, False
        except Exception:
            text, model, conf = self._best_of_n(responses)
            return text, model, conf, False

    def judge(
        self,
        panel_result: PanelResult,
        strategy: AggregationStrategy = AggregationStrategy.BEST_OF_N,
        question: str = "",
    ) -> JudgmentResult:
        responses = {}
        latencies = {}

        for pr in panel_result.responses:
            if pr.success and pr.result:
                responses[pr.model] = pr.result.text
                latencies[pr.model] = pr.result.latency_ms

        if not responses:
            return JudgmentResult(
                response="No successful responses from panel",
                selected_model="none",
                confidence=0.0,
                strategy=strategy.value,
                reasoning="All models failed",
                all_responses={},
            )

        if strategy == AggregationStrategy.MAJORITY_VOTE:
            text, model, conf = self._majority_vote(responses)
            reasoning = "Selected response most similar to consensus"
        elif strategy == AggregationStrategy.CONFIDENCE_WEIGHTED:
            text, model, conf = self._confidence_weighted(responses, latencies)
            reasoning = "Weighted by quality and latency"
        elif strategy == AggregationStrategy.JUDGE_MODEL:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run, self._judge_model_select(responses, question)
                        )
                        text, model, conf, judge_succeeded = future.result()
                else:
                    text, model, conf, judge_succeeded = loop.run_until_complete(
                        self._judge_model_select(responses, question)
                    )
            except RuntimeError:
                text, model, conf, judge_succeeded = asyncio.run(
                    self._judge_model_select(responses, question)
                )
            if judge_succeeded:
                reasoning = f"Selected by {self._judge_model} as best response"
            else:
                reasoning = f"Fallback to best_of_n ({self._judge_model} unavailable/inconclusive)"
        else:
            text, model, conf = self._best_of_n(responses)
            reasoning = "Selected highest quality response"

        return JudgmentResult(
            response=text,
            selected_model=model,
            confidence=conf,
            strategy=strategy.value,
            reasoning=reasoning,
            all_responses=responses,
        )

    async def judge_async(
        self,
        panel_result: PanelResult,
        strategy: AggregationStrategy = AggregationStrategy.BEST_OF_N,
        question: str = "",
    ) -> JudgmentResult:
        responses = {}
        latencies = {}

        for pr in panel_result.responses:
            if pr.success and pr.result:
                responses[pr.model] = pr.result.text
                latencies[pr.model] = pr.result.latency_ms

        if not responses:
            return JudgmentResult(
                response="No successful responses from panel",
                selected_model="none",
                confidence=0.0,
                strategy=strategy.value,
                reasoning="All models failed",
                all_responses={},
            )

        if strategy == AggregationStrategy.MAJORITY_VOTE:
            text, model, conf = self._majority_vote(responses)
            reasoning = "Selected response most similar to consensus"
        elif strategy == AggregationStrategy.CONFIDENCE_WEIGHTED:
            text, model, conf = self._confidence_weighted(responses, latencies)
            reasoning = "Weighted by quality and latency"
        elif strategy == AggregationStrategy.JUDGE_MODEL:
            text, model, conf, judge_succeeded = await self._judge_model_select(responses, question)
            if judge_succeeded:
                reasoning = f"Selected by {self._judge_model} as best response"
            else:
                reasoning = f"Fallback to best_of_n ({self._judge_model} unavailable/inconclusive)"
        else:
            text, model, conf = self._best_of_n(responses)
            reasoning = "Selected highest quality response"

        return JudgmentResult(
            response=text,
            selected_model=model,
            confidence=conf,
            strategy=strategy.value,
            reasoning=reasoning,
            all_responses=responses,
        )

    def judge_single(self, response: PanelResponse) -> JudgmentResult:
        if not response.success or not response.result:
            return JudgmentResult(
                response=response.error or "Model call failed",
                selected_model=response.model,
                confidence=0.0,
                strategy="single",
                reasoning="Single model call failed",
                all_responses={},
            )

        return JudgmentResult(
            response=response.result.text,
            selected_model=response.model,
            confidence=self._score_response(response.result.text),
            strategy="single",
            reasoning="Single model response",
            all_responses={response.model: response.result.text},
        )
