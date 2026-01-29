import json
from dataclasses import dataclass, field
from pathlib import Path

from core.judge import AggregationStrategy, Judge
from core.panel import Panel
from core.router import Router


@dataclass
class BenchmarkResult:
    query: str
    expected: str
    actual: str
    model: str
    correct: bool
    latency_ms: float
    cost_usd: float
    mode: str


@dataclass
class BenchmarkSummary:
    total: int
    correct: int
    accuracy: float
    total_cost: float
    avg_latency: float
    by_mode: dict[str, dict[str, float]] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(
        self,
        router: Router | None = None,
        panel: Panel | None = None,
        judge: Judge | None = None,
    ):
        self._router = router or Router()
        self._panel = panel or Panel()
        self._judge = judge or Judge()

    def _check_answer(self, expected: str, actual: str) -> bool:
        exp_lower = expected.lower().strip()
        act_lower = actual.lower().strip()
        if exp_lower in act_lower:
            return True
        if act_lower in exp_lower:
            return True
        exp_words = set(exp_lower.split())
        act_words = set(act_lower.split())
        overlap = len(exp_words & act_words) / max(len(exp_words), 1)
        return overlap > 0.5

    async def run_single(
        self,
        query: str,
        expected: str,
        mode: str = "auto",
    ) -> BenchmarkResult:
        routing = self._router.route(query)

        if mode == "panel" or (mode == "auto" and routing.use_panel):
            models = routing.panel_models or [routing.model]
            result = await self._panel.deliberate(models, query)
            judgment = self._judge.judge(result, AggregationStrategy.BEST_OF_N)
            return BenchmarkResult(
                query=query,
                expected=expected,
                actual=judgment.response,
                model=judgment.selected_model,
                correct=self._check_answer(expected, judgment.response),
                latency_ms=result.total_latency_ms,
                cost_usd=result.total_cost_usd,
                mode="panel",
            )

        single = await self._panel.single(routing.model, query)
        judgment = self._judge.judge_single(single)
        return BenchmarkResult(
            query=query,
            expected=expected,
            actual=judgment.response,
            model=judgment.selected_model,
            correct=self._check_answer(expected, judgment.response),
            latency_ms=single.result.latency_ms if single.result else 0,
            cost_usd=single.result.cost_usd if single.result else 0,
            mode="single",
        )

    async def run_benchmark(
        self,
        dataset: list[dict[str, str]],
        modes: list[str] | None = None,
    ) -> dict[str, BenchmarkSummary]:
        modes = modes or ["auto", "panel", "single"]
        summaries: dict[str, BenchmarkSummary] = {}

        for mode in modes:
            results = []
            for item in dataset:
                result = await self.run_single(
                    item["query"],
                    item["expected"],
                    mode,
                )
                results.append(result)

            correct = sum(1 for r in results if r.correct)
            total_cost = sum(r.cost_usd for r in results)
            avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0

            summaries[mode] = BenchmarkSummary(
                total=len(results),
                correct=correct,
                accuracy=correct / len(results) if results else 0,
                total_cost=total_cost,
                avg_latency=avg_latency,
            )

        return summaries

    def load_dataset(self, path: Path) -> list[dict[str, str]]:
        with open(path) as f:
            return json.load(f)
