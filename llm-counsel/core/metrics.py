import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class QueryMetrics:
    query: str
    models: list[str]
    selected_model: str
    latency_ms: float
    total_cost_usd: float
    cost_by_model: dict[str, float]
    tokens_by_model: dict[str, tuple[int, int]]
    latency_by_model: dict[str, float]
    failed_models: list[str]
    timestamp: float
    cache_hit: bool
    panel_mode: bool


@dataclass
class ModelStats:
    queries: int = 0
    selected_count: int = 0
    failure_count: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    latencies: list[float] = field(default_factory=list)


@dataclass
class AnalyticsSummary:
    total_queries: int
    total_cost: float
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cache_hit_rate: float
    panel_usage_rate: float
    cost_by_model: dict[str, float]
    queries_by_model: dict[str, int]
    selections_by_model: dict[str, int]
    failures_by_model: dict[str, int]
    estimated_gpt4o_cost: float
    cost_savings: float


class MetricsTracker:
    GPT4O_COST_PER_1K_IN = 0.005
    GPT4O_COST_PER_1K_OUT = 0.015

    def __init__(self, max_history: int = 10000):
        self._history: list[QueryMetrics] = []
        self._max_history = max_history
        self._model_stats: dict[str, ModelStats] = defaultdict(ModelStats)
        self._cache_hits = 0
        self._cache_misses = 0
        self._panel_queries = 0
        self._single_queries = 0

    def record(
        self,
        query: str,
        models: list[str],
        selected_model: str,
        latency_ms: float,
        total_cost_usd: float,
        cost_by_model: dict[str, float],
        tokens_by_model: dict[str, tuple[int, int]],
        latency_by_model: dict[str, float],
        cache_hit: bool,
        panel_mode: bool,
        failed_models: list[str] | None = None,
    ) -> None:
        failed_models = failed_models or []

        metrics = QueryMetrics(
            query=query,
            models=models,
            selected_model=selected_model,
            latency_ms=latency_ms,
            total_cost_usd=total_cost_usd,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            latency_by_model=latency_by_model,
            failed_models=failed_models,
            timestamp=time.time(),
            cache_hit=cache_hit,
            panel_mode=panel_mode,
        )

        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        for model in models:
            stats = self._model_stats[model]
            stats.queries += 1

            if model in failed_models:
                stats.failure_count += 1
            else:
                model_cost = cost_by_model.get(model, 0.0)
                stats.total_cost += model_cost
                model_latency = latency_by_model.get(model, latency_ms)
                stats.latencies.append(model_latency)
                stats.total_latency_ms += model_latency
                if len(stats.latencies) > 1000:
                    stats.latencies = stats.latencies[-1000:]
                tokens = tokens_by_model.get(model, (0, 0))
                stats.total_tokens_in += tokens[0]
                stats.total_tokens_out += tokens[1]

        if selected_model and selected_model != "none":
            self._model_stats[selected_model].selected_count += 1

        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        if panel_mode:
            self._panel_queries += 1
        else:
            self._single_queries += 1

    def record_single(
        self,
        query: str,
        model: str,
        latency_ms: float,
        cost_usd: float,
        tokens_in: int,
        tokens_out: int,
        cache_hit: bool,
        failed: bool = False,
    ) -> None:
        failed_models = [model] if failed else []
        cost_by_model = {} if failed else {model: cost_usd}
        tokens_by_model = {} if failed else {model: (tokens_in, tokens_out)}
        latency_by_model = {} if failed else {model: latency_ms}
        selected = "none" if failed else model

        self.record(
            query=query,
            models=[model],
            selected_model=selected,
            latency_ms=latency_ms,
            total_cost_usd=cost_usd,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            latency_by_model=latency_by_model,
            cache_hit=cache_hit,
            panel_mode=False,
            failed_models=failed_models,
        )

    def _estimate_gpt4o_cost(self) -> float:
        total = 0.0
        for m in self._history:
            if m.cache_hit or m.selected_model == "none":
                continue
            selected_tokens = m.tokens_by_model.get(m.selected_model)
            if selected_tokens:
                tokens_in, tokens_out = selected_tokens
            else:
                tokens_in, tokens_out = 100, 200
            estimated_in = tokens_in if tokens_in else 100
            estimated_out = tokens_out if tokens_out else 200
            total += (estimated_in / 1000) * self.GPT4O_COST_PER_1K_IN
            total += (estimated_out / 1000) * self.GPT4O_COST_PER_1K_OUT
        return total

    def summary(self) -> AnalyticsSummary:
        total_queries = len(self._history)

        if total_queries == 0:
            return AnalyticsSummary(
                total_queries=0,
                total_cost=0.0,
                total_latency_ms=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                cache_hit_rate=0.0,
                panel_usage_rate=0.0,
                cost_by_model={},
                queries_by_model={},
                selections_by_model={},
                failures_by_model={},
                estimated_gpt4o_cost=0.0,
                cost_savings=0.0,
            )

        all_latencies = [m.latency_ms for m in self._history]
        latency_arr = np.array(all_latencies)

        total_cost = sum(m.total_cost_usd for m in self._history)
        total_latency = sum(all_latencies)

        total_cache = self._cache_hits + self._cache_misses
        total_mode = self._panel_queries + self._single_queries

        cost_by_model = {m: s.total_cost for m, s in self._model_stats.items()}
        queries_by_model = {m: s.queries for m, s in self._model_stats.items()}
        selections_by_model = {
            m: s.selected_count for m, s in self._model_stats.items() if s.selected_count > 0
        }
        failures_by_model = {
            m: s.failure_count for m, s in self._model_stats.items() if s.failure_count > 0
        }

        gpt4o_cost = self._estimate_gpt4o_cost()
        savings = max(0, gpt4o_cost - total_cost)

        return AnalyticsSummary(
            total_queries=total_queries,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            avg_latency_ms=total_latency / total_queries,
            p50_latency_ms=float(np.percentile(latency_arr, 50)),
            p95_latency_ms=float(np.percentile(latency_arr, 95)),
            p99_latency_ms=float(np.percentile(latency_arr, 99)),
            cache_hit_rate=self._cache_hits / total_cache if total_cache > 0 else 0.0,
            panel_usage_rate=self._panel_queries / total_mode if total_mode > 0 else 0.0,
            cost_by_model=cost_by_model,
            queries_by_model=queries_by_model,
            selections_by_model=selections_by_model,
            failures_by_model=failures_by_model,
            estimated_gpt4o_cost=gpt4o_cost,
            cost_savings=savings,
        )

    def recent(self, n: int = 100) -> list[QueryMetrics]:
        return self._history[-n:]

    def clear(self) -> None:
        self._history = []
        self._model_stats = defaultdict(ModelStats)
        self._cache_hits = 0
        self._cache_misses = 0
        self._panel_queries = 0
        self._single_queries = 0


_metrics: MetricsTracker | None = None


def get_metrics() -> MetricsTracker:
    global _metrics
    if _metrics is None:
        _metrics = MetricsTracker()
    return _metrics
