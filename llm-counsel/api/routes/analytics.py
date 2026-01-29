from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.routes.query import get_cache
from core.metrics import get_metrics

router = APIRouter()


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    entries: int
    hit_rate: float


class LatencyStats(BaseModel):
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class CostStats(BaseModel):
    total_cost: float
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    estimated_gpt4o_cost: float
    cost_savings: float


class AnalyticsResponse(BaseModel):
    cache: CacheStatsResponse
    total_queries: int
    latency: LatencyStats
    cost: CostStats
    queries_by_model: dict[str, int] = Field(default_factory=dict)
    selections_by_model: dict[str, int] = Field(default_factory=dict)
    failures_by_model: dict[str, int] = Field(default_factory=dict)
    panel_usage_rate: float


@router.get("", response_model=AnalyticsResponse)
async def get_analytics() -> AnalyticsResponse:
    cache = get_cache()
    cache_stats = cache.stats()
    metrics = get_metrics()
    summary = metrics.summary()

    return AnalyticsResponse(
        cache=CacheStatsResponse(
            hits=cache_stats.hits,
            misses=cache_stats.misses,
            entries=cache_stats.entries,
            hit_rate=cache_stats.hit_rate,
        ),
        total_queries=summary.total_queries,
        latency=LatencyStats(
            avg_ms=summary.avg_latency_ms,
            p50_ms=summary.p50_latency_ms,
            p95_ms=summary.p95_latency_ms,
            p99_ms=summary.p99_latency_ms,
        ),
        cost=CostStats(
            total_cost=summary.total_cost,
            cost_by_model=summary.cost_by_model,
            estimated_gpt4o_cost=summary.estimated_gpt4o_cost,
            cost_savings=summary.cost_savings,
        ),
        queries_by_model=summary.queries_by_model,
        selections_by_model=summary.selections_by_model,
        failures_by_model=summary.failures_by_model,
        panel_usage_rate=summary.panel_usage_rate,
    )


@router.get("/cache", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    cache = get_cache()
    stats = cache.stats()
    return CacheStatsResponse(
        hits=stats.hits,
        misses=stats.misses,
        entries=stats.entries,
        hit_rate=stats.hit_rate,
    )


@router.post("/cache/clear")
async def clear_cache() -> dict[str, str]:
    cache = get_cache()
    cache.clear()
    return {"status": "cleared"}


@router.post("/metrics/clear")
async def clear_metrics() -> dict[str, str]:
    metrics = get_metrics()
    metrics.clear()
    return {"status": "cleared"}


class RecentQueryInfo(BaseModel):
    query: str
    models: list[str] = Field(default_factory=list)
    selected_model: str
    failed_models: list[str] = Field(default_factory=list)
    latency_ms: float
    total_cost_usd: float
    cost_by_model: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool
    panel_mode: bool


@router.get("/recent", response_model=list[RecentQueryInfo])
async def get_recent_queries(n: int = 100) -> list[RecentQueryInfo]:
    metrics = get_metrics()
    recent = metrics.recent(n)
    return [
        RecentQueryInfo(
            query=m.query[:100] + "..." if len(m.query) > 100 else m.query,
            models=list(m.models),
            selected_model=m.selected_model,
            failed_models=list(m.failed_models),
            latency_ms=m.latency_ms,
            total_cost_usd=m.total_cost_usd,
            cost_by_model=dict(m.cost_by_model),
            cache_hit=m.cache_hit,
            panel_mode=m.panel_mode,
        )
        for m in recent
    ]
