from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.cache import SemanticCache
from core.dissent import DissentDetector
from core.judge import AggregationStrategy, Judge
from core.metrics import get_metrics
from core.panel import Panel
from core.router import Router

router = APIRouter()

_router: Router | None = None
_panel: Panel | None = None
_judge: Judge | None = None
_dissent: DissentDetector | None = None
_cache: SemanticCache | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def get_panel() -> Panel:
    global _panel
    if _panel is None:
        _panel = Panel()
    return _panel


def get_judge() -> Judge:
    global _judge
    if _judge is None:
        _judge = Judge()
    return _judge


def get_dissent() -> DissentDetector:
    global _dissent
    if _dissent is None:
        _dissent = DissentDetector()
    return _dissent


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


class QueryMode(str, Enum):
    AUTO = "auto"
    PANEL = "panel"
    SINGLE = "single"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100000)
    mode: QueryMode = QueryMode.AUTO
    models: list[str] | None = None
    system: str | None = None
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0, le=2)
    use_cache: bool = True
    budget_mode: bool = False


class DissentPairInfo(BaseModel):
    model_a: str
    model_b: str
    similarity: float
    divergent_claims_a: list[str] = Field(default_factory=list)
    divergent_claims_b: list[str] = Field(default_factory=list)


class DissentInfo(BaseModel):
    has_dissent: bool
    level: str
    summary: str
    pairs: list[DissentPairInfo] = Field(default_factory=list)
    consensus_claims: list[str] = Field(default_factory=list)
    unique_claims: dict[str, list[str]] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    response: str
    model_used: str
    confidence: float
    latency_ms: float
    cost_usd: float
    cache_hit: bool = False
    dissent: DissentInfo | None = None
    routing_reason: str | None = None
    all_responses: dict[str, str] | None = None


@router.post("/query", response_model=QueryResponse)
async def submit_query(req: QueryRequest) -> QueryResponse:
    cache = get_cache()
    metrics = get_metrics()

    if req.use_cache:
        hit = cache.get(req.query)
        if hit:
            metrics.record_single(
                query=req.query,
                model=hit.model,
                latency_ms=0,
                cost_usd=0,
                tokens_in=0,
                tokens_out=0,
                cache_hit=True,
            )
            return QueryResponse(
                response=hit.response,
                model_used=hit.model,
                confidence=hit.similarity,
                latency_ms=0,
                cost_usd=0,
                cache_hit=True,
                routing_reason="Cache hit",
            )

    routing = get_router().route(req.query, prefer_budget=req.budget_mode)
    panel = get_panel()
    judge = get_judge()
    dissent_detector = get_dissent()

    use_panel = req.mode == QueryMode.PANEL or (req.mode == QueryMode.AUTO and routing.use_panel)

    if use_panel:
        models = req.models or routing.panel_models
        if not models:
            models = [routing.model]

        result = await panel.deliberate(
            models=models,
            prompt=req.query,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=req.system,
        )

        judgment = await judge.judge_async(
            result, AggregationStrategy.BEST_OF_N, question=req.query
        )

        dissent_info: DissentInfo | None = None
        if len(result.successful_models) >= 2:
            dr = dissent_detector.detect(judgment.all_responses)
            dissent_info = DissentInfo(
                has_dissent=dr.has_dissent,
                level=dr.dissent_level,
                summary=dr.summary,
                pairs=[
                    DissentPairInfo(
                        model_a=p.model_a,
                        model_b=p.model_b,
                        similarity=p.similarity,
                        divergent_claims_a=list(p.divergent_claims_a),
                        divergent_claims_b=list(p.divergent_claims_b),
                    )
                    for p in dr.pairs
                ],
                consensus_claims=list(dr.consensus_claims),
                unique_claims={k: list(v) for k, v in dr.unique_claims.items()},
            )

        cost_by_model: dict[str, float] = {}
        tokens_by_model: dict[str, tuple[int, int]] = {}
        latency_by_model: dict[str, float] = {}
        failed_models: list[str] = []

        for r in result.responses:
            if r.result:
                cost_by_model[r.model] = r.result.cost_usd
                tokens_by_model[r.model] = (r.result.tokens_in, r.result.tokens_out)
                latency_by_model[r.model] = r.result.latency_ms
            else:
                failed_models.append(r.model)

        all_attempted = list(cost_by_model.keys()) + failed_models

        metrics.record(
            query=req.query,
            models=all_attempted,
            selected_model=judgment.selected_model,
            latency_ms=result.total_latency_ms,
            total_cost_usd=result.total_cost_usd,
            cost_by_model=cost_by_model,
            tokens_by_model=tokens_by_model,
            latency_by_model=latency_by_model,
            failed_models=failed_models,
            cache_hit=False,
            panel_mode=True,
        )

        if req.use_cache and judgment.response and judgment.selected_model != "none":
            cache.put(req.query, judgment.response, judgment.selected_model)

        return QueryResponse(
            response=judgment.response,
            model_used=judgment.selected_model,
            confidence=judgment.confidence,
            latency_ms=result.total_latency_ms,
            cost_usd=result.total_cost_usd,
            dissent=dissent_info,
            routing_reason=routing.reason,
            all_responses=judgment.all_responses if len(judgment.all_responses) > 1 else None,
        )

    model = req.models[0] if req.models else routing.model
    result = await panel.single(
        model=model,
        prompt=req.query,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        system=req.system,
    )

    judgment = judge.judge_single(result)

    tokens_in = result.result.tokens_in if result.result else 0
    tokens_out = result.result.tokens_out if result.result else 0
    latency = result.result.latency_ms if result.result else result.attempt_latency_ms
    cost = result.result.cost_usd if result.result else 0

    metrics.record_single(
        query=req.query,
        model=model,
        latency_ms=latency,
        cost_usd=cost,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_hit=False,
        failed=not result.success,
    )

    if req.use_cache and judgment.response and result.success:
        cache.put(req.query, judgment.response, model)

    return QueryResponse(
        response=judgment.response,
        model_used=judgment.selected_model,
        confidence=judgment.confidence,
        latency_ms=latency,
        cost_usd=cost,
        routing_reason=routing.reason,
    )
