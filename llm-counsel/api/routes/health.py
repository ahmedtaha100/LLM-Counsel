from fastapi import APIRouter
from pydantic import BaseModel

from models.registry import get_registry

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    models_available: list[str]


class DetailedHealthResponse(BaseModel):
    status: str
    models: dict[str, bool]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    registry = get_registry()
    models = registry.list_models()
    return HealthResponse(
        status="healthy" if models else "degraded",
        models_available=models,
    )


@router.get("/health/models", response_model=DetailedHealthResponse)
async def models_health() -> DetailedHealthResponse:
    registry = get_registry()
    health_results = await registry.health_all()
    all_healthy = all(health_results.values()) if health_results else False
    return DetailedHealthResponse(
        status="healthy" if all_healthy else "degraded",
        models=health_results,
    )
