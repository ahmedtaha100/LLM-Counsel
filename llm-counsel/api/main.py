from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware.logging import LoggingMiddleware
from api.routes import analytics, health, query


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="LLM Counsel",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)

app.include_router(query.router, tags=["query"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(health.router, tags=["health"])
