import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from config.settings import settings
from models.adapters.base import CompletionResult

if TYPE_CHECKING:
    from models.registry import ModelRegistry


@dataclass
class PanelResponse:
    model: str
    result: CompletionResult | None
    error: str | None = None
    success: bool = True
    attempt_latency_ms: float = 0.0


@dataclass
class PanelResult:
    responses: list[PanelResponse]
    total_latency_ms: float
    total_cost_usd: float
    successful_models: list[str] = field(default_factory=list)
    failed_models: list[str] = field(default_factory=list)


class Panel:
    def __init__(
        self,
        registry: "ModelRegistry | None" = None,
        timeout_seconds: float | None = None,
    ):
        from models.registry import get_registry

        self._registry = registry or get_registry()
        self._timeout = timeout_seconds or settings.request_timeout

    async def _call_model(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system: str | None,
    ) -> PanelResponse:
        start = time.perf_counter()
        try:
            adapter = self._registry.get(model_name)
            result = await asyncio.wait_for(
                adapter.complete(prompt, max_tokens, temperature, system),
                timeout=self._timeout,
            )
            latency = (time.perf_counter() - start) * 1000
            return PanelResponse(
                model=model_name, result=result, attempt_latency_ms=latency
            )
        except TimeoutError:
            latency = (time.perf_counter() - start) * 1000
            return PanelResponse(
                model=model_name,
                result=None,
                error="Timeout",
                success=False,
                attempt_latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return PanelResponse(
                model=model_name,
                result=None,
                error=str(e),
                success=False,
                attempt_latency_ms=latency,
            )

    async def deliberate(
        self,
        models: list[str],
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> PanelResult:
        max_tokens = max_tokens or settings.default_max_tokens
        temperature = temperature or settings.default_temperature

        if not models:
            return PanelResult(
                responses=[],
                total_latency_ms=0,
                total_cost_usd=0,
            )

        tasks = [self._call_model(m, prompt, max_tokens, temperature, system) for m in models]

        start = time.perf_counter()
        responses = await asyncio.gather(*tasks)
        total_latency = (time.perf_counter() - start) * 1000

        successful = []
        failed = []
        total_cost = 0.0

        for resp in responses:
            if resp.success and resp.result:
                successful.append(resp.model)
                total_cost += resp.result.cost_usd
            else:
                failed.append(resp.model)

        return PanelResult(
            responses=list(responses),
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
            successful_models=successful,
            failed_models=failed,
        )

    async def single(
        self,
        model: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ) -> PanelResponse:
        max_tokens = max_tokens or settings.default_max_tokens
        temperature = temperature or settings.default_temperature
        return await self._call_model(model, prompt, max_tokens, temperature, system)
