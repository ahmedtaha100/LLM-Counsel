import time

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from models.adapters.base import BaseAdapter, CompletionResult


class AnthropicAdapter(BaseAdapter):
    def __init__(
        self,
        api_key: str,
        model_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ):
        super().__init__(api_key, model_id, cost_per_1k_input, cost_per_1k_output)
        self.client = AsyncAnthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> CompletionResult:
        start = time.perf_counter()
        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = (time.perf_counter() - start) * 1000

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        text = response.content[0].text if response.content else ""

        return CompletionResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            cost_usd=self.calculate_cost(tokens_in, tokens_out),
            model=self.model_id,
            raw_response=response,
        )

    async def health_check(self) -> bool:
        try:
            await self.client.messages.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
