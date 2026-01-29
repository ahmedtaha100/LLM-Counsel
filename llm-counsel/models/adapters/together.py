import time

from tenacity import retry, stop_after_attempt, wait_exponential
from together import AsyncTogether

from models.adapters.base import BaseAdapter, CompletionResult


class TogetherAdapter(BaseAdapter):
    def __init__(
        self,
        api_key: str,
        model_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ):
        super().__init__(api_key, model_id, cost_per_1k_input, cost_per_1k_output)
        self.client = AsyncTogether(api_key=api_key)

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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = (time.perf_counter() - start) * 1000

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        text = response.choices[0].message.content or ""

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
            await self.client.models.list()
            return True
        except Exception:
            return False
