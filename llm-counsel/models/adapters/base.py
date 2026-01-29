from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    model: str
    raw_response: Any = None


class BaseAdapter(ABC):
    def __init__(
        self,
        api_key: str,
        model_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

    def calculate_cost(self, tokens_in: int, tokens_out: int) -> float:
        input_cost = (tokens_in / 1000) * self.cost_per_1k_input
        output_cost = (tokens_out / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> CompletionResult:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass
