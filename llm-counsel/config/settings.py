from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent


def load_yaml(filename: str) -> dict[str, Any]:
    with open(CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    together_api_key: str = Field(default="")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    cache_similarity_threshold: float = Field(default=0.92)
    cache_ttl_seconds: int = Field(default=3600)
    default_temperature: float = Field(default=0.7)
    default_max_tokens: int = Field(default=4096)
    request_timeout: int = Field(default=60)
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
models_config = load_yaml("models.yaml")
routing_config = load_yaml("routing_rules.yaml")
