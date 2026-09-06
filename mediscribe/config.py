from dataclasses import dataclass, field


@dataclass
class Settings:
    top_k: int = 4
    similarity_threshold: float = 0.20
    embedding_dimension: int = 3072
    degraded_mode: bool = False
    llm_temperature: float = 0.1
    max_chunk_chars: int = 500
    chunk_overlap: int = 80
    provider_name: str = "demo"
    api_key: str = "demo-key"
    provider_keys: dict[str, list[str]] = field(default_factory=lambda: {"demo": ["demo-key"]})


SETTINGS = Settings()
