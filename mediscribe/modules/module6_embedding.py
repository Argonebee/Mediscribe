from __future__ import annotations

import os
from typing import List

from mediscribe.config import SETTINGS
from mediscribe.contracts import Embedding


class MockEmbeddingProvider:
    def embed(self, text: str) -> List[float]:
        # deterministic hash-based vector simulating an embedding result
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(3072):
            val = (digest[i % len(digest)] / 255.0) - 0.5
            values.append(round(val, 6))
        return values


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for real embeddings; install with: pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", SETTINGS.api_key)
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")


class EmbeddingService:
    def __init__(self, provider: MockEmbeddingProvider | OpenAIEmbeddingProvider | None = None, use_real: bool | None = None):
        if use_real is None:
            use_real = SETTINGS.provider_name != "demo"
        
        if provider is not None:
            self.provider = provider
        elif use_real:
            self.provider = OpenAIEmbeddingProvider()
        else:
            self.provider = MockEmbeddingProvider()
        
        self.is_real = isinstance(self.provider, OpenAIEmbeddingProvider)

    def generate(self, chunk_text: str, chunk_id: str) -> Embedding:
        values = self.provider.embed(chunk_text)
        model_name = "openai-text-embedding-3-small" if self.is_real else "mock-embedding-provider"
        return Embedding(
            chunk_id=chunk_id,
            model_name=model_name,
            vector=values,
            dimension=len(values),
        )
