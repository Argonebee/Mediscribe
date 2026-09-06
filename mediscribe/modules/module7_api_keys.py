from __future__ import annotations

from typing import Any, Dict, List


class APIKeyManager:
    def __init__(self, keys: List[str] | None = None, keys_by_provider: Dict[str, List[str]] | None = None):
        self.keys_by_provider = keys_by_provider or {"demo": keys or ["demo-key-1", "demo-key-2"]}
        self._active_index_by_provider: Dict[str, int] = {provider: 0 for provider in self.keys_by_provider}

    def get_active_key(self, provider: str = "demo") -> str:
        keys = self.keys_by_provider.get(provider, self.keys_by_provider.get("demo", ["demo-key-1"]))
        return keys[self._active_index_by_provider.get(provider, 0) % len(keys)]

    def rotate_key(self, provider: str = "demo") -> str:
        keys = self.keys_by_provider.get(provider, self.keys_by_provider.get("demo", ["demo-key-1"]))
        self._active_index_by_provider[provider] = (self._active_index_by_provider.get(provider, 0) + 1) % len(keys)
        return self.get_active_key(provider)

    def execute(self, provider: str, request: Dict[str, Any]) -> Dict[str, Any]:
        key = self.get_active_key(provider)
        return {
            "provider": provider,
            "key": key,
            "request": request,
            "status": "ok",
        }
