from __future__ import annotations

import os

from .base import LLMBackend
from .ollama import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT_SECONDS, OllamaBackend


def get_backend(
    backend_name: str = "ollama",
    *,
    model_name: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> LLMBackend:
    """Creates and returns an LLMBackend instance matching backend_name.

    Example:
        >>> backend = get_backend("ollama", model_name="LiquidAI/lfm2.5-1.2b-instruct")
        >>> isinstance(backend, LLMBackend)
        True
    """
    normalized_name = backend_name.strip().lower()
    if normalized_name == "ollama":
        target_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return OllamaBackend(
            model_name=model_name,
            base_url=target_base_url,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"Unsupported backend: '{backend_name}'. Expected: 'ollama'")
