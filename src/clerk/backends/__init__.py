from __future__ import annotations

from .base import ChatResponse, LLMBackend, ToolCall, ToolNotSupportedError
from .factory import get_backend
from .ollama import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaBackend,
)

__all__ = [
    "DEFAULT_LLM_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "ChatResponse",
    "LLMBackend",
    "OllamaBackend",
    "ToolCall",
    "ToolNotSupportedError",
    "get_backend",
]
