from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ToolNotSupportedError(RuntimeError):
    """Raised when an LLM backend or model does not support tool calling."""


@dataclass(frozen=True)
class ToolCall:
    """Represents a function tool call requested by an LLM.

    Example:
        >>> call = ToolCall(name="search_srt", arguments={"query": "prazo"})
        >>> call.name
        'search_srt'
    """

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    """Represents a structured chat response from an LLM backend.

    Example:
        >>> resp = ChatResponse(content="Olá")
        >>> resp.content
        'Olá'
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol defining the interface for LLM inference backends.

    Example:
        >>> class CustomBackend(LLMBackend):
        ...     def generate(self, prompt: str) -> str:
        ...         return "summary"
        ...     def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        ...         return ChatResponse(content="reply")
        ...     def cleanup(self) -> None:
        ...         pass
    """

    def generate(self, prompt: str) -> str:
        """Generates text from the given prompt.

        Args:
            prompt: Text prompt to send to the model.

        Returns:
            Sanitized model output string.
        """
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Executes a multi-turn chat turn with optional tool definitions."""
        ...

    def cleanup(self) -> None:
        """Releases backend resources and unloads model from memory."""
        ...
