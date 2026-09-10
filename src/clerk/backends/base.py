from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol defining the interface for LLM inference backends.

    Example:
        >>> class CustomBackend(LLMBackend):
        ...     def generate(self, prompt: str) -> str:
        ...         return "summary"
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

    def cleanup(self) -> None:
        """Releases backend resources and unloads model from memory."""
        ...
