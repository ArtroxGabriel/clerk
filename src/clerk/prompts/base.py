from __future__ import annotations

from typing import Protocol


LANGUAGE_NAMES: dict[str, str] = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}


def get_language_name(lang_code: str | None) -> str:
    if not lang_code:
        return "Portuguese"
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)


class PromptStrategy(Protocol):
    def build_summary_prompt(self, transcript: str, language: str, is_video: bool) -> str:
        ...

    def build_consolidation_prompt(self, category: str, items: str, language: str) -> str:
        ...


class PromptManager:
    """Factory for obtaining prompt strategies tailored to CPU vs GPU models or custom user templates."""

    @staticmethod
    def get_strategy(
        is_gpu_model: bool = False,
        custom_prompt: str | None = None,
        custom_consolidation_prompt: str | None = None,
    ) -> PromptStrategy:
        from .cpu import CpuPromptStrategy
        from .custom import CustomPromptStrategy
        from .gpu import GpuPromptStrategy

        fallback = GpuPromptStrategy() if is_gpu_model else CpuPromptStrategy()

        if custom_prompt or custom_consolidation_prompt:
            return CustomPromptStrategy(
                summary_template=custom_prompt,
                consolidation_template=custom_consolidation_prompt,
                fallback_strategy=fallback,
            )

        return fallback

