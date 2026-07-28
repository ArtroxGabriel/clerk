from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import PromptStrategy


class CustomPromptStrategy:
    """Strategy supporting user-defined custom prompt templates for summary and consolidation stages."""

    def __init__(
        self,
        summary_template: str | None = None,
        consolidation_template: str | None = None,
        fallback_strategy: PromptStrategy | None = None,
    ) -> None:
        self.summary_template = summary_template
        self.consolidation_template = consolidation_template
        self.fallback_strategy = fallback_strategy

    def build_summary_prompt(self, transcript: str, language: str, is_video: bool) -> str:
        if not self.summary_template:
            if self.fallback_strategy:
                return self.fallback_strategy.build_summary_prompt(
                    transcript=transcript,
                    language=language,
                    is_video=is_video,
                )
            raise ValueError("No summary template or fallback strategy provided.")

        template = self.summary_template
        if "{transcript}" in template:
            prompt = template.replace("{transcript}", transcript)
            prompt = prompt.replace("{language}", language)
            return prompt

        # If user template omits {transcript} placeholder, append transcript with security delimiters
        formatted = template.replace("{language}", language)
        return (
            f"{formatted}\n\nTranscript:\n<<<TRANSCRIPT>>>\n{transcript}\n<<<END TRANSCRIPT>>>"
        )

    def build_consolidation_prompt(self, category: str, items: str, language: str) -> str:
        if not self.consolidation_template:
            if self.fallback_strategy:
                return self.fallback_strategy.build_consolidation_prompt(
                    category=category,
                    items=items,
                    language=language,
                )
            raise ValueError("No consolidation template or fallback strategy provided.")

        template = self.consolidation_template
        if "{items}" in template:
            prompt = template.replace("{items}", items)
            prompt = prompt.replace("{category}", category)
            prompt = prompt.replace("{language}", language)
            return prompt

        formatted = template.replace("{category}", category).replace("{language}", language)
        return f"{formatted}\n\nItems to consolidate:\n<<<ITEMS>>>\n{items}\n<<<END ITEMS>>>"
