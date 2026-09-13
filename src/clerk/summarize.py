from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from .backends import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    LLMBackend,
    OllamaBackend,
    get_backend,
)
from .backends.ollama import UNLOAD_TIMEOUT_SECONDS
from .prompts import (
    PromptManager,
    PromptStrategy,
    clean_srt_for_prompt,
    get_language_name,
    is_meaningful_transcript,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORDS_PER_CHUNK = 2000


def split_transcript_smart(
    transcript: str,
    max_words: int = DEFAULT_MAX_WORDS_PER_CHUNK,
) -> list[str]:
    """Splits transcript into chunks without breaking mid-sentence, word, or phrase."""
    cleaned = clean_srt_for_prompt(transcript)
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    units: list[str] = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        sub_sentences = re.split(r"(?<=[.!?])\s+", line_str)
        for s in sub_sentences:
            if s.strip():
                units.append(s.strip())

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for unit in units:
        words = unit.split()
        unit_word_count = len(words)
        if not words:
            continue

        if unit_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            clauses = re.split(r"(?<=[,;:])\s+", unit)
            if len(clauses) == 1:
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i : i + max_words]))
            else:
                sub_chunk: list[str] = []
                sub_count = 0
                for clause in clauses:
                    c_words = clause.split()
                    if not c_words:
                        continue
                    if sub_count + len(c_words) > max_words and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [clause]
                        sub_count = len(c_words)
                    else:
                        sub_chunk.append(clause)
                        sub_count += len(c_words)
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
            continue

        if current_word_count + unit_word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [unit]
            current_word_count = unit_word_count
        else:
            current_chunk.append(unit)
            current_word_count += unit_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


split_transcript_by_words = split_transcript_smart


@dataclass(frozen=True)
class SummaryConfig:
    sections: list[str]
    primary_section: str


VIDEO_CONFIG = SummaryConfig(
    sections=["Resumo geral", "Principais tópicos", "Momentos importantes", "Conclusões ou mensagens finais"],
    primary_section="Resumo geral",
)
MEETING_CONFIG = SummaryConfig(
    sections=["Pontos principais", "Decisões", "Ações", "Pendências"],
    primary_section="Pontos principais",
)


def get_summary_config(is_video: bool) -> SummaryConfig:
    return VIDEO_CONFIG if is_video else MEETING_CONFIG


def format_empty_fallback(section: str, primary_section: str) -> str:
    return f"- Nenhum {section.lower()} registrado." if section == primary_section else "- Nenhuma registrada."


def unload_ollama_model(
    model_name: str,
    base_url: str,
    timeout_seconds: float = UNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Unload model from Ollama memory by posting keep_alive: 0."""
    OllamaBackend(model_name=model_name, base_url=base_url, timeout_seconds=timeout_seconds).cleanup()


def parse_summary_sections(summary: str, is_video: bool = False) -> dict[str, list[str]]:
    config = get_summary_config(is_video)
    sections: dict[str, list[str]] = {sec: [] for sec in config.sections}
    current_section: str | None = None

    regex_pattern = r"^##\s*(" + "|".join(re.escape(sec) for sec in config.sections) + r")\b"

    for line in summary.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue

        header_match = re.match(regex_pattern, line_strip, re.IGNORECASE)

        if header_match:
            matched_name = header_match.group(1).lower()
            for key in sections.keys():
                if key.lower() == matched_name:
                    current_section = key
                    break
            continue

        if current_section:
            item_match = re.match(r"^([-*]|\d+\.)\s*(.*)$", line_strip)
            content = item_match.group(2).strip() if item_match else line_strip
            lower_content = content.lower()

            if content and not any(
                phrase in lower_content
                for phrase in [
                    "nenhuma registrada",
                    "nenhum ponto",
                    "nenhum resumo",
                    "não há",
                    "none registered",
                    "no summary",
                ]
            ):
                sections[current_section].append(content)

    return sections


def _call_ollama_generate(
    prompt: str,
    model_name: str,
    base_url: str,
    timeout_seconds: float,
) -> str:
    backend = OllamaBackend(model_name=model_name, base_url=base_url, timeout_seconds=timeout_seconds)
    return backend.generate(prompt)


def _process_chunks(
    chunks: list[str],
    prompt_strategy: PromptStrategy,
    backend: LLMBackend,
    lang_name: str,
    is_video: bool,
    config: SummaryConfig,
) -> dict[str, list[str]]:
    """Generates and parses section items for each transcript chunk."""
    combined_sections: dict[str, list[str]] = {sec: [] for sec in config.sections}
    for i, chunk in enumerate(chunks):
        logger.info("Summarizing chunk %d/%d...", i + 1, len(chunks))
        prompt = prompt_strategy.build_summary_prompt(
            transcript=chunk,
            language=lang_name,
            is_video=is_video,
        )
        summary = backend.generate(prompt)
        if summary:
            chunk_sections = parse_summary_sections(summary, is_video=is_video)
            for sec, items in chunk_sections.items():
                if sec in combined_sections:
                    combined_sections[sec].extend(items)
    return combined_sections


def _consolidate_sections(
    combined_sections: dict[str, list[str]],
    prompt_strategy: PromptStrategy,
    backend: LLMBackend,
    lang_name: str,
    config: SummaryConfig,
) -> str:
    """Consolidates accumulated section items into structured markdown."""
    logger.info("Consolidating section summaries...")
    summaries: dict[str, str] = {}
    for sec in config.sections:
        items = combined_sections.get(sec, [])
        if not items:
            summaries[sec] = format_empty_fallback(sec, config.primary_section)
            continue

        items_text = "\n".join(f"- {item}" for item in items)
        prompt = prompt_strategy.build_consolidation_prompt(
            category=sec,
            items=items_text,
            language=lang_name,
        )
        content = backend.generate(prompt)
        summaries[sec] = content or format_empty_fallback(sec, config.primary_section)

    return "\n\n".join(f"## {sec}\n{summaries[sec]}" for sec in config.sections).strip()


def summarize_transcript(
    transcript: str,
    model_name: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_words_per_chunk: int = DEFAULT_MAX_WORDS_PER_CHUNK,
    language: str = "pt",
    is_video: bool = False,
    is_gpu_model: bool = False,
    custom_prompt: str | None = None,
    custom_consolidation_prompt: str | None = None,
    backend: LLMBackend | None = None,
) -> str:
    """Generates structured summary for a transcript using an LLM backend."""
    cleaned_transcript = clean_srt_for_prompt(transcript)
    if not transcript or not transcript.strip() or not is_meaningful_transcript(transcript):
        logger.error("Transcript is empty, garbled, or contains insufficient speech content")
        raise ValueError("transcript is empty")

    lang_name = get_language_name(language)
    config = get_summary_config(is_video)

    if not is_gpu_model:
        lower_m = model_name.lower()
        if any(kw in lower_m for kw in ("llama3", "gpu", "cuda", "8b")):
            is_gpu_model = True

    prompt_strategy = PromptManager.get_strategy(
        is_gpu_model=is_gpu_model,
        custom_prompt=custom_prompt,
        custom_consolidation_prompt=custom_consolidation_prompt,
    )
    active_backend = backend or get_backend(
        "ollama",
        model_name=model_name,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )

    words = cleaned_transcript.split()
    try:
        if len(words) <= max_words_per_chunk:
            prompt = prompt_strategy.build_summary_prompt(
                transcript=cleaned_transcript,
                language=lang_name,
                is_video=is_video,
            )
            content = active_backend.generate(prompt)
            if content:
                return content
            logger.error("LLM backend returned empty response")
            raise RuntimeError("empty summary")

        logger.info(
            "Transcript length (%d words) exceeds chunk size (%d). Processing in chunks...",
            len(words),
            max_words_per_chunk,
        )
        chunks = split_transcript_smart(cleaned_transcript, max_words_per_chunk)
        combined_sections = _process_chunks(chunks, prompt_strategy, active_backend, lang_name, is_video, config)
        return _consolidate_sections(combined_sections, prompt_strategy, active_backend, lang_name, config)
    finally:
        active_backend.cleanup()
