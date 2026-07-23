from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)


DEFAULT_LLM_MODEL = "LiquidAI/lfm2.5-1.2b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 300.0
UNLOAD_TIMEOUT_SECONDS = 10.0
PULL_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_WORDS_PER_CHUNK = 2000

PROMPT_TEMPLATE = """
You will receive the transcript of a meeting.
Provide an objective summary in {language}, using only the explicit content from the transcript.

Mandatory format:
## Pontos principais
- [Key discussion point 1]

## Decisões
- [Decision made or 'Nenhuma registrada.']

## Ações
- [Action item or 'Nenhuma registrada.']

## Pendências
- [Pending issue or 'Nenhuma registrada.']

Do not invent missing facts.
Transcript:
{transcript}
""".strip()

VIDEO_SUMMARY_PROMPT_TEMPLATE = """
You will receive the transcript of a video.
Provide an objective and structured summary in {language}, using only the explicit content from the transcript.

Focus on the main ideas, key explanations, and important moments presented in the video.

Mandatory format:
## Resumo geral
- [A concise overview of what the video is about]

## Principais tópicos
- [Key topic 1]

## Momentos importantes
- [Key explanation or important moment 1]

## Conclusões ou mensagens finais
- [Final takeaway or conclusion]

Do not invent missing facts.
Transcript:
{transcript}
""".strip()

CONSOLIDATE_PROMPT_TEMPLATE = """
You will receive a list of items for the category '{category}' extracted from different parts of a meeting transcript.
Your task is to consolidate these items into a single, concise list in {language} without duplicates or redundancies.

Keep only the explicit facts provided in the items list. Do not add new facts or assumptions.
If the list is empty, respond only with: - Nenhuma registrada.

Mandatory format (return ONLY the list of topics):
- Consolidated item 1
- Consolidated item 2

Items to consolidate:
{items}
""".strip()

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


def split_transcript_by_words(
    transcript: str,
    max_words: int = DEFAULT_MAX_WORDS_PER_CHUNK,
) -> list[str]:
    lines = transcript.splitlines()
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for line in lines:
        words = line.split()
        if not words:
            continue

        line_word_count = len(words)

        if line_word_count > max_words:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            for i in range(0, line_word_count, max_words):
                chunk_words = words[i : i + max_words]
                chunks.append(" ".join(chunk_words))
            continue

        if current_word_count + line_word_count > max_words:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_word_count = line_word_count
        else:
            current_chunk.append(line)
            current_word_count += line_word_count

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


@dataclass(frozen=True)
class SummaryConfig:
    sections: list[str]
    prompt_template: str
    primary_section: str


def get_summary_config(is_video: bool) -> SummaryConfig:
    if is_video:
        return SummaryConfig(
            sections=[
                "Resumo geral",
                "Principais tópicos",
                "Momentos importantes",
                "Conclusões ou mensagens finais",
            ],
            prompt_template=VIDEO_SUMMARY_PROMPT_TEMPLATE,
            primary_section="Resumo geral",
        )
    return SummaryConfig(
        sections=["Pontos principais", "Decisões", "Ações", "Pendências"],
        prompt_template=PROMPT_TEMPLATE,
        primary_section="Pontos principais",
    )


def format_empty_fallback(section: str, primary_section: str) -> str:
    if section == primary_section:
        return f"- Nenhum {section.lower()} registrado."
    return "- Nenhuma registrada."


def unload_ollama_model(
    model_name: str,
    base_url: str,
    timeout_seconds: float = UNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Unload model from Ollama memory by posting keep_alive: 0."""
    payload = {
        "model": model_name,
        "keep_alive": 0,
    }
    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            client.post("/api/generate", json=payload)
        logger.info("Unloaded Ollama model '%s' from memory", model_name)
    except Exception as e:
        logger.warning("Failed to unload Ollama model '%s': %s", model_name, e)


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
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        response = client.post("/api/generate", json=payload)

        if response.status_code != 200 and ("not found" in response.text.lower() or response.status_code == 404):
            logger.info("Model '%s' not found locally in Ollama. Attempting automatic model pull...", model_name)
            pull_resp = client.post("/api/pull", json={"name": model_name, "stream": False}, timeout=PULL_TIMEOUT_SECONDS)
            if pull_resp.status_code == 200:
                logger.info("Model '%s' successfully pulled. Resuming generation...", model_name)
                response = client.post("/api/generate", json=payload)

    if response.status_code != 200:
        logger.error("Ollama request failed: %s %s", response.status_code, response.text)
        raise RuntimeError("ollama request failed")

    data = response.json()
    content = data.get("response", "").strip()
    return content


def summarize_transcript(
    transcript: str,
    model_name: str = DEFAULT_LLM_MODEL,
    base_url: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_words_per_chunk: int = DEFAULT_MAX_WORDS_PER_CHUNK,
    language: str = "pt",
    is_video: bool = False,
) -> str:
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    if not transcript.strip():
        logger.error("Transcript is empty")
        raise ValueError("transcript is empty")

    lang_name = get_language_name(language)
    config = get_summary_config(is_video)

    words = transcript.split()
    try:
        if len(words) <= max_words_per_chunk:
            content = _call_ollama_generate(
                prompt=config.prompt_template.format(transcript=transcript, language=lang_name),
                model_name=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            if content:
                return content
            logger.error("Ollama returned empty response")
            raise RuntimeError("empty summary")

        logger.info(
            "Transcript length (%d words) exceeds chunk size (%d words). Processing in chunks...",
            len(words),
            max_words_per_chunk,
        )
        chunks = split_transcript_by_words(transcript, max_words_per_chunk)

        combined_sections: dict[str, list[str]] = {sec: [] for sec in config.sections}

        for i, chunk in enumerate(chunks):
            logger.info("Summarizing chunk %d/%d...", i + 1, len(chunks))
            chunk_summary = _call_ollama_generate(
                prompt=config.prompt_template.format(transcript=chunk, language=lang_name),
                model_name=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            if chunk_summary:
                chunk_sections = parse_summary_sections(chunk_summary, is_video=is_video)
                for sec, items in chunk_sections.items():
                    if sec in combined_sections:
                        combined_sections[sec].extend(items)

        logger.info("Consolidating section summaries...")
        consolidated_summaries: dict[str, str] = {}
        for sec, items in combined_sections.items():
            if not items:
                consolidated_summaries[sec] = format_empty_fallback(sec, config.primary_section)
                continue

            items_text = "\n".join(f"- {item}" for item in items)
            prompt = CONSOLIDATE_PROMPT_TEMPLATE.format(
                category=sec, items=items_text, language=lang_name
            )

            consolidated_content = _call_ollama_generate(
                prompt=prompt,
                model_name=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )

            if not consolidated_content:
                consolidated_content = format_empty_fallback(sec, config.primary_section)

            consolidated_summaries[sec] = consolidated_content

        final_parts = [f"## {sec}\n{consolidated_summaries[sec]}" for sec in config.sections]
        return "\n\n".join(final_parts).strip()
    finally:
        unload_ollama_model(model_name=model_name, base_url=base_url)


