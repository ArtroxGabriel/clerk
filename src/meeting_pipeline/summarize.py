from __future__ import annotations

import logging
import os

import httpx

import re

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You will receive the transcript of a meeting.
Provide an objective summary in {language}, using only the explicit content from the transcript.

Mandatory format:
## Pontos principais
- ...

## Decisões
- ...

## Ações
- ...

## Pendências
- ...

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
- ...

## Principais tópicos
- ...

## Momentos importantes
- ...

## Conclusões ou mensagens finais
- ...

Do not invent missing facts.
Transcript:
{transcript}
""".strip()

CONSOLIDATE_PROMPT_TEMPLATE = """
You will receive a list of items for the category '{category}' extracted from different parts of a meeting transcript.
Your task is to consolidate these items into a single, concise list in {language} without duplicates or redundancies.

Keep only the explicit facts provided in the items list. Do not add new facts or assumptions.
If the list is empty, respond only with: - Nenhuma registrada. (or - Nenhum ponto principal registrado. for category Pontos principais).

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


def split_transcript_by_words(transcript: str, max_words: int = 2000) -> list[str]:
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


def parse_summary_sections(summary: str, is_video: bool = False) -> dict[str, list[str]]:
    expected_sections = (
        ["Resumo geral", "Principais tópicos", "Momentos importantes", "Conclusões ou mensagens finais"]
        if is_video
        else ["Pontos principais", "Decisões", "Ações", "Pendências"]
    )
    sections: dict[str, list[str]] = {sec: [] for sec in expected_sections}
    current_section: str | None = None

    for line in summary.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue

        if is_video:
            header_match = re.match(
                r"^##\s*(Resumo geral|Principais tópicos|Momentos importantes|Conclusões ou mensagens finais)\b",
                line_strip,
                re.IGNORECASE,
            )
        else:
            header_match = re.match(
                r"^##\s*(Pontos principais|Decisões|Ações|Pendências)\b",
                line_strip,
                re.IGNORECASE,
            )

        if header_match:
            matched_name = header_match.group(1).lower()
            for key in sections.keys():
                if key.lower() == matched_name:
                    current_section = key
                    break
            continue

        if current_section:
            if line_strip.startswith(("-", "*", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                item_match = re.match(r"^([-*]|\d+\.)\s*(.*)$", line_strip)
                if item_match:
                    content = item_match.group(2).strip()
                    lower_content = content.lower()
                    if content and not any(
                        phrase in lower_content
                        for phrase in ["nenhuma registrada", "nenhum ponto", "não há", "none registered"]
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

    if response.status_code != 200:
        logger.error("Ollama request failed: %s %s", response.status_code, response.text)
        raise RuntimeError("ollama request failed")

    data = response.json()
    content = data.get("response", "").strip()
    return content


def summarize_transcript(
    transcript: str,
    model_name: str = "LiquidAI/lfm2.5-1.2b-instruct",
    base_url: str | None = None,
    timeout_seconds: float = 300.0,
    max_words_per_chunk: int = 2000,
    language: str = "pt",
    is_video: bool = False,
) -> str:
    if base_url is None:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    if not transcript.strip():
        logger.error("Transcript is empty")
        raise ValueError("transcript is empty")

    lang_name = get_language_name(language)
    prompt_template = VIDEO_SUMMARY_PROMPT_TEMPLATE if is_video else PROMPT_TEMPLATE
    expected_sections = (
        ["Resumo geral", "Principais tópicos", "Momentos importantes", "Conclusões ou mensagens finais"]
        if is_video
        else ["Pontos principais", "Decisões", "Ações", "Pendências"]
    )

    words = transcript.split()
    if len(words) <= max_words_per_chunk:
        content = _call_ollama_generate(
            prompt=prompt_template.format(transcript=transcript, language=lang_name),
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

    combined_sections: dict[str, list[str]] = {sec: [] for sec in expected_sections}

    for i, chunk in enumerate(chunks):
        logger.info("Summarizing chunk %d/%d...", i + 1, len(chunks))
        chunk_summary = _call_ollama_generate(
            prompt=prompt_template.format(transcript=chunk, language=lang_name),
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
            consolidated_summaries[sec] = (
                "- Nenhuma registrada."
                if sec != expected_sections[0]
                else f"- Nenhum {expected_sections[0].lower()} registrado."
            )
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
            consolidated_content = (
                "- Nenhuma registrada."
                if sec != expected_sections[0]
                else f"- Nenhum {expected_sections[0].lower()} registrado."
            )

        consolidated_summaries[sec] = consolidated_content

    final_parts = [f"## {sec}\n{consolidated_summaries[sec]}" for sec in expected_sections]
    return "\n\n".join(final_parts).strip()

