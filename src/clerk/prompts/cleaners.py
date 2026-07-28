from __future__ import annotations

import re


NOISE_PATTERNS = re.compile(
    r"^(\s*(\[\s*(music|música|applause|cheering|laughter|noise|silence|audio|ruído)\s*\]|\(\s*(music|música|applause|cheering|laughter|noise|silence|audio|ruído)\s*\)|subtitles by|legendas por|obrigado por assistir|thanks for watching|\.+|\?+|,+)\s*)+$",
    re.IGNORECASE,
)
DEFAULT_MIN_TRANSCRIPT_WORDS = 3


def clean_srt_for_prompt(transcript: str) -> str:
    """Removes SRT sequence numbers and timestamp headers for clean LLM prompt input."""
    if not transcript or not transcript.strip():
        return ""

    lines: list[str] = []
    timestamp_pattern = re.compile(
        r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}"
    )
    number_pattern = re.compile(r"^\d+$")

    for line in transcript.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if number_pattern.match(line_str):
            continue
        if timestamp_pattern.match(line_str):
            continue
        lines.append(line_str)

    return "\n".join(lines).strip()


def clean_llm_output(text: str) -> str:
    """Strips any <<<text>>> delimiter tags echoed by the LLM."""
    if not text:
        return ""
    cleaned = re.sub(r"<\s*<\s*<[^>]+>\s*>\s*>", "", text)
    lines = [line for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def is_meaningful_transcript(transcript: str, min_words: int = DEFAULT_MIN_TRANSCRIPT_WORDS) -> bool:
    """Verifies whether a transcript contains meaningful content or is garbled/noise/minimal ASR output."""
    cleaned = clean_srt_for_prompt(transcript)
    if not cleaned or not cleaned.strip():
        return False

    if NOISE_PATTERNS.match(cleaned.strip()):
        return False

    text_no_annotations = re.sub(r"\[[^\]]*\]|\([^\)]*\)", "", cleaned).strip()
    words = [w for w in re.findall(r"\b[a-zA-ZÀ-ÿ0-9_-]+\b", text_no_annotations) if not w.isdigit()]

    if len(words) < min_words:
        return False

    unique_words = set(w.lower() for w in words)
    if len(words) >= min_words and len(unique_words) <= 1:
        return False

    return True
