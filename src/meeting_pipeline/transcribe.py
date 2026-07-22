from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        secs += int(milliseconds // 1000)
        milliseconds = milliseconds % 1000
        if secs >= 60:
            minutes += int(secs // 60)
            secs = secs % 60
            if minutes >= 60:
                hours += int(minutes // 60)
                minutes = minutes % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def segments_to_srt(segments: list) -> tuple[str, str]:
    lines: list[str] = []
    srt_blocks: list[str] = []
    index = 1

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(text)
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        srt_blocks.append(f"{index}\n{start_str} --> {end_str}\n{text}\n")
        index += 1

    plain_text = "\n".join(lines).strip()
    srt_text = "\n".join(srt_blocks).strip()
    return plain_text, srt_text


def transcribe_file(
    audio_path: Path,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "pt",
) -> tuple[str, str, dict]:
    if not audio_path.exists():
        logger.error("Audio file does not exist: %s", audio_path)
        raise FileNotFoundError(audio_path)

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )

    # Convert generator to list to iterate for both SRT and plain text
    segments_list = list(segments)
    plain_text, srt_text = segments_to_srt(segments_list)

    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }

    if plain_text:
        return plain_text, srt_text, metadata

    logger.error("Empty transcript generated")
    raise RuntimeError("empty transcript")

