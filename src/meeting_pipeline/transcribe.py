from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def transcribe_file(
    audio_path: Path,
    model_name: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = "pt",
) -> tuple[str, dict]:
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

    lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(text)

    transcript = "\n".join(lines).strip()
    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }

    if transcript:
        return transcript, metadata

    logger.error("Empty transcript generated")
    raise RuntimeError("empty transcript")
