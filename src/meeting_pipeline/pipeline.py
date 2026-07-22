from __future__ import annotations

import json
import logging
from pathlib import Path

from .audio import extract_audio
from .summarize import summarize_transcript
from .transcribe import transcribe_file

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    llm_model: str,
    language: str | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "normalized.wav"
    transcript_path = output_dir / "transcript.txt"
    summary_path = output_dir / "meeting_points.md"
    metadata_path = output_dir / "transcript_metadata.json"

    normalized_audio = extract_audio(input_path, audio_path)
    transcript, metadata = transcribe_file(
        normalized_audio,
        model_name=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute_type,
        language=language,
    )

    summary = summarize_transcript(
        transcript=transcript,
        model_name=llm_model,
    )

    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Transcript written to %s", transcript_path)
    logger.info("Summary written to %s", summary_path)

    return transcript_path, summary_path
