from __future__ import annotations

import json
import logging
from pathlib import Path

from .audio import extract_audio
from .summarize import summarize_transcript
from .transcribe import transcribe_file

import time

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    whisper_model: str,
    whisper_device: str,
    whisper_compute_type: str,
    llm_model: str,
    language: str | None,
    is_video: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / "normalized.wav"
    transcript_path = output_dir / "transcript.srt"
    summary_path = output_dir / "meeting_points.md"
    metadata_path = output_dir / "transcript_metadata.json"

    t_start = time.perf_counter()

    logger.info("Starting audio extraction from %s...", input_path)
    t0 = time.perf_counter()
    normalized_audio = extract_audio(input_path, audio_path)
    t_audio = time.perf_counter() - t0
    logger.info("Audio extraction completed in %.2fs", t_audio)

    logger.info("Starting transcription...")
    t0 = time.perf_counter()
    plain_text_transcript, srt_transcript, metadata = transcribe_file(
        normalized_audio,
        model_name=whisper_model,
        device=whisper_device,
        compute_type=whisper_compute_type,
        language=language,
    )
    t_transcribe = time.perf_counter() - t0
    logger.info("Transcription completed in %.2fs", t_transcribe)

    logger.info("Starting summarization...")
    t0 = time.perf_counter()
    summary = summarize_transcript(
        transcript=plain_text_transcript,
        model_name=llm_model,
        language=language or "pt",
        is_video=is_video,
    )

    t_summarize = time.perf_counter() - t0
    logger.info("Summarization completed in %.2fs", t_summarize)

    t_total = time.perf_counter() - t_start
    logger.info("Total pipeline execution time: %.2fs", t_total)

    metadata["timings"] = {
        "audio_extraction_seconds": round(t_audio, 3),
        "transcription_seconds": round(t_transcribe, 3),
        "summarization_seconds": round(t_summarize, 3),
        "total_seconds": round(t_total, 3),
    }

    transcript_path.write_text(srt_transcript + "\n", encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Transcript written to %s", transcript_path)
    logger.info("Summary written to %s", summary_path)

    return transcript_path, summary_path


