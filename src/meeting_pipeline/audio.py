from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import uuid

logger = logging.getLogger(__name__)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return

    logger.error("ffmpeg not found in PATH")
    raise RuntimeError("ffmpeg not found in PATH")


def ensure_yt_dlp() -> None:
    if shutil.which("yt-dlp"):
        return

    logger.error("yt-dlp not found in PATH")
    raise RuntimeError("yt-dlp not found in PATH")


def download_youtube_audio(url: str) -> Path:
    ensure_yt_dlp()
    temp_dir = Path("/tmp")
    output_filename = f"meeting_yt_{uuid.uuid4().hex}"
    output_path = temp_dir / f"{output_filename}.wav"

    command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(temp_dir / f"{output_filename}.%(ext)s"),
        url,
    ]

    logger.info("Downloading YouTube audio from %s...", url)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("yt-dlp failed: %s", result.stderr.strip())
        raise RuntimeError(f"Failed to download YouTube audio: {result.stderr.strip()}")

    if not output_path.exists():
        logger.error("Downloaded file not found at %s", output_path)
        raise RuntimeError("Downloaded YouTube audio file not found")

    return output_path



def extract_audio(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        raise FileNotFoundError(input_path)

    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return output_path

    logger.error("ffmpeg failed: %s", result.stderr.strip())
    raise RuntimeError("audio extraction failed")
