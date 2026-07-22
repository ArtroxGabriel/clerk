from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return

    logger.error("ffmpeg not found in PATH")
    raise RuntimeError("ffmpeg not found in PATH")


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
