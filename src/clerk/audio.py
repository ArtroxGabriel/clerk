from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_RATE = "16000"
DEFAULT_CHANNELS = "1"
DEFAULT_AUDIO_CODEC = "pcm_s16le"


def ensure_binary(binary_name: str) -> None:
    if shutil.which(binary_name):
        return

    logger.error("%s not found in PATH", binary_name)
    raise RuntimeError(f"{binary_name} not found in PATH")


def ensure_ffmpeg() -> None:
    ensure_binary("ffmpeg")


def ensure_yt_dlp() -> None:
    ensure_binary("yt-dlp")


def extract_youtube_id(url: str) -> str | None:
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def download_youtube_subtitles(url: str, language: str = "pt") -> Path | None:
    """Attempts to download embedded or auto-generated YouTube subtitles (.srt) via yt-dlp.

    Returns the path to the downloaded .srt file in the system temp dir, or None if unavailable.
    """
    ensure_yt_dlp()
    work_dir = Path(tempfile.mkdtemp(prefix="clerk_subs_"))
    lang_code = "pt" if not language or language.lower() == "auto" else language.lower()
    sub_langs = f"{lang_code}.*,{lang_code},en.*,en" if lang_code != "en" else "en.*,en"

    command = [
        "yt-dlp",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        sub_langs,
        "--sub-format",
        "srt",
        "--convert-subs",
        "srt",
        "--skip-download",
        "--restrict-filenames",
        "-o",
        str(work_dir / "yt_%(title)s.%(ext)s"),
        url,
    ]

    try:
        logger.info("Checking for existing YouTube subtitles (%s)...", lang_code)
        subprocess.run(command, capture_output=True, text=True, check=False)
        srt_files = sorted(work_dir.glob("*.srt"))
        for srt_file in srt_files:
            if srt_file.stat().st_size > 0:
                base_stem = re.sub(r"\.[a-zA-Z0-9_-]+$", "", srt_file.stem)
                dest_path = Path(tempfile.gettempdir()) / f"{base_stem}_transcript.srt"
                shutil.move(str(srt_file), str(dest_path))
                logger.info("Downloaded YouTube subtitles to %s", dest_path)
                return dest_path
        return None
    except Exception as err:
        logger.debug("Subtitle fast-path unavailable: %s", err)
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def download_youtube_audio(url: str) -> Path:
    ensure_yt_dlp()
    temp_dir = Path(tempfile.gettempdir())
    yt_id = extract_youtube_id(url)
    default_filename = f"yt_{yt_id}" if yt_id else f"yt_{uuid.uuid4().hex[:8]}"

    command = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--restrict-filenames",
        "--print",
        "after_move:filepath",
        "-o",
        str(temp_dir / "yt_%(title)s.%(ext)s"),
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

    if result.stdout.strip():
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        if lines:
            candidate = Path(lines[-1])
            if candidate.exists():
                return candidate

    fallback_path = temp_dir / f"{default_filename}.wav"
    if fallback_path.exists():
        return fallback_path

    logger.error("Downloaded file not found")
    raise RuntimeError("Downloaded YouTube audio file not found")



def extract_audio(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        raise FileNotFoundError(input_path)

    ensure_ffmpeg()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error("Permission denied creating output directory %s: %s", output_path.parent, e)
        raise PermissionError(f"Permission denied creating directory {output_path.parent}. Check write permissions.") from e

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        DEFAULT_AUDIO_CODEC,
        "-ar",
        DEFAULT_SAMPLE_RATE,
        "-ac",
        DEFAULT_CHANNELS,
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

    stderr_msg = result.stderr.strip()
    logger.error("ffmpeg failed: %s", stderr_msg)
    if "Permission denied" in stderr_msg:
        raise PermissionError(f"Permission denied writing to {output_path}. Check output directory permissions.")
    raise RuntimeError("audio extraction failed")

