from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from clerk.audio import (
    download_youtube_audio,
    ensure_ffmpeg,
    ensure_yt_dlp,
    extract_audio,
)


def test_ensure_ffmpeg_success() -> None:
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        # Should not raise any error
        ensure_ffmpeg()


def test_ensure_ffmpeg_failure() -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="ffmpeg not found in PATH"):
            ensure_ffmpeg()


def test_ensure_yt_dlp_success() -> None:
    with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
        ensure_yt_dlp()


def test_ensure_yt_dlp_failure() -> None:
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="yt-dlp not found in PATH"):
            ensure_yt_dlp()


def test_download_youtube_audio_success(tmp_path: Path) -> None:
    mock_run = MagicMock()
    mock_run.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
         patch("subprocess.run", return_value=mock_run) as mock_sub, \
         patch("pathlib.Path.exists", return_value=True):
        result_path = download_youtube_audio("https://www.youtube.com/watch?v=12345")
        assert result_path.parent == Path("/tmp")
        assert result_path.name.startswith("yt_")
        assert result_path.name.endswith(".wav")
        mock_sub.assert_called_once()
        cmd = mock_sub.call_args[0][0]
        assert "yt-dlp" in cmd
        assert "https://www.youtube.com/watch?v=12345" in cmd


def test_download_youtube_audio_failure() -> None:
    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stderr = "yt-dlp download failed error"

    with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
         patch("subprocess.run", return_value=mock_run):
        with pytest.raises(RuntimeError, match="Failed to download YouTube audio"):
            download_youtube_audio("https://www.youtube.com/watch?v=invalid")


def test_extract_audio_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(Path("nonexistent.mp4"), Path("output.wav"))


def test_extract_audio_success(tmp_path: Path) -> None:
    input_file = tmp_path / "input.mp4"
    input_file.write_text("dummy content")
    output_file = tmp_path / "output.wav"

    mock_run = MagicMock()
    mock_run.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.run", return_value=mock_run) as mock_subprocess_run:
        result = extract_audio(input_file, output_file)
        assert result == output_file
        mock_subprocess_run.assert_called_once()
        args = mock_subprocess_run.call_args[0][0]
        assert "ffmpeg" in args
        assert str(input_file) in args
        assert str(output_file) in args


def test_extract_audio_ffmpeg_failure(tmp_path: Path) -> None:
    input_file = tmp_path / "input.mp4"
    input_file.write_text("dummy content")
    output_file = tmp_path / "output.wav"

    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stderr = "FFmpeg error message"

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("subprocess.run", return_value=mock_run):
        with pytest.raises(RuntimeError, match="audio extraction failed"):
            extract_audio(input_file, output_file)

