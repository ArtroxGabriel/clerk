from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clerk.audio import (
    download_youtube_audio,
    download_youtube_subtitles,
    ensure_ffmpeg,
    ensure_yt_dlp,
    extract_audio,
    extract_youtube_id,
)


def test_extract_youtube_id() -> None:
    assert extract_youtube_id("https://www.youtube.com/watch?v=bWZndc9ycII") == "bWZndc9ycII"
    assert extract_youtube_id("https://youtu.be/bWZndc9ycII") == "bWZndc9ycII"
    assert extract_youtube_id("https://www.youtube.com/embed/bWZndc9ycII") == "bWZndc9ycII"
    assert extract_youtube_id("https://www.youtube.com/shorts/bWZndc9ycII") == "bWZndc9ycII"
    assert extract_youtube_id("invalid_url_without_id") is None


def test_ensure_ffmpeg_success() -> None:
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        # Should not raise any error
        ensure_ffmpeg()


def test_ensure_ffmpeg_failure() -> None:
    with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="ffmpeg not found in PATH"):
        ensure_ffmpeg()


def test_ensure_yt_dlp_success() -> None:
    with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
        ensure_yt_dlp()


def test_ensure_yt_dlp_failure() -> None:
    with patch("shutil.which", return_value=None), pytest.raises(RuntimeError, match="yt-dlp not found in PATH"):
        ensure_yt_dlp()


def test_download_youtube_audio_success(tmp_path: Path) -> None:
    mock_run = MagicMock()
    mock_run.returncode = 0
    mock_run.stdout = "/tmp/yt_O_Bug_de_Timezone.wav\n"

    with patch("shutil.which", return_value="/usr/bin/yt-dlp"), \
         patch("subprocess.run", return_value=mock_run) as mock_sub, \
         patch("pathlib.Path.exists", return_value=True):
        result_path = download_youtube_audio("https://www.youtube.com/watch?v=12345")
        assert result_path == Path("/tmp/yt_O_Bug_de_Timezone.wav")
        mock_sub.assert_called_once()
        cmd = mock_sub.call_args[0][0]
        assert "yt-dlp" in cmd
        assert "--restrict-filenames" in cmd
        assert "https://www.youtube.com/watch?v=12345" in cmd


def test_download_youtube_audio_uses_system_tempdir(tmp_path: Path) -> None:
    # Arrange
    expected_file = tmp_path / "yt_dQw4w9WgXcQ.wav"
    expected_file.write_text("wav data")
    mock_run = MagicMock(returncode=0, stdout="")

    # Act
    with (
        patch("shutil.which", return_value="/usr/bin/yt-dlp"),
        patch("subprocess.run", return_value=mock_run),
        patch("clerk.audio.tempfile.gettempdir", return_value=str(tmp_path)),
    ):
        result_path = download_youtube_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # Assert
    assert result_path == expected_file


def test_download_youtube_audio_failure() -> None:
    mock_run = MagicMock()
    mock_run.returncode = 1
    mock_run.stderr = "yt-dlp download failed error"

    with (
        patch("shutil.which", return_value="/usr/bin/yt-dlp"),
        patch("subprocess.run", return_value=mock_run),
        pytest.raises(RuntimeError, match="Failed to download YouTube audio"),
    ):
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
         patch("subprocess.run", return_value=mock_run), pytest.raises(RuntimeError, match="audio extraction failed"):
        extract_audio(input_file, output_file)


def test_download_youtube_subtitles_success(tmp_path: Path) -> None:
    # Arrange
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    srt_in_work = work_dir / "yt_Aula_Python.pt.srt"
    srt_in_work.write_text("1\n00:00:00,000 --> 00:00:02,000\nOlá mundo\n", encoding="utf-8")
    mock_run = MagicMock(returncode=0, stdout="", stderr="")

    # Act
    with (
        patch("shutil.which", return_value="/usr/bin/yt-dlp"),
        patch("clerk.audio.tempfile.mkdtemp", return_value=str(work_dir)),
        patch("clerk.audio.tempfile.gettempdir", return_value=str(tmp_path)),
        patch("subprocess.run", return_value=mock_run) as mock_sub,
    ):
        result_path = download_youtube_subtitles("https://www.youtube.com/watch?v=12345", language="pt")

    # Assert
    assert result_path == tmp_path / "yt_Aula_Python_transcript.srt"
    assert result_path is not None and result_path.exists()
    assert "Olá mundo" in result_path.read_text(encoding="utf-8")
    cmd = mock_sub.call_args[0][0]
    assert "--write-subs" in cmd
    assert "--write-auto-subs" in cmd
    assert "--skip-download" in cmd


def test_download_youtube_subtitles_returns_none_when_unavailable(tmp_path: Path) -> None:
    # Arrange
    work_dir = tmp_path / "work_empty"
    work_dir.mkdir()
    mock_run = MagicMock(returncode=0, stdout="", stderr="")

    # Act
    with (
        patch("shutil.which", return_value="/usr/bin/yt-dlp"),
        patch("clerk.audio.tempfile.mkdtemp", return_value=str(work_dir)),
        patch("subprocess.run", return_value=mock_run),
    ):
        result_path = download_youtube_subtitles("https://www.youtube.com/watch?v=12345", language="pt")

    # Assert
    assert result_path is None

