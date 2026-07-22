from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from meeting_pipeline.cli import app


runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--target" in result.stdout


def test_cli_missing_argument() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_cli_nonexistent_file() -> None:
    result = runner.invoke(app, ["--target", "nonexistent_file.mp3"])
    assert result.exit_code != 0
    assert "Input file does not exist" in result.output


def test_cli_success_local_file(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    output_dir = tmp_path / "output_dir"

    with patch("meeting_pipeline.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"))) as mock_run:
        result = runner.invoke(
            app,
            [
                "--target",
                str(input_file),
                "--output-dir",
                str(output_dir),
                "--whisper-model",
                "tiny",
                "--llm-model",
                "LiquidAI/lfm2.5-1.2b-instruct",
                "--language",
                "pt",
                "--verbose"
            ]
        )
        assert result.exit_code == 0
        assert "Transcript: out/transcript.srt" in result.stdout
        assert "Meeting points: out/meeting_points.md" in result.stdout
        mock_run.assert_called_once_with(
            input_path=input_file,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
        )


def test_cli_success_youtube_url(tmp_path: Path) -> None:
    mock_temp_file = tmp_path / "yt_download.wav"
    mock_temp_file.write_text("yt audio")
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch("meeting_pipeline.cli.download_youtube_audio", return_value=mock_temp_file) as mock_dl, \
         patch("meeting_pipeline.cli.run_pipeline", return_value=(Path("out/transcript.srt"), Path("out/meeting_points.md"))):
        result = runner.invoke(app, ["--target", yt_url])
        assert result.exit_code == 0
        mock_dl.assert_called_once_with(yt_url)
        # Temp file should be deleted in finally block
        assert not mock_temp_file.exists()


def test_cli_keyboard_interrupt(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    with patch("meeting_pipeline.cli.run_pipeline", side_effect=KeyboardInterrupt()):
        result = runner.invoke(app, ["--target", str(input_file)])
        assert result.exit_code == 130
        assert "Process interrupted by user" in result.output


def test_cli_pipeline_failure(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    with patch("meeting_pipeline.cli.run_pipeline", side_effect=ValueError("Pipeline error")):
        result = runner.invoke(app, ["--target", str(input_file)])
        assert result.exit_code == 1

