from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from meeting_pipeline.cli import app


runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_cli_missing_argument() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0


def test_cli_nonexistent_file() -> None:
    result = runner.invoke(app, ["nonexistent_file.mp3"])
    assert result.exit_code != 0


def test_cli_success(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")
    output_dir = tmp_path / "output_dir"

    with patch("meeting_pipeline.cli.run_pipeline", return_value=(Path("out/transcript.txt"), Path("out/meeting_points.md"))) as mock_run:
        result = runner.invoke(
            app,
            [
                str(input_file),
                "--output-dir",
                str(output_dir),
                "--whisper-model",
                "tiny",
                "--llm-model",
                "gemma:2b",
                "--language",
                "pt",
                "--verbose"
            ]
        )
        assert result.exit_code == 0
        assert "Transcript: out/transcript.txt" in result.stdout
        assert "Meeting points: out/meeting_points.md" in result.stdout
        mock_run.assert_called_once_with(
            input_path=input_file,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="gemma:2b",
            language="pt",
        )


def test_cli_pipeline_failure(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("mock audio content")

    with patch("meeting_pipeline.cli.run_pipeline", side_effect=ValueError("Pipeline error")):
        result = runner.invoke(app, [str(input_file)])
        assert result.exit_code == 1
