from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from clerk.pipeline import run_pipeline


def test_run_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp3"
    input_path.write_text("audio contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 120.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nMock transcription"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "input_normalized.wav") as mock_extract, \
         patch("clerk.pipeline.transcribe_file", return_value=("Mock transcription", mock_srt, mock_metadata)) as mock_transcribe, \
         patch("clerk.pipeline.summarize_transcript", return_value="Mock summary") as mock_summarize:

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
        )

        assert tx_path == output_dir / "input_transcript.srt"
        assert sum_path == output_dir / "input_meeting_points.md"
        assert metadata_res["language"] == "pt"

        mock_extract.assert_called_once_with(input_path, output_dir / "input_normalized.wav")
        mock_transcribe.assert_called_once_with(
            output_dir / "input_normalized.wav",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="pt",
            batch_size=2,
            verbose=False,
        )
        mock_summarize.assert_called_once_with(
            transcript="Mock transcription",
            model_name="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=False,
        )

        assert (output_dir / "input_transcript.srt").read_text(encoding="utf-8") == mock_srt + "\n"
        assert (output_dir / "input_meeting_points.md").read_text(encoding="utf-8") == "Mock summary\n"

        metadata_content = json.loads((output_dir / "input_metadata.json").read_text(encoding="utf-8"))
        assert metadata_content == mock_metadata


def test_run_pipeline_video_mode(tmp_path: Path) -> None:
    input_path = tmp_path / "presentation.mp4"
    input_path.write_text("video contents")
    output_dir = tmp_path / "output"

    mock_metadata = {"language": "pt", "duration": 60.0}
    mock_srt = "1\n00:00:00,000 --> 00:00:02,000\nVideo text"

    with patch("clerk.pipeline.extract_audio", return_value=output_dir / "presentation_normalized.wav"), \
         patch("clerk.pipeline.transcribe_file", return_value=("Video text", mock_srt, mock_metadata)), \
         patch("clerk.pipeline.summarize_transcript", return_value="Video summary"):

        tx_path, sum_path, metadata_res = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            llm_model="LiquidAI/lfm2.5-1.2b-instruct",
            language="pt",
            is_video=True,
        )

        assert tx_path == output_dir / "presentation_transcript.srt"
        assert sum_path == output_dir / "presentation_resume.md"
        assert (output_dir / "presentation_resume.md").read_text(encoding="utf-8") == "Video summary\n"

