from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from clerk.config import ResolvedConfig
from clerk.ui import (
    create_transcription_progress,
    get_console,
    get_err_console,
    print_step_complete,
    prompt_recovery_menu,
    render_config_panel,
    render_markdown_preview,
    render_startup_banner,
    render_status_summary,
    step_spinner,
)


def _make_dummy_config(**kwargs) -> ResolvedConfig:
    defaults = {
        "target": "sample.mp4",
        "output_dir": Path("output"),
        "selected_preset": "gpu",
        "whisper_model": "large-v3-turbo",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 4,
        "llm_model": "gemma4:e2b",
        "language": "pt",
        "is_url": False,
        "is_video": True,
        "transcribe_only": False,
        "summarize_only": False,
        "custom_prompt": None,
        "custom_consolidation_prompt": None,
        "resume": False,
        "force": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return ResolvedConfig(**defaults)


def test_get_console_returns_console() -> None:
    console = get_console()
    assert isinstance(console, Console)
    err_console = get_err_console()
    assert isinstance(err_console, Console)


def test_render_startup_banner() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)

    render_startup_banner(
        target="sample.mp4",
        preset="gpu",
        whisper_model="large-v3-turbo",
        llm_model="gemma4:e2b",
        device="cuda",
        console=test_console,
    )

    output = buf.getvalue()
    assert "clerk" in output
    assert "sample.mp4" in output
    assert "large-v3-turbo" in output
    assert "gemma4:e2b" in output
    assert "cuda" in output.lower()


def test_render_config_panel() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)
    cfg = _make_dummy_config()

    render_config_panel(cfg, console=test_console)
    output = buf.getvalue()

    assert "Configuration" in output
    assert "sample.mp4" in output
    assert "large-v3-turbo" in output
    assert "gemma4:e2b" in output
    assert "Video" in output


def test_render_config_panel_meeting_mode() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)
    cfg = _make_dummy_config(is_video=False, transcribe_only=True)

    render_config_panel(cfg, console=test_console)
    output = buf.getvalue()

    assert "Meeting" in output
    assert "Transcribe Only" in output


def test_render_status_summary() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)

    metadata = {
        "language": "pt",
        "language_probability": 0.98,
        "duration": 125.4,
        "duration_after_vad": 110.2,
        "timings": {
            "audio_extraction_seconds": 1.2,
            "transcription_seconds": 10.5,
            "summarization_seconds": 4.3,
            "total_seconds": 16.0,
        },
        "models": {
            "whisper_model": "large-v3-turbo",
            "whisper_device": "cuda",
            "whisper_compute_type": "float16",
            "whisper_batch_size": 4,
            "llm_model": "gemma4:e2b",
        },
        "word_counts": {
            "transcript_words": 1500,
            "summary_words": 300,
        },
        "output_files": {
            "metadata_path": "output/transcript_metadata.json",
        },
    }

    render_status_summary(
        transcript_path=Path("output/test.srt"),
        summary_path=Path("output/test_resume.md"),
        output_dir=Path("output"),
        metadata=metadata,
        verbose=True,
        console=test_console,
    )

    output = buf.getvalue()
    assert "Pipeline Status" in output
    assert "test.srt" in output
    assert "test_resume.md" in output
    assert "large-v3-turbo" in output
    assert "gemma4:e2b" in output
    assert "1500" in output
    assert "300" in output


def test_render_status_summary_skipped_models() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)

    metadata = {
        "models": {
            "whisper_model": "skipped",
            "llm_model": "skipped",
        },
        "timings": {},
        "word_counts": {},
        "output_files": {},
    }

    render_status_summary(
        transcript_path=None,
        summary_path=None,
        output_dir=Path("output"),
        metadata=metadata,
        verbose=False,
        console=test_console,
    )

    output = buf.getvalue()
    assert "Skipped" in output
    assert "N/A (skipped)" in output


def test_render_markdown_preview() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)

    sample_md = "# Pauta\n- Discutido projeto clerk v2."
    render_markdown_preview(sample_md, title="Ata de Reunião", console=test_console)

    output = buf.getvalue()
    assert "Ata de Reunião" in output
    assert "Pauta" in output
    assert "Discutido projeto clerk v2" in output


def test_step_spinner() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    with step_spinner("Normalizing audio...", console=test_console):
        pass  # Block runs inside spinner context


def test_prompt_recovery_menu_non_interactive() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)

    llm, whisper, _compute, _device, retry = prompt_recovery_menu(
        error=RuntimeError("cuda out of memory"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=False,
        console=test_console,
    )

    assert retry is False
    assert llm == "gemma4:e2b"
    assert whisper == "large-v3-turbo"


def test_prompt_recovery_menu_libcublas_hint() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    prompt_recovery_menu(
        error=RuntimeError("Library libcublas.so.12 is not found"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=False,
        console=test_console,
    )

    output = buf.getvalue()
    assert "libcublas" in output.lower() or "cuda" in output.lower()


@patch("questionary.select")
@patch("questionary.text")
def test_prompt_recovery_menu_options(mock_text: MagicMock, mock_select: MagicMock) -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    # Option 1: LLM model change
    mock_select.return_value.ask.return_value = "1"
    mock_text.return_value.ask.return_value = "llama3.1:8b"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is True
    assert llm == "llama3.1:8b"

    # Option 2: Whisper model change
    mock_select.return_value.ask.return_value = "2"
    mock_text.return_value.ask.return_value = "small"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is True
    assert whisper == "small"

    # Option 3: Compute type change
    mock_select.return_value.ask.return_value = "3"
    mock_text.return_value.ask.return_value = "int8"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is True
    assert compute == "int8"

    # Option 4: Device change
    mock_select.return_value.ask.return_value = "4"
    mock_text.return_value.ask.return_value = "cpu"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is True
    assert device == "cpu"

    # Option 5: Retry as-is
    mock_select.return_value.ask.return_value = "5"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is True

    # Option 6: Exit
    mock_select.return_value.ask.return_value = "6"
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is False

    # Cancelled / None
    mock_select.return_value.ask.return_value = None
    llm, whisper, compute, device, retry = prompt_recovery_menu(
        error=RuntimeError("fail"),
        current_llm="gemma4:e2b",
        current_whisper="large-v3-turbo",
        current_compute="float16",
        current_device="cuda",
        is_interactive=True,
        console=test_console,
    )
    assert retry is False


def test_create_transcription_progress() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)
    with create_transcription_progress(console=test_console) as progress:
        task = progress.add_task("test", total=10.0, model="tiny")
        progress.update(task, completed=5.0)
    output = buf.getvalue()
    assert "Transcribing with Whisper" in output or "tiny" in output


def test_print_step_complete() -> None:
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True, width=100)
    print_step_complete("Transcription completed", elapsed=12.34, console=test_console)
    output = buf.getvalue()
    assert "Transcription completed" in output
    assert "00:00:12:34" in output
