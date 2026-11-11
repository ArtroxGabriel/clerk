from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import typer
from rich.logging import RichHandler

from .audio import download_youtube_audio, download_youtube_subtitles
from .config import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    GPU_PRESETS,
    PRESETS,
    VALID_COMPUTE_TYPES,
    VALID_DEVICES,
    ResolvedConfig,
    is_gpu_available,
    resolve_cli_config,
)
from .pipeline import run_pipeline
from .ui import (
    format_time_hhmmssmm,
    get_console,
    prompt_recovery_menu,
    render_config_panel,
    render_markdown_preview,
    render_startup_banner,
    render_status_summary,
)

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

__all__ = [
    "EXIT_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_SUCCESS",
    "GPU_PRESETS",
    "PRESETS",
    "VALID_COMPUTE_TYPES",
    "VALID_DEVICES",
    "app",
    "format_time_hhmmssmm",
    "is_gpu_available",
    "main",
    "print_pipeline_status",
]


def configure_logging(verbose: bool) -> None:
    """Configure unified logging with RichHandler to match TUI presentation."""
    console = get_console()
    if verbose:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], format="%(message)s", force=True)
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    else:
        handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
        )
        logging.basicConfig(level=logging.WARNING, handlers=[handler], format="%(message)s", force=True)
        for logger_name in ("clerk", "httpx", "httpcore", "faster_whisper", "urllib3"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def print_pipeline_status(
    transcript_path: Path | None,
    summary_path: Path | None,
    output_dir: Path,
    metadata: dict,
    verbose: bool,
) -> None:
    render_status_summary(
        transcript_path=transcript_path,
        summary_path=summary_path,
        output_dir=output_dir,
        metadata=metadata,
        verbose=verbose,
    )


def _print_verbose_config(cfg: ResolvedConfig) -> None:
    render_config_panel(cfg)
    logger.debug(
        "Resolved config: target=%s, output_dir=%s, preset=%s, whisper_model=%s, whisper_device=%s, "
        "whisper_compute_type=%s, whisper_batch_size=%d, llm_model=%s, language=%s, is_video=%s, "
        "resume=%s, force=%s, transcribe_only=%s, summarize_only=%s",
        cfg.target,
        cfg.output_dir,
        cfg.selected_preset,
        cfg.whisper_model,
        cfg.whisper_device,
        cfg.whisper_compute_type,
        cfg.whisper_batch_size,
        cfg.llm_model,
        cfg.language,
        cfg.is_video,
        cfg.resume,
        cfg.force,
        cfg.transcribe_only,
        cfg.summarize_only,
    )


def _prepare_input_target(cfg: ResolvedConfig) -> tuple[Path, Path | None, bool]:
    """Resolve input media target or download YouTube media."""
    if not cfg.is_url:
        return Path(cfg.target), None, cfg.summarize_only

    t0 = time.perf_counter()
    sub_file = None if (cfg.force or cfg.transcribe_only) else download_youtube_subtitles(cfg.target, language=cfg.language)
    if sub_file and sub_file.exists():
        logger.info("YouTube subtitles fetched in %.2fs (skipping Whisper)", time.perf_counter() - t0)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        persisted_srt = cfg.output_dir / sub_file.name
        shutil.copy2(sub_file, persisted_srt)
        return persisted_srt, sub_file, True

    temp_file = download_youtube_audio(cfg.target)
    logger.info("YouTube audio download completed in %.2fs", time.perf_counter() - t0)
    return temp_file, temp_file, cfg.summarize_only


def _prompt_pipeline_recovery(
    error: Exception,
    current_llm: str,
    current_whisper: str,
    current_compute: str,
    current_device: str,
) -> tuple[str, str, str, str, bool]:
    """Displays interactive recovery options on pipeline error. Returns updated configs and retry flag."""
    return prompt_recovery_menu(
        error=error,
        current_llm=current_llm,
        current_whisper=current_whisper,
        current_compute=current_compute,
        current_device=current_device,
    )


def _run_pipeline_with_recovery(
    input_path: Path,
    cfg: ResolvedConfig,
    effective_summarize_only: bool,
) -> tuple[Path | None, Path | None, dict]:
    """Execute pipeline with interactive recovery retry loop."""
    effective_llm = cfg.llm_model
    effective_whisper = cfg.whisper_model
    effective_compute = cfg.whisper_compute_type
    effective_device = cfg.whisper_device
    effective_resume = cfg.resume

    while True:
        try:
            return run_pipeline(
                input_path=input_path,
                output_dir=cfg.output_dir,
                whisper_model=effective_whisper,
                whisper_device=effective_device,
                whisper_compute_type=effective_compute,
                llm_model=effective_llm,
                language=cfg.language,
                whisper_batch_size=cfg.whisper_batch_size,
                is_video=cfg.is_video,
                verbose=cfg.verbose,
                transcribe_only=cfg.transcribe_only,
                summarize_only=effective_summarize_only,
                custom_prompt=cfg.custom_prompt,
                custom_consolidation_prompt=cfg.custom_consolidation_prompt,
                resume=effective_resume,
            )
        except (KeyboardInterrupt, typer.Exit):
            raise
        except Exception as e:
            (
                effective_llm,
                effective_whisper,
                effective_compute,
                effective_device,
                retry,
            ) = _prompt_pipeline_recovery(
                error=e,
                current_llm=effective_llm,
                current_whisper=effective_whisper,
                current_compute=effective_compute,
                current_device=effective_device,
            )
            if retry:
                effective_resume = True
                continue
            logger.exception("Pipeline execution failed")
            raise typer.Exit(code=EXIT_ERROR) from e


@app.command()
def main(
    positional_target: str | None = typer.Argument(
        None,
        help="Input file path or YouTube URL of the video/audio to process.",
        show_default=False,
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help="Input file path or YouTube URL of the video/audio to process.",
    ),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Configuration profile ('cpu', 'fast', or GPU profiles 'gpu'/'cuda'/'accurate' when GPU mode is enabled).",
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Shortcut for GPU configuration (--preset gpu)."),
    fast: bool = typer.Option(False, "--fast", help="Shortcut for fast CPU configuration (--preset fast)."),
    whisper_model: str | None = typer.Option(
        None, "--whisper-model", help="Whisper model name ('tiny', 'small', 'medium', 'large-v3', 'large-v3-turbo')."
    ),
    whisper_device: str | None = typer.Option(None, "--whisper-device", help="Whisper model device (e.g., 'cpu', 'cuda')."),
    whisper_compute_type: str | None = typer.Option(
        None, "--whisper-compute-type", help="Whisper model compute type (e.g., 'int8', 'int8_float16', 'float16', 'float32')."
    ),
    whisper_batch_size: int | None = typer.Option(
        None,
        "--whisper-batch-size",
        help="Batch size for faster-whisper transcription (Default: 2 on CPU, 4 on GPU).",
    ),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    language: str = typer.Option(
        "pt", "--language", help="Summary/transcription language ('pt', 'en', 'es', etc., or 'auto' for automatic speech detection)."
    ),
    video: bool = typer.Option(False, "--video", help="Enforce video summary prompt template (saves summary to resume.md)."),
    meeting: bool = typer.Option(
        False, "--meeting", help="Enforce meeting minutes (Ata) prompt template (saves summary to ata.md)."
    ),
    transcribe_only: bool = typer.Option(
        False, "--transcribe-only", help="Execute only audio extraction and speech transcription steps."
    ),
    summarize_only: bool = typer.Option(
        False, "--summarize-only", help="Execute only transcript summarization step (target can be .srt file)."
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Inline custom summary prompt template (supports {transcript} and {language})."
    ),
    prompt_file: Path | None = typer.Option(None, "--prompt-file", help="Path to custom summary prompt template file."),
    consolidation_prompt: str | None = typer.Option(
        None,
        "--consolidation-prompt",
        help="Inline custom consolidation prompt template (supports {category}, {items}, and {language}).",
    ),
    consolidation_prompt_file: Path | None = typer.Option(
        None, "--consolidation-prompt-file", help="Path to custom consolidation prompt template file."
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Reuse existing intermediate files (.wav / .srt / summary) if available."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-generation of all pipeline steps."),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)
    cfg = resolve_cli_config(
        target=target if target is not None else positional_target,
        output_dir=output_dir,
        preset=preset,
        gpu=gpu,
        fast=fast,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_batch_size=whisper_batch_size,
        llm_model=llm_model,
        language=language,
        video=video,
        meeting=meeting,
        transcribe_only=transcribe_only,
        summarize_only=summarize_only,
        prompt=prompt,
        prompt_file=prompt_file,
        consolidation_prompt=consolidation_prompt,
        consolidation_prompt_file=consolidation_prompt_file,
        resume=resume,
        force=force,
        verbose=verbose,
        gpu_supported=is_gpu_available(),
    )

    render_startup_banner(
        target=cfg.target,
        preset=cfg.selected_preset,
        whisper_model=cfg.whisper_model,
        llm_model=cfg.llm_model,
        device=cfg.whisper_device,
    )

    if cfg.verbose:
        _print_verbose_config(cfg)

    temp_file: Path | None = None
    try:
        input_path, temp_file, effective_summarize_only = _prepare_input_target(cfg)
        transcript_path, summary_path, metadata = _run_pipeline_with_recovery(
            input_path=input_path,
            cfg=cfg,
            effective_summarize_only=effective_summarize_only,
        )
        print_pipeline_status(
            transcript_path=transcript_path,
            summary_path=summary_path,
            output_dir=cfg.output_dir,
            metadata=metadata,
            verbose=cfg.verbose,
        )
        if summary_path and summary_path.exists():
            render_markdown_preview(
                markdown_content=summary_path.read_text(encoding="utf-8"),
                title=summary_path.name,
            )
    except KeyboardInterrupt as err:
        typer.echo("\nProcess interrupted by user. Exiting...", err=True)
        raise typer.Exit(code=EXIT_INTERRUPTED) from err
    except typer.Exit:
        raise
    except Exception as err:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=EXIT_ERROR) from err
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.debug("Deleted temporary YouTube audio file: %s", temp_file)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", temp_file, e)
