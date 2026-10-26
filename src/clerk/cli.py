from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import typer

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
    resolve_prompt_option,
)
from .pipeline import run_pipeline

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

_resolve_prompt_option = resolve_prompt_option

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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    if verbose:
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


def format_time_hhmmssmm(seconds: float) -> str:
    total_cs = round(seconds * 100)
    hours, rem = divmod(total_cs, 360_000)
    mins, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{cs:02d}"


def print_pipeline_status(
    transcript_path: Path | None,
    summary_path: Path | None,
    output_dir: Path,
    metadata: dict,
    verbose: bool,
) -> None:
    timings = metadata.get("timings", {})
    models = metadata.get("models", {})
    word_counts = metadata.get("word_counts", {})
    output_files = metadata.get("output_files", {})
    metadata_file = output_files.get("metadata_path", str(output_dir / "transcript_metadata.json"))

    transcript_display = str(transcript_path) if transcript_path else "N/A (skipped)"
    summary_display = str(summary_path) if summary_path else "N/A (skipped)"

    typer.echo("\n==================================================")
    typer.echo("                 Pipeline Status                  ")
    typer.echo("==================================================")
    typer.echo("📁 Output Paths:")
    typer.echo(f"  • Transcript (SRT) : {transcript_display}")
    typer.echo(f"  • Summary          : {summary_display}")
    typer.echo(f"  • Metadata JSON    : {metadata_file}")

    typer.echo("\n🤖 Models Used:")
    whisper_model = models.get("whisper_model")
    if whisper_model == "skipped":
        typer.echo("  • Whisper          : Skipped")
    else:
        typer.echo(
            f"  • Whisper          : {whisper_model} "
            f"(device: {models.get('whisper_device')}, compute: {models.get('whisper_compute_type')}, batch_size: {models.get('whisper_batch_size')})"
        )
    llm_model = models.get("llm_model")
    if llm_model == "skipped":
        typer.echo("  • LLM              : Skipped")
    else:
        typer.echo(f"  • LLM              : {llm_model}")

    typer.echo("\n⏱️ Execution Time:")
    typer.echo(f"  • Total Time       : {format_time_hhmmssmm(float(timings.get('total_seconds', 0.0)))}")
    if verbose:
        typer.echo(f"    - Audio Extract  : {format_time_hhmmssmm(float(timings.get('audio_extraction_seconds', 0.0)))}")
        typer.echo(f"    - Transcription  : {format_time_hhmmssmm(float(timings.get('transcription_seconds', 0.0)))}")
        typer.echo(f"    - Summarization  : {format_time_hhmmssmm(float(timings.get('summarization_seconds', 0.0)))}")

    typer.echo("\n📊 Audio & Content Metrics:")
    lang_prob = metadata.get("language_probability")
    prob_str = f" ({lang_prob:.0%})" if lang_prob is not None else ""
    typer.echo(f"  • Language         : {metadata.get('language')}{prob_str}")
    duration = float(metadata.get("duration", 0.0))
    speech_duration = float(metadata.get("duration_after_vad", 0.0))
    typer.echo(
        f"  • Audio Duration   : {format_time_hhmmssmm(duration)} "
        f"(Speech: {format_time_hhmmssmm(speech_duration)})"
    )
    typer.echo(
        f"  • Word Counts      : Transcript ({word_counts.get('transcript_words', 0)} words) -> Summary ({word_counts.get('summary_words', 0)} words)"
    )
    typer.echo("==================================================\n")


def _print_verbose_config(cfg: ResolvedConfig) -> None:
    typer.echo("\n--- Configuration ---")
    typer.echo(f"  Target: {cfg.target}")
    typer.echo(f"  Output Dir: {cfg.output_dir}")
    typer.echo(f"  Preset: {cfg.selected_preset}")
    typer.echo(f"  Whisper Model: {cfg.whisper_model}")
    typer.echo(f"  Whisper Device: {cfg.whisper_device}")
    typer.echo(f"  Whisper Compute Type: {cfg.whisper_compute_type}")
    typer.echo(f"  Whisper Batch Size: {cfg.whisper_batch_size}")
    typer.echo(f"  LLM Model: {cfg.llm_model}")
    typer.echo(f"  Language: {cfg.language}")
    typer.echo(f"  Mode: {'Video' if cfg.is_video else 'Meeting'}")
    typer.echo(f"  Resume: {cfg.resume}")
    typer.echo(f"  Force: {cfg.force}")
    if cfg.transcribe_only:
        typer.echo("  Step: Transcribe Only")
    elif cfg.summarize_only:
        typer.echo("  Step: Summarize Only")
    typer.echo("---------------------\n")
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


def _prompt_pipeline_recovery(
    error: Exception,
    current_llm: str,
    current_whisper: str,
    current_compute: str,
    current_device: str,
) -> tuple[str, str, str, str, bool]:
    """Displays interactive recovery options on pipeline error. Returns updated configs and retry flag."""
    import sys

    if not sys.stdin.isatty():
        return current_llm, current_whisper, current_compute, current_device, False

    typer.echo(f"\n⚠️  Pipeline error: {error}", err=True)
    typer.echo("Model or pipeline failure detected. Choose recovery option:", err=True)
    typer.echo(f"  [1] Enter a new LLM model name (current: {current_llm})")
    typer.echo(f"  [2] Enter a new Whisper model name (current: {current_whisper})")
    typer.echo(f"  [3] Change Whisper compute type (current: {current_compute})")
    typer.echo(f"  [4] Change Whisper device (current: {current_device})")
    typer.echo("  [5] Retry pipeline with current configuration")
    typer.echo("  [6] Exit")

    choice = typer.prompt("Select option [1-6]", default="6")
    if choice == "1":
        new_llm = typer.prompt("Enter new LLM model name").strip()
        if new_llm:
            return new_llm, current_whisper, current_compute, current_device, True
    elif choice == "2":
        new_whisper = typer.prompt("Enter new Whisper model name").strip()
        if new_whisper:
            return current_llm, new_whisper, current_compute, current_device, True
    elif choice == "3":
        new_compute = typer.prompt("Enter new Whisper compute type (e.g. int8, float32, default)").strip()
        if new_compute:
            return current_llm, current_whisper, new_compute, current_device, True
    elif choice == "4":
        new_dev = typer.prompt("Enter new Whisper device (e.g. cpu, cuda)").strip()
        if new_dev:
            return current_llm, current_whisper, current_compute, new_dev, True
    elif choice == "5":
        return current_llm, current_whisper, current_compute, current_device, True

    return current_llm, current_whisper, current_compute, current_device, False


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

    if cfg.verbose:
        _print_verbose_config(cfg)

    temp_file: Path | None = None
    effective_resume = cfg.resume
    effective_summarize_only = cfg.summarize_only
    effective_llm = cfg.llm_model
    effective_whisper = cfg.whisper_model
    effective_compute = cfg.whisper_compute_type
    effective_device = cfg.whisper_device

    try:
        if cfg.is_url:
            t0 = time.perf_counter()
            sub_file = None if (cfg.force or cfg.transcribe_only) else download_youtube_subtitles(cfg.target, language=cfg.language)
            if sub_file and sub_file.exists():
                logger.info("YouTube subtitles fetched in %.2fs (skipping Whisper)", time.perf_counter() - t0)
                temp_file = sub_file
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                persisted_srt = cfg.output_dir / sub_file.name
                shutil.copy2(sub_file, persisted_srt)
                input_path = persisted_srt
                effective_summarize_only = True
            else:
                temp_file = download_youtube_audio(cfg.target)
                logger.info("YouTube audio download completed in %.2fs", time.perf_counter() - t0)
                input_path = temp_file
        else:
            input_path = Path(cfg.target)

        while True:
            try:
                transcript_path, summary_path, metadata = run_pipeline(
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
                break
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

        print_pipeline_status(
            transcript_path=transcript_path,
            summary_path=summary_path,
            output_dir=cfg.output_dir,
            metadata=metadata,
            verbose=cfg.verbose,
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
