from __future__ import annotations

import logging
from pathlib import Path
import time

import typer

from .audio import download_youtube_audio
from .pipeline import run_pipeline

import os

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, str | int]] = {
    "cpu": {
        "whisper_model": "large-v3",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "whisper_batch_size": 2,
        "llm_model": "LiquidAI/lfm2.5-1.2b-instruct",
    },
    "fast": {
        "whisper_model": "small",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "whisper_batch_size": 2,
        "llm_model": "LiquidAI/lfm2.5-1.2b-instruct",
    },
    "gpu": {
        "whisper_model": "large-v3",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 8,
        "llm_model": "llama3.1:8b",
    },
    "cuda": {
        "whisper_model": "large-v3",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 8,
        "llm_model": "llama3.1:8b",
    },
    "accurate": {
        "whisper_model": "large-v3",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 4,
        "llm_model": "llama3.1:8b",
    },
}

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

GPU_PRESETS = {"gpu", "cuda", "accurate"}
VALID_COMPUTE_TYPES = {"int8", "int8_float16", "int8_bfloat16", "int8_float32", "float16", "bfloat16", "float32", "default"}
VALID_DEVICES = {"cpu", "cuda", "gpu", "auto"}


def is_gpu_available() -> bool:
    env_gpu = os.environ.get("ENABLE_GPU", "").strip().lower()
    if env_gpu in ("false", "0", "no", "off", "disable", "disabled"):
        return False
    if env_gpu in ("true", "1", "yes", "on", "enable", "enabled"):
        return True

    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


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
    hours, remainder = divmod(total_cs, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{cs:02d}"


def print_pipeline_status(
    transcript_path: Path,
    summary_path: Path,
    output_dir: Path,
    metadata: dict,
    verbose: bool,
) -> None:
    timings = metadata.get("timings", {})
    models = metadata.get("models", {})
    word_counts = metadata.get("word_counts", {})
    output_files = metadata.get("output_files", {})
    metadata_file = output_files.get("metadata_path", str(output_dir / "transcript_metadata.json"))

    typer.echo("\n==================================================")
    typer.echo("                 Pipeline Status                  ")
    typer.echo("==================================================")
    typer.echo("📁 Output Paths:")
    typer.echo(f"  • Transcript (SRT) : {transcript_path}")
    typer.echo(f"  • Summary          : {summary_path}")
    typer.echo(f"  • Metadata JSON    : {metadata_file}")

    typer.echo("\n🤖 Models Used:")
    typer.echo(
        f"  • Whisper          : {models.get('whisper_model')} "
        f"(device: {models.get('whisper_device')}, compute: {models.get('whisper_compute_type')}, batch_size: {models.get('whisper_batch_size')})"
    )
    typer.echo(f"  • LLM              : {models.get('llm_model')}")

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


@app.command()
def main(
    target: str = typer.Option(
        ...,
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
    whisper_model: str | None = typer.Option(None, "--whisper-model", help="Whisper model name ('tiny'., 'small', 'medium', 'large-v3')."),
    whisper_device: str | None = typer.Option(None, "--whisper-device", help="Whisper model device (e.g., 'cpu', 'cuda')."),
    whisper_compute_type: str | None = typer.Option(None, "--whisper-compute-type", help="Whisper model compute type (e.g., 'int8', 'int8_float16', 'float16', 'float32')."),
    whisper_batch_size: int | None = typer.Option(
        None,
        "--whisper-batch-size",
        help="Batch size for faster-whisper transcription. Higher values increase transcription speed but consume significantly more RAM/VRAM memory (Default: 2).",
    ),
    llm_model: str | None = typer.Option(None, "--llm-model"),
    language: str = typer.Option("pt", "--language"),
    video: bool = typer.Option(
        False,
        "--video",
        help="Enforce video summary prompt template (saves summary to resume.md).",
    ),
    meeting: bool = typer.Option(
        False,
        "--meeting",
        help="Enforce meeting summary prompt template (saves summary to meeting_points.md).",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)

    if not target or not target.strip():
        typer.echo("Error: --target cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    target = target.strip()

    if video and meeting:
        typer.echo("Error: Cannot specify both --video and --meeting options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if gpu and fast:
        typer.echo("Error: Cannot specify both --gpu and --fast options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    gpu_supported = is_gpu_available()
    allowed_presets = (
        sorted(list(PRESETS.keys()))
        if gpu_supported
        else sorted([k for k in PRESETS.keys() if k not in GPU_PRESETS])
    )

    if gpu and not gpu_supported:
        typer.echo(
            "Error: GPU execution is disabled or unavailable in CPU mode. "
            f"Available presets: {', '.join(allowed_presets)}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    # Determine preset profile
    selected_preset = preset.lower() if preset else None
    if not selected_preset:
        if gpu:
            selected_preset = "gpu"
        elif fast:
            selected_preset = "fast"
        else:
            selected_preset = "cpu"

    if selected_preset not in allowed_presets:
        available = ", ".join(allowed_presets)
        if selected_preset in GPU_PRESETS and not gpu_supported:
            typer.echo(
                f"Error: Preset '{selected_preset}' requires GPU execution, which is disabled or unavailable in CPU mode. "
                f"Available presets: {available}",
                err=True,
            )
        else:
            typer.echo(f"Error: Unknown preset '{selected_preset}'. Available presets: {available}", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    defaults = PRESETS[selected_preset]

    # Explicit flags override preset defaults
    effective_whisper_model = (whisper_model or str(defaults["whisper_model"])).strip()
    effective_whisper_device = (whisper_device or str(defaults["whisper_device"])).strip()
    effective_whisper_compute_type = (whisper_compute_type or str(defaults["whisper_compute_type"])).strip()
    effective_whisper_batch_size = whisper_batch_size if whisper_batch_size is not None else int(defaults.get("whisper_batch_size", 2))
    effective_llm_model = (llm_model or str(defaults["llm_model"])).strip()

    if not effective_whisper_model:
        typer.echo("Error: --whisper-model cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if not effective_llm_model:
        typer.echo("Error: --llm-model cannot be empty.", err=True)
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_batch_size < 1:
        typer.echo(
            f"Error: --whisper-batch-size must be a positive integer (got {effective_whisper_batch_size}).",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_device.lower() not in VALID_DEVICES:
        allowed_devs = ", ".join(sorted(list(VALID_DEVICES)))
        typer.echo(
            f"Error: Invalid --whisper-device '{effective_whisper_device}'. Allowed values: {allowed_devs}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if (effective_whisper_device.lower() in ("cuda", "gpu")) and not gpu_supported:
        typer.echo(
            "Error: '--whisper-device cuda' was specified, but GPU execution is disabled or unavailable in CPU mode.",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    if effective_whisper_compute_type.lower() not in VALID_COMPUTE_TYPES:
        allowed_types = ", ".join(sorted(list(VALID_COMPUTE_TYPES)))
        typer.echo(
            f"Error: Invalid --whisper-compute-type '{effective_whisper_compute_type}'. Allowed values: {allowed_types}",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    is_url = (
        target.startswith(("http://", "https://", "www."))
        or "youtube.com" in target
        or "youtu.be" in target
    )

    if not is_url:
        local_path = Path(target)
        if not local_path.exists():
            logger.error("Input file does not exist: %s", local_path)
            typer.echo(f"Error: Input file does not exist: {local_path}", err=True)
            raise typer.Exit(code=EXIT_ERROR)
        if not local_path.is_file():
            logger.error("Input path is not a file: %s", local_path)
            typer.echo(f"Error: Input path is not a file: {local_path}", err=True)
            raise typer.Exit(code=EXIT_ERROR)

    if video:
        is_video = True
    elif meeting:
        is_video = False
    else:
        is_video = is_url

    temp_file: Path | None = None

    try:
        if is_url:
            t0 = time.perf_counter()
            temp_file = download_youtube_audio(target)
            t_dl = time.perf_counter() - t0
            logger.info("YouTube audio download completed in %.2fs", t_dl)
            input_path = temp_file
        else:
            input_path = Path(target)

        transcript_path, summary_path, metadata = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model=effective_whisper_model,
            whisper_device=effective_whisper_device,
            whisper_compute_type=effective_whisper_compute_type,
            llm_model=effective_llm_model,
            language=language,
            whisper_batch_size=effective_whisper_batch_size,
            is_video=is_video,
            verbose=verbose,
        )

        print_pipeline_status(
            transcript_path=transcript_path,
            summary_path=summary_path,
            output_dir=output_dir,
            metadata=metadata,
            verbose=verbose,
        )

    except KeyboardInterrupt:
        typer.echo("\nProcess interrupted by user. Exiting...", err=True)
        raise typer.Exit(code=EXIT_INTERRUPTED)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=EXIT_ERROR)
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.debug("Deleted temporary YouTube audio file: %s", temp_file)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", temp_file, e)
