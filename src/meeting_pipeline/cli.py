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
        "whisper_model": "medium",
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
        "whisper_model": "medium",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 8,
        "llm_model": "llama3.1:8b",
    },
    "cuda": {
        "whisper_model": "medium",
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

    typer.echo("\n==================================================")
    typer.echo("                 Pipeline Status                  ")
    typer.echo("==================================================")
    typer.echo("📁 Output Paths:")
    typer.echo(f"  • Transcript (SRT) : {transcript_path}")
    typer.echo(f"  • Meeting Points   : {summary_path}")
    typer.echo(f"  • Metadata JSON    : {output_dir / 'transcript_metadata.json'}")

    typer.echo("\n🤖 Models Used:")
    typer.echo(
        f"  • Whisper          : {models.get('whisper_model')} "
        f"(device: {models.get('whisper_device')}, compute: {models.get('whisper_compute_type')}, batch_size: {models.get('whisper_batch_size')})"
    )
    typer.echo(f"  • LLM              : {models.get('llm_model')}")

    typer.echo("\n⏱️ Execution Time:")
    typer.echo(f"  • Total Time       : {timings.get('total_seconds', 0.0):.2f}s")
    if verbose:
        typer.echo(f"    - Audio Extract  : {timings.get('audio_extraction_seconds', 0.0):.2f}s")
        typer.echo(f"    - Transcription  : {timings.get('transcription_seconds', 0.0):.2f}s")
        typer.echo(f"    - Summarization  : {timings.get('summarization_seconds', 0.0):.2f}s")

    typer.echo("\n📊 Audio & Content Metrics:")
    lang_prob = metadata.get("language_probability")
    prob_str = f" ({lang_prob:.0%})" if lang_prob is not None else ""
    typer.echo(f"  • Language         : {metadata.get('language')}{prob_str}")
    typer.echo(
        f"  • Audio Duration   : {metadata.get('duration', 0.0):.1f}s "
        f"(Speech: {metadata.get('duration_after_vad', 0.0):.1f}s)"
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
    whisper_model: str | None = typer.Option(None, "--whisper-model"),
    whisper_device: str | None = typer.Option(None, "--whisper-device"),
    whisper_compute_type: str | None = typer.Option(None, "--whisper-compute-type"),
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
        help="Use video summary prompt template instead of meeting template.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)

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
    effective_whisper_model = whisper_model or str(defaults["whisper_model"])
    effective_whisper_device = whisper_device or str(defaults["whisper_device"])
    effective_whisper_compute_type = whisper_compute_type or str(defaults["whisper_compute_type"])
    effective_whisper_batch_size = whisper_batch_size or int(defaults.get("whisper_batch_size", 2))
    effective_llm_model = llm_model or str(defaults["llm_model"])

    if effective_whisper_device == "cuda" and not gpu_supported:
        typer.echo(
            "Error: '--whisper-device cuda' was specified, but GPU execution is disabled or unavailable in CPU mode.",
            err=True,
        )
        raise typer.Exit(code=EXIT_ERROR)

    is_url = (
        target.startswith(("http://", "https://", "www."))
        or "youtube.com" in target
        or "youtu.be" in target
    )
    is_video = is_url or video

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
            if not input_path.exists():
                logger.error("Input file does not exist: %s", input_path)
                typer.echo(f"Error: Input file does not exist: {input_path}", err=True)
                raise typer.Exit(code=EXIT_ERROR)

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
                logger.info("Deleted temporary YouTube audio file: %s", temp_file)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", temp_file, e)
