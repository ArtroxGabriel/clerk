from __future__ import annotations

import logging
from pathlib import Path

import typer

import time

from .audio import download_youtube_audio
from .pipeline import run_pipeline

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)



def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


@app.command()
def main(
    target: str = typer.Option(
        ...,
        "--target",
        help="Input file path or YouTube URL of the video/audio to process.",
    ),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    whisper_model: str = typer.Option("small", "--whisper-model"),
    whisper_device: str = typer.Option("cpu", "--whisper-device"),
    whisper_compute_type: str = typer.Option("int8", "--whisper-compute-type"),
    llm_model: str = typer.Option("LiquidAI/lfm2.5-1.2b-instruct", "--llm-model"),
    language: str = typer.Option("pt", "--language"),
    video: bool = typer.Option(
        False,
        "--video",
        help="Use video summary prompt template instead of meeting template.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)

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
                raise typer.Exit(code=1)

        transcript_path, summary_path = run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            llm_model=llm_model,
            language=language,
            is_video=is_video,
        )

        typer.echo(f"Transcript: {transcript_path}")
        typer.echo(f"Meeting points: {summary_path}")

    except KeyboardInterrupt:
        typer.echo("\nProcess interrupted by user. Exiting...", err=True)
        raise typer.Exit(code=130)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=1)
    finally:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                logger.info("Deleted temporary YouTube audio file: %s", temp_file)
            except Exception as e:
                logger.warning("Failed to delete temporary file %s: %s", temp_file, e)

