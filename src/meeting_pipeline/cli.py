from __future__ import annotations

import logging
from pathlib import Path

import typer

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
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("output"), "--output-dir"),
    whisper_model: str = typer.Option("small", "--whisper-model"),
    whisper_device: str = typer.Option("cpu", "--whisper-device"),
    whisper_compute_type: str = typer.Option("int8", "--whisper-compute-type"),
    llm_model: str = typer.Option("gemma:2b", "--llm-model"),
    language: str = typer.Option("pt", "--language"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    configure_logging(verbose)

    try:
        transcript_path, summary_path = run_pipeline(
            input_path=input_file,
            output_dir=output_dir,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            llm_model=llm_model,
            language=language,
        )
    except Exception:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=1)

    typer.echo(f"Transcript: {transcript_path}")
    typer.echo(f"Meeting points: {summary_path}")
