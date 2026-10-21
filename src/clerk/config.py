from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer

logger = logging.getLogger(__name__)

PRESETS: dict[str, dict[str, str | int]] = {
    "cpu": {
        "whisper_model": "large-v3-turbo",
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
        "whisper_model": "large-v3-turbo",
        "whisper_device": "cuda",
        "whisper_compute_type": "float16",
        "whisper_batch_size": 4,
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
PRESETS["cuda"] = PRESETS["gpu"]

EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

GPU_PRESETS = {"gpu", "cuda", "accurate"}
VALID_COMPUTE_TYPES = {
    "int8",
    "int8_float16",
    "int8_bfloat16",
    "int8_float32",
    "float16",
    "bfloat16",
    "float32",
    "default",
}
VALID_DEVICES = {"cpu", "cuda", "gpu", "auto"}


@dataclass(frozen=True)
class ResolvedConfig:
    target: str
    output_dir: Path
    selected_preset: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_batch_size: int
    llm_model: str
    language: str
    is_url: bool
    is_video: bool
    transcribe_only: bool
    summarize_only: bool
    custom_prompt: str | None
    custom_consolidation_prompt: str | None
    resume: bool
    force: bool
    verbose: bool


def is_gpu_available() -> bool:
    """Checks whether GPU execution is enabled and available via environment or CTranslate2."""
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


def resolve_prompt_option(text: str | None, path: Path | None, opt_name: str) -> str | None:
    """Validates and resolves inline prompt text or prompt file path."""
    if text and path:
        typer.echo(f"Error: Cannot specify both --{opt_name} and --{opt_name}-file options simultaneously.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if not path:
        return text
    if not path.exists():
        typer.echo(f"Error: Prompt file does not exist: {path}", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if not path.is_file():
        typer.echo(f"Error: Prompt file path is not a file: {path}", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    return path.read_text(encoding="utf-8")


def _abort(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=EXIT_ERROR)


def _validate_target_path(target: str, is_url: bool, transcribe_only: bool, summarize_only: bool) -> bool:
    """Validates local target path and auto-enables summarize_only for .srt inputs."""
    if is_url:
        return summarize_only

    local_path = Path(target)
    if not local_path.exists():
        logger.error("Input file does not exist: %s", local_path)
        _abort(f"Error: Input file does not exist: {local_path}")
    if not local_path.is_file():
        logger.error("Input path is not a file: %s", local_path)
        _abort(f"Error: Input path is not a file: {local_path}")

    if local_path.suffix.lower() == ".srt" and not transcribe_only:
        return True
    return summarize_only


def resolve_cli_config(
    *,
    target: str | None,
    output_dir: Path,
    preset: str | None,
    gpu: bool,
    fast: bool,
    whisper_model: str | None,
    whisper_device: str | None,
    whisper_compute_type: str | None,
    whisper_batch_size: int | None,
    llm_model: str | None,
    language: str,
    video: bool,
    meeting: bool,
    transcribe_only: bool,
    summarize_only: bool,
    prompt: str | None,
    prompt_file: Path | None,
    consolidation_prompt: str | None,
    consolidation_prompt_file: Path | None,
    resume: bool,
    force: bool,
    verbose: bool,
    gpu_supported: bool,
) -> ResolvedConfig:
    """Validates CLI flags and resolves preset defaults into an immutable ResolvedConfig."""
    if not target or not target.strip():
        _abort("Error: --target cannot be empty.")
    clean_target = target.strip()

    if resume and force:
        _abort("Error: Cannot specify both --resume and --force simultaneously.")
    if video and meeting:
        _abort("Error: Cannot specify both --video and --meeting options simultaneously.")
    if gpu and fast:
        _abort("Error: Cannot specify both --gpu and --fast options simultaneously.")
    if transcribe_only and summarize_only:
        _abort("Error: Cannot specify both --transcribe-only and --summarize-only options simultaneously.")

    effective_custom_prompt = resolve_prompt_option(prompt, prompt_file, "prompt")
    effective_custom_consolidation_prompt = resolve_prompt_option(
        consolidation_prompt, consolidation_prompt_file, "consolidation-prompt"
    )

    allowed_presets = (
        sorted(PRESETS.keys()) if gpu_supported else sorted([k for k in PRESETS if k not in GPU_PRESETS])
    )

    if gpu and not gpu_supported:
        _abort(
            "Error: GPU execution is disabled or unavailable in CPU mode. "
            f"Available presets: {', '.join(allowed_presets)}"
        )

    selected_preset = preset.lower() if preset else ("gpu" if gpu else ("fast" if fast else "cpu"))
    if selected_preset not in allowed_presets:
        available = ", ".join(allowed_presets)
        if selected_preset in GPU_PRESETS and not gpu_supported:
            _abort(
                f"Error: Preset '{selected_preset}' requires GPU execution, which is disabled or unavailable in CPU mode. "
                f"Available presets: {available}"
            )
        _abort(f"Error: Unknown preset '{selected_preset}'. Available presets: {available}")

    defaults = PRESETS[selected_preset]
    eff_whisper_model = (whisper_model or str(defaults["whisper_model"])).strip()
    eff_whisper_device = (whisper_device or str(defaults["whisper_device"])).strip()
    eff_whisper_compute = (whisper_compute_type or str(defaults["whisper_compute_type"])).strip()
    eff_batch_size = whisper_batch_size if whisper_batch_size is not None else int(defaults.get("whisper_batch_size", 2))
    eff_llm_model = (llm_model or str(defaults["llm_model"])).strip()

    if not eff_whisper_model:
        _abort("Error: --whisper-model cannot be empty.")
    if not eff_llm_model:
        _abort("Error: --llm-model cannot be empty.")
    if eff_batch_size < 1:
        _abort(f"Error: --whisper-batch-size must be a positive integer (got {eff_batch_size}).")
    if eff_whisper_device.lower() not in VALID_DEVICES:
        _abort(f"Error: Invalid --whisper-device '{eff_whisper_device}'. Allowed values: {', '.join(sorted(VALID_DEVICES))}")
    if eff_whisper_device.lower() in ("cuda", "gpu") and not gpu_supported:
        _abort("Error: '--whisper-device cuda' was specified, but GPU execution is disabled or unavailable in CPU mode.")

    env_gpu = os.environ.get("ENABLE_GPU", "").strip().lower()
    if env_gpu in ("true", "1", "yes", "on", "enable", "enabled") and (
        selected_preset in ("cpu", "fast") or eff_whisper_device.lower() == "cpu"
    ):
        typer.echo(
            "Warning: Running in CPU mode inside a GPU-enabled container. "
            "For CPU workloads, consider using the lightweight clerk:cpu image (saves ~4GB).",
            err=True,
        )

    if eff_whisper_compute.lower() not in VALID_COMPUTE_TYPES:
        _abort(
            f"Error: Invalid --whisper-compute-type '{eff_whisper_compute}'. "
            f"Allowed values: {', '.join(sorted(VALID_COMPUTE_TYPES))}"
        )
    if eff_whisper_device.lower() in ("cpu", "auto") and eff_whisper_compute.lower() in (
        "float16",
        "int8_float16",
        "bfloat16",
        "int8_bfloat16",
    ):
        _abort(
            f"Error: '--whisper-compute-type {eff_whisper_compute}' requires GPU (cuda). "
            "On CPU, available compute types are: default, float32, int8, int8_float32."
        )

    is_url = clean_target.startswith(("http://", "https://", "www.")) or "youtube.com" in clean_target or "youtu.be" in clean_target
    eff_summarize_only = _validate_target_path(clean_target, is_url, transcribe_only, summarize_only)
    is_video = True if video else (False if meeting else is_url)

    return ResolvedConfig(
        target=clean_target,
        output_dir=output_dir,
        selected_preset=selected_preset,
        whisper_model=eff_whisper_model,
        whisper_device=eff_whisper_device,
        whisper_compute_type=eff_whisper_compute,
        whisper_batch_size=eff_batch_size,
        llm_model=eff_llm_model,
        language=language,
        is_url=is_url,
        is_video=is_video,
        transcribe_only=transcribe_only,
        summarize_only=eff_summarize_only,
        custom_prompt=effective_custom_prompt,
        custom_consolidation_prompt=effective_custom_consolidation_prompt,
        resume=resume,
        force=force,
        verbose=verbose,
    )
