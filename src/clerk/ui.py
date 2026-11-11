from __future__ import annotations

import contextlib
import difflib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from .config import ResolvedConfig
    from .srt_search import SrtMatchWindow

_stdout_console: Console | None = None
_stderr_console: Console | None = None


def get_console() -> Console:
    """Return standard output Console singleton.

    Example:
        >>> console = get_console()
        >>> console.is_terminal is not None
        True
    """
    global _stdout_console
    if _stdout_console is None:
        _stdout_console = Console(highlight=False)
    return _stdout_console


def get_err_console() -> Console:
    """Return standard error Console singleton.

    Example:
        >>> err_console = get_err_console()
        >>> err_console.is_terminal is not None
        True
    """
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(stderr=True, highlight=False)
    return _stderr_console


def format_time_hhmmssmm(seconds: float) -> str:
    """Format seconds into hh:mm:ss:cs string format."""
    total_cs = round(seconds * 100)
    hours, rem = divmod(total_cs, 360_000)
    mins, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{cs:02d}"


_format_time_hhmmssmm = format_time_hhmmssmm


def render_startup_banner(
    target: str | None,
    preset: str | None,
    whisper_model: str,
    llm_model: str,
    device: str,
    console: Console | None = None,
) -> None:
    """Display modern startup banner with target, preset badge, and model metadata.

    Example:
        >>> render_startup_banner("file.mp4", "gpu", "large-v3-turbo", "gemma4:e2b", "cuda")
    """
    out = console or get_console()
    is_gpu = device.lower().startswith("cuda") or device.lower() == "gpu"
    device_style = "bold green" if is_gpu else "bold yellow"
    device_label = f"[{device_style}]{device.upper()}[/{device_style}]"

    content = Table.grid(padding=(0, 2))
    content.add_column(style="bold cyan", justify="right")
    content.add_column(style="white")

    content.add_row("Target:", target if target else "[dim]Not specified[/dim]")
    content.add_row("Device:", f"{device_label} (Preset: [bold]{preset or 'default'}[/bold])")
    content.add_row("Whisper:", f"[bold blue]{whisper_model}[/bold blue]")
    content.add_row("LLM:", f"[bold magenta]{llm_model}[/bold magenta]")

    banner = Panel(
        content,
        title="[bold magenta]⚡ clerk[/bold magenta] [dim]v3.0.0[/dim]",
        subtitle="[dim]transcription & summarization pipeline[/dim]",
        border_style="magenta",
        padding=(1, 2),
    )
    out.print(banner)


def render_config_panel(cfg: ResolvedConfig, console: Console | None = None) -> None:
    """Render resolved configuration options inside a styled Rich panel.

    Example:
        >>> render_config_panel(cfg)
    """
    out = console or get_console()
    text = Text()
    text.append(f"  Target: {cfg.target}\n", style="white")
    text.append(f"  Output Dir: {cfg.output_dir}\n", style="white")
    text.append(f"  Preset: {cfg.selected_preset}\n", style="white")
    text.append(f"  Whisper Model: {cfg.whisper_model}\n", style="white")
    text.append(f"  Whisper Device: {cfg.whisper_device}\n", style="white")
    text.append(f"  Whisper Compute Type: {cfg.whisper_compute_type}\n", style="white")
    text.append(f"  Whisper Batch Size: {cfg.whisper_batch_size}\n", style="white")
    text.append(f"  LLM Model: {cfg.llm_model}\n", style="white")
    text.append(f"  Language: {cfg.language}\n", style="white")
    text.append(f"  Mode: {'Video' if cfg.is_video else 'Meeting'}\n", style="white")
    text.append(f"  Resume: {cfg.resume}\n", style="white")
    text.append(f"  Force: {cfg.force}\n", style="white")

    if cfg.transcribe_only:
        text.append("  Step: Transcribe Only\n", style="cyan")
    elif cfg.summarize_only:
        text.append("  Step: Summarize Only\n", style="cyan")

    panel = Panel(text, title="[bold]--- Configuration ---[/bold]", border_style="blue", padding=(0, 1))
    out.print(panel)


def _build_timings_table(timings: dict[str, Any], verbose: bool) -> Table:
    t_table = Table(box=None, show_header=False, pad_edge=False)
    t_table.add_column("Phase", style="bold cyan")
    t_table.add_column("Duration", style="bold green", justify="right")

    t_table.add_row("  • Total Time", _format_time_hhmmssmm(float(timings.get("total_seconds", 0.0))))
    if verbose:
        t_table.add_row(
            "    - Audio Extract", _format_time_hhmmssmm(float(timings.get("audio_extraction_seconds", 0.0)))
        )
        t_table.add_row(
            "    - Transcription", _format_time_hhmmssmm(float(timings.get("transcription_seconds", 0.0)))
        )
        t_table.add_row(
            "    - Summarization", _format_time_hhmmssmm(float(timings.get("summarization_seconds", 0.0)))
        )
    return t_table


def _build_metrics_table(metadata: dict[str, Any], word_counts: dict[str, Any]) -> Table:
    m_table = Table(box=None, show_header=False, pad_edge=False)
    m_table.add_column("Metric", style="bold cyan")
    m_table.add_column("Value", style="white")

    lang_prob = metadata.get("language_probability")
    prob_str = f" ({lang_prob:.0%})" if lang_prob is not None else ""
    m_table.add_row("  • Detected Language", f"{metadata.get('language')}{prob_str}")

    duration = float(metadata.get("duration", 0.0))
    speech = float(metadata.get("duration_after_vad", 0.0))
    m_table.add_row("  • Audio Duration", f"{_format_time_hhmmssmm(duration)} (Speech: {_format_time_hhmmssmm(speech)})")
    t_words = word_counts.get("transcript_words", 0)
    s_words = word_counts.get("summary_words", 0)
    m_table.add_row("  • Word Counts", f"{t_words} words transcript ➔ {s_words} words summary")
    return m_table


def _build_paths_text(transcript_path: Path | None, summary_path: Path | None, metadata_file: str) -> Text:
    t_disp = str(transcript_path) if transcript_path else "N/A (skipped)"
    s_disp = str(summary_path) if summary_path else "N/A (skipped)"
    paths_text = Text()
    paths_text.append("📁 Output Paths:\n", style="bold yellow")
    paths_text.append("  • Transcript (SRT) : ", style="bold cyan")
    paths_text.append(f"{t_disp}\n", style="bold green" if transcript_path else "dim")
    paths_text.append("  • Summary          : ", style="bold cyan")
    paths_text.append(f"{s_disp}\n", style="bold magenta" if summary_path else "dim")
    paths_text.append("  • Metadata JSON    : ", style="bold cyan")
    paths_text.append(f"{metadata_file}\n")
    return paths_text


def _build_models_text(models: dict[str, Any]) -> Text:
    w_model = models.get("whisper_model")
    if w_model == "skipped":
        w_line = "  • Whisper          : Skipped"
    else:
        w_line = (
            f"  • Whisper          : {w_model} "
            f"(device: {models.get('whisper_device')}, compute: {models.get('whisper_compute_type')}, batch_size: {models.get('whisper_batch_size')})"
        )
    l_model = models.get("llm_model")
    l_line = "  • LLM              : Skipped" if l_model == "skipped" else f"  • LLM              : {l_model}"

    models_text = Text()
    models_text.append("🤖 Models Used:\n", style="bold yellow")
    models_text.append(f"{w_line}\n")
    models_text.append(f"{l_line}\n")
    return models_text


def render_status_summary(
    transcript_path: Path | None,
    summary_path: Path | None,
    output_dir: Path,
    metadata: dict[str, Any],
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Render completion summary card with paths, models, timings, and metrics.

    Example:
        >>> render_status_summary(transcript_path, summary_path, output_dir, metadata)
    """
    out = console or get_console()
    timings = metadata.get("timings", {})
    models = metadata.get("models", {})
    word_counts = metadata.get("word_counts", {})
    output_files = metadata.get("output_files", {})
    metadata_file = output_files.get("metadata_path", str(output_dir / "transcript_metadata.json"))

    group = Group(
        _build_paths_text(transcript_path, summary_path, metadata_file),
        _build_models_text(models),
        Text("\n"),
        Text("⏱️  Execution Times:\n", style="bold yellow"),
        _build_timings_table(timings, verbose),
        Text("\n"),
        Text("📊 Audio & Content Metrics:\n", style="bold yellow"),
        _build_metrics_table(metadata, word_counts),
    )
    panel = Panel(group, title="[bold green]✔ Pipeline Status[/bold green]", border_style="green", padding=(1, 2))
    out.print(panel)


def create_transcription_progress(console: Console | None = None) -> Progress:
    """Create a styled Rich Progress bar for speech-to-text transcription."""
    out = console or get_console()
    return Progress(
        SpinnerColumn(spinner_name="dots", style="bold cyan"),
        TextColumn("[bold cyan]Transcribing with Whisper[/bold cyan] [dim]({task.fields[model]})[/dim]"),
        BarColumn(bar_width=30, style="blue", complete_style="bold green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=out,
        transient=False,
    )


def print_step_complete(message: str, elapsed: float | None = None, console: Console | None = None) -> None:
    """Print step completion confirmation with green checkmark."""
    out = console or get_console()
    timing_str = f" [dim]({_format_time_hhmmssmm(elapsed)})[/dim]" if elapsed is not None else ""
    out.print(f"[bold green]✔[/bold green] {message}{timing_str}")


def render_markdown_preview(markdown_content: str, title: str = "Generated Summary", console: Console | None = None) -> None:
    """Render preview of generated markdown text inside a stylized panel.

    Example:
        >>> render_markdown_preview("# Contexto\\nDecidido...", "Ata de Reunião")
    """
    out = console or get_console()
    md = Markdown(markdown_content)
    panel = Panel(md, title=f"[bold cyan]📄 {title}[/bold cyan]", border_style="cyan", padding=(1, 2))
    out.print(panel)


@contextlib.contextmanager
def step_spinner(status_text: str, console: Console | None = None) -> Generator[Callable[[str], None]]:
    """Context manager showing a live rich spinner during pipeline step execution.

    Example:
        >>> with step_spinner("Extracting audio...") as update_status:
        ...     update_status("Step 1/2...")
    """
    out = console or get_console()
    with out.status(f"[bold cyan]{status_text}[/bold cyan]", spinner="dots") as status:
        yield lambda msg: status.update(f"[bold cyan]{msg}[/bold cyan]")


RECOVERY_DISPATCH: dict[str, tuple[str, int]] = {
    "1": ("Enter new LLM model name:", 0),
    "2": ("Enter new Whisper model name:", 1),
    "3": ("Enter new Whisper compute type (e.g. int8, float32, default):", 2),
    "4": ("Enter new Whisper device (e.g. cpu, cuda):", 3),
}


def prompt_recovery_menu(
    error: Exception,
    current_llm: str,
    current_whisper: str,
    current_compute: str,
    current_device: str,
    is_interactive: bool | None = None,
    console: Console | None = None,
) -> tuple[str, str, str, str, bool]:
    """Interactive recovery prompt on pipeline errors using Questionary with non-interactive fallback.

    Example:
        >>> llm, whisper, comp, dev, retry = prompt_recovery_menu(err, "gemma", "turbo", "fp16", "cuda")
    """
    out = console or get_err_console()
    interactive = is_interactive if is_interactive is not None else sys.stdin.isatty()

    out.print(Panel(f"[bold red]Pipeline Error:[/bold red] {error}", border_style="red", padding=(0, 1)))

    if "libcublas" in str(error).lower():
        out.print(
            Panel(
                "💡 [bold yellow]Missing CUDA runtime libraries.[/bold yellow]\n"
                "Install CUDA dependencies with: [bold green]uv sync --extra cuda[/bold green]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    if not interactive:
        return current_llm, current_whisper, current_compute, current_device, False

    choices = [
        questionary.Choice(f"1. Enter a new LLM model name (current: {current_llm})", value="1"),
        questionary.Choice(f"2. Enter a new Whisper model name (current: {current_whisper})", value="2"),
        questionary.Choice(f"3. Change Whisper compute type (current: {current_compute})", value="3"),
        questionary.Choice(f"4. Change Whisper device (current: {current_device})", value="4"),
        questionary.Choice("5. Retry pipeline with current configuration", value="5"),
        questionary.Choice("6. Exit", value="6"),
    ]

    selected = questionary.select("Model or pipeline failure detected. Choose recovery option:", choices=choices).ask()

    if selected in RECOVERY_DISPATCH:
        prompt_msg, idx = RECOVERY_DISPATCH[selected]
        new_val = questionary.text(prompt_msg).ask()
        if new_val and new_val.strip():
            configs = [current_llm, current_whisper, current_compute, current_device]
            configs[idx] = new_val.strip()
            return configs[0], configs[1], configs[2], configs[3], True
    elif selected == "5":
        return current_llm, current_whisper, current_compute, current_device, True

    return current_llm, current_whisper, current_compute, current_device, False



def render_interactive_banner(
    summary_path: Path,
    srt_path: Path | None,
    llm_model: str,
    strategy_name: str,
    console: Console | None = None,
) -> None:
    """Display interactive session header with active files, strategy, and slash commands."""
    out = console or get_console()
    srt_disp = (
        f"[bold green]{srt_path.name}[/bold green] [dim]({srt_path})[/dim]"
        if srt_path
        else "[dim yellow]Not found (Ata-only mode)[/dim yellow]"
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column(style="white")
    grid.add_row("Summary File:", f"[bold magenta]{summary_path.name}[/bold magenta] [dim]({summary_path})[/dim]")
    grid.add_row("SRT Grounding:", srt_disp)
    grid.add_row("LLM & Strategy:", f"[bold]{llm_model}[/bold] [dim]({strategy_name})[/dim]")
    grid.add_row(
        "Commands:",
        "[cyan]/search <term>[/cyan] · [cyan]/show[/cyan] · [cyan]/diff[/cyan] · [cyan]/undo[/cyan] · [cyan]/help[/cyan] · [cyan]/exit[/cyan]",
    )
    grid.add_row(
        "Tip:",
        "[dim]Tell me who participated, assign action-item owners, add meeting context, or ask about the transcript.[/dim]",
    )
    out.print(Panel(grid, title="[bold cyan]💬 Interactive Ata Refinement[/bold cyan]", border_style="cyan", padding=(1, 2)))


def render_srt_search_results(
    query: str,
    matches: list[SrtMatchWindow],
    console: Console | None = None,
) -> None:
    """Render timestamped SRT search matches inside a styled Rich panel."""
    out = console or get_console()
    if not matches:
        out.print(
            Panel(
                f"[yellow]No SRT matches found for query:[/yellow] [bold]{query}[/bold]",
                title="[bold yellow]🔍 SRT Search[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return

    text = Text()
    for i, win in enumerate(matches):
        if i > 0:
            text.append("\n\n")
        text.append(f"[{win.start_time} -> {win.end_time}] ", style="bold green")
        text.append(win.text, style="white")

    title = f"[bold green]🔍 SRT Grounding ({len(matches)} excerpts for '{query}')[/bold green]"
    out.print(Panel(text, title=title, border_style="green", padding=(1, 2)))


def render_ata_diff(
    old_ata: str,
    new_ata: str,
    title: str = "Proposed Ata Changes",
    console: Console | None = None,
) -> None:
    """Render a colorized unified diff between the current and proposed Ata markdown."""
    out = console or get_console()
    diff_lines = list(difflib.unified_diff(old_ata.splitlines(), new_ata.splitlines(), lineterm="", n=2))
    if not diff_lines:
        out.print("[dim]No structural changes between current and proposed Ata.[/dim]")
        return

    diff_text = Text()
    for line in diff_lines:
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            diff_text.append(f"+ {line[1:].lstrip()}\n", style="bold green")
        elif line.startswith("-"):
            diff_text.append(f"- {line[1:].lstrip()}\n", style="bold red")
        elif line.strip():
            diff_text.append(f"  {line.lstrip()}\n", style="dim")

    out.print(Panel(diff_text, title=f"[bold yellow]± {title}[/bold yellow]", border_style="yellow", padding=(1, 2)))


def prompt_edit_confirmation(summary_filename: str, is_interactive: bool | None = None) -> str:
    """Ask the user whether to accept, discard, or further adjust a proposed Ata update."""
    interactive = is_interactive if is_interactive is not None else sys.stdin.isatty()
    if not interactive:
        return "discard"

    choices = [
        questionary.Choice(f"✔ Accept & save to {summary_filename}", value="accept"),
        questionary.Choice("✖ Discard this change", value="discard"),
        questionary.Choice("✎ Ask model to adjust further", value="adjust"),
    ]
    selected = questionary.select("Apply proposed update to the Ata?", choices=choices).ask()
    return str(selected) if selected in ("accept", "discard", "adjust") else "discard"
