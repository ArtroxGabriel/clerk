# `clerk` Documentation

Welcome to the technical documentation for **`clerk` (`v3.0.0`)**, a modern Python CLI application designed to automate audio extraction, high-accuracy speech-to-text (STT) transcription, structured Portuguese LLM summarization, and interactive `.srt`-grounded Ata refinement from local media files or YouTube URLs.

---

## 🚀 High-Level Architecture Overview

The `clerk` pipeline coordinates four core stages sequentially, followed by an optional interactive refinement loop:

```mermaid
graph TD
    A["Input Media (File, YouTube URL, .srt, or .md)"] --> B["Audio Normalization (src/clerk/audio.py)"]
    B --> C["16kHz Mono WAV"]
    C --> D["Speech-to-Text Transcription (src/clerk/transcribe.py)"]
    D --> E["Timestamped SRT File"]
    E --> F["LLM Summarization (src/clerk/summarize.py & src/clerk/prompts/)"]
    F --> G["Structured Markdown Summary (_ata.md / _resume.md) & JSON Metadata"]
    G -.->|"--interactive (-i) or .md target"| H["Interactive Refinement & SRT Grounding (src/clerk/interactive.py & src/clerk/srt_search.py)"]
    E -.->|"rg / grep SRT Excerpts"| H
```

---

## 📦 Core Modules Map

| Module | Location | Purpose & Primary Responsibilities |
|---|---|---|
| **Prompts & Strategies** | [`src/clerk/prompts/`](modules/prompts.md) | CPU/GPU prompt strategy protocol, SRT timestamp cleaning, prompt injection protection (`<<<TRANSCRIPT>>>`), and noise guard rails. |
| **Speech-to-Text** | [`src/clerk/transcribe.py`](modules/transcribe.md) | `faster-whisper` (`BatchedInferencePipeline`) integration, VAD filtering, batching, and SRT subtitle formatting. |
| **Summarizer & Ollama** | [`src/clerk/summarize.py`](modules/summarize.md) | Ollama `/api/generate` & `/api/chat` communication, sentence/clause boundary chunking (`split_transcript_smart`), section parsing, and model memory unloading. |
| **Audio Processing** | [`src/clerk/audio.py`](modules/audio.md) | `ffmpeg` audio extraction to 16kHz mono WAV and `yt-dlp` YouTube subtitle fast-path / audio downloads. |
| **Pipeline Coordinator** | [`src/clerk/pipeline.py`](modules/pipeline.md) | Sequential and isolated step execution, execution timing metrics (`hh:mm:ss:mm`), and JSON metadata export. |
| **Interactive & SRT Search** | [`src/clerk/interactive.py`](modules/interactive.md) | Multi-turn interactive Ata refinement (`CpuInteractiveStrategy` & `GpuInteractiveStrategy`), `rg` $\rightarrow$ `grep` $\rightarrow$ Python `.srt` block search (`src/clerk/srt_search.py`), section-safe patching, diff preview, and `/undo`. |
| **CLI Application** | [`src/clerk/cli.py`](modules/cli.md) | Typer CLI commands, option validation (CPU vs GPU limits), presets (`cpu`, `fast`, `gpu`, `cuda`, `accurate`), `-i` / `--srt` flags, and interactive model recovery menu. |

---

## 🛠️ Quick Developer Commands

```bash
# Run test suite
uv run clerk-test

# Run static type checker
uv run pyrefly check

# Run linter and code style check
uv run ruff check
```
