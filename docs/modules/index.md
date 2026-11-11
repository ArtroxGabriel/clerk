# Modules Overview

The `clerk` codebase is structured into modular, decoupled Python components under `src/clerk/`. Each module is designed with single-responsibility design principles.

---

## 🗺️ Module Reference Index

### 1. [Prompts & Security (`src/clerk/prompts/`)](prompts.md)
Implements the Strategy Pattern (`CpuPromptStrategy`, `GpuPromptStrategy`, and `CustomPromptStrategy`), SRT timestamp stripping (`clean_srt_for_prompt`), prompt injection protection via `<<<TRANSCRIPT>>>` delimiters, tag sanitization (`clean_llm_output`), and ASR noise guard rails (`is_meaningful_transcript`).

### 2. [Speech-to-Text (`src/clerk/transcribe.py`)](transcribe.md)
Interfaces with `faster-whisper` (`BatchedInferencePipeline`) for CTranslate2 accelerated neural transcription. Features VAD filtering, CUDA library preloading, and SRT block generator output.

### 3. [Summarization & Ollama (`src/clerk/summarize.py` & `src/clerk/backends/`)](summarize.md)
Manages HTTP communication with local Ollama LLMs (`http://127.0.0.1:11434`) via `/api/generate` and `/api/chat`, boundary-aware transcript chunking (`split_transcript_smart`), section header parsing, entity phonetic reconciliation, missing model auto-pulling (`/api/pull`), and VRAM model unloading (`keep_alive: 0`).

### 4. [Audio Processing (`src/clerk/audio.py`)](audio.md)
Handles system binary checks (`ffmpeg`, `yt-dlp`), YouTube subtitle fast-path (`download_youtube_subtitles`), title-restricted YouTube audio downloading, and 16kHz mono WAV audio normalization via `ffmpeg`.

### 5. [Pipeline Coordinator (`src/clerk/pipeline.py`)](pipeline.md)
Coordinates sequential and isolated stage execution (Audio $\rightarrow$ Transcribe $\rightarrow$ Summarize), immediate step file persistence, execution timing metrics calculation, and JSON metadata generation (`<stem>_metadata.json`).

### 6. [Interactive Refinement & SRT Search (`src/clerk/interactive.py` & `src/clerk/srt_search.py`)](interactive.md)
Provides a multi-turn interactive TUI session (`--interactive` / `-i`) backed by `rg` $\rightarrow$ `grep` $\rightarrow$ Python `.srt` block windowing, `CpuInteractiveStrategy` (deterministic SRT pre-search) vs. `GpuInteractiveStrategy` (agentic `search_srt` tool-calling), section-safe Markdown patching, Rich diff preview, and `/undo` persistence.

### 7. [CLI Application (`src/clerk/cli.py`)](cli.md)
Entrypoint powered by Typer. Handles hardware/speed presets (`cpu`, `fast`, `gpu`, `cuda`, `accurate`), upfront CPU compute type validation, `.srt` and `.md` target auto-detection, centisecond execution time formatting (`hh:mm:ss:mm`), and the interactive error recovery loop.
