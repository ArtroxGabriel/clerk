# Modules Overview

The `clerk` codebase is structured into modular, decoupled Python components under `src/clerk/`. Each module is designed with single-responsibility design principles.

---

## 🗺️ Module Reference Index

### 1. [Prompts & Security (`src/clerk/prompts.py`)](prompts.md)
Implements the Strategy Pattern (`CpuPromptStrategy` and `GpuPromptStrategy`), SRT timestamp stripping (`clean_srt_for_prompt`), prompt injection protection via `<<<TRANSCRIPT>>>` delimiters, tag sanitization (`clean_llm_output`), and ASR noise guard rails (`is_meaningful_transcript`).

### 2. [Speech-to-Text (`src/clerk/transcribe.py`)](transcribe.md)
Interfaces with `faster-whisper` (`BatchedInferencePipeline`) for CTranslate2 accelerated neural transcription. Features VAD filtering, automatic compute type adaptation (`int8` to `int8_float16` on CUDA), and SRT block generator output.

### 3. [Summarization & Ollama (`src/clerk/summarize.py`)](summarize.md)
Manages HTTP communication with local Ollama LLMs (`http://127.0.0.1:11434`), boundary-aware transcript chunking (`split_transcript_smart`), section header parsing, missing model auto-pulling (`/api/pull`), and VRAM model unloading (`keep_alive: 0`).

### 4. [Audio Processing (`src/clerk/audio.py`)](audio.md)
Handles system binary checks (`ffmpeg`, `yt-dlp`), title-restricted YouTube audio downloading (`--restrict-filenames`), and 16kHz mono WAV audio normalization via `ffmpeg`.

### 5. [Pipeline Coordinator (`src/clerk/pipeline.py`)](pipeline.md)
Coordinates sequential stage execution (Audio $\rightarrow$ Transcribe $\rightarrow$ Summarize), execution timing metrics calculation, and JSON metadata generation (`<stem>_metadata.json`).

### 6. [CLI Application (`src/clerk/cli.py`)](cli.md)
Entrypoint powered by Typer. Handles hardware/speed presets (`cpu`, `fast`, `gpu`, `cuda`, `accurate`), upfront CPU compute type validation, centisecond execution time formatting (`hh:mm:ss:mm`), and the interactive error recovery loop.
