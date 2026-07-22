# Local Meeting Pipeline

A high-performance Python-based local pipeline to process meeting audio and video files, transcribe them, and extract meeting bullet points using local models.

---

## 🛠️ Architecture

The pipeline processes files end-to-end entirely on your local machine:

```mermaid
graph TD
    A[Audio or Video Input] --> B[Audio Extraction & Normalization]
    B -->|Convert to WAV 16kHz Mono| C[Transcription with faster-whisper]
    C -->|Generate transcript.txt| D[LLM Meeting Points with local Ollama]
    D -->|Generate meeting_points.md| E[Outputs: Transcript + Summary + Metadata]
```

1. **Audio/Video Extraction & Normalization**: The input file (e.g., `.mp3`, `.mp4`, `.wav`, `.mkv`) is processed via `ffmpeg` to extract the audio channel and normalize it into a standard format: PCM 16-bit, 16000Hz, mono WAV.
2. **Transcription**: The normalized audio is transcribed using the **`faster-whisper`** library.
3. **Summarization (Local LLM)**: The transcript is fed to a local instance of **Ollama** (defaulting to the `gemma:2b` model) to construct structural meeting bullet points (Key Points, Decisions, Actions, and Pendencies) in Portuguese.

---

## 📋 Prerequisites

Before running the pipeline, ensure you have the following installed on your system:

1. **Python >= 3.14**
2. **[uv](https://docs.astral.sh/uv/)** — Fast Python package installer and resolver.
3. **FFmpeg** — Required for audio extraction and format normalization.
   * *Linux*: `sudo apt install ffmpeg` (or your distro's package manager)
   * *macOS*: `brew install ffmpeg`
4. **[Ollama](https://ollama.com/)** — Running locally with your model of choice pulled:
   ```bash
   ollama pull gemma:2b
   ```

---

## 🚀 Installation

Clone the repository and synchronize the environment using `uv`:

```bash
uv sync
```

This will automatically create a virtual environment, install all dependencies (including dev tools like `pytest` and `pyrefly`), and expose the CLI scripts.

---

## 💻 CLI Usage

Run the pipeline using `uv run`:

```bash
uv run meeting-pipeline <path_to_audio_or_video_file> [OPTIONS]
```

### Options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--output-dir` | `Path` | `output` | Directory where output files will be saved |
| `--whisper-model` | `str` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`, etc.) |
| `--whisper-device` | `str` | `cpu` | Device to run Whisper inference on (`cpu` or `cuda`) |
| `--whisper-compute-type`| `str` | `int8` | Model quantization/compute type (`int8`, `float16`, etc.) |
| `--llm-model` | `str` | `gemma:2b` | Ollama model name to use for summarization |
| `--language` | `str` | `pt` | Language code for transcription (e.g., `pt`, `en`) |
| `--verbose` | `flag` | `False` | Enable debug logs |

### Examples

**Standard Audio Run:**
```bash
uv run meeting-pipeline sample.mp3 --whisper-model small --language pt
```

**Fast / Low-Resource Run:**
```bash
uv run meeting-pipeline sample.mp3 --whisper-model tiny --language pt --verbose
```

---

## 📁 Outputs

All outputs are saved to the designated `--output-dir` (default: `output/`):
* **`normalized.wav`**: The 16kHz mono audio track extracted/normalized from the input.
* **`transcript.txt`**: The full text transcription of the meeting.
* **`transcript_metadata.json`**: Metadata containing runtime parameters, detected language, and duration.
* **`meeting_points.md`**: Markdown document summarizing main points, decisions, actions, and pendencies in Portuguese.

---

## 🧪 Development & Quality Assurance

### Run Tests
A comprehensive suite of unit tests covers the CLI, transcription, normalization, and summarization modules. Run the registered test script:

```bash
uv run meeting-pipeline-test
```

### Code Formatting & Quality
Check lint rules using Ruff:
```bash
uv run ruff check
```

### Static Type Checking
Perform static analysis and type checking using the **Pyrefly** LSP/type checker:
```bash
uv run pyrefly check
```
