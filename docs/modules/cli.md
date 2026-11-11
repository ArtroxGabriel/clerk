# Module: `src/clerk/cli.py` & `src/clerk/config.py`

The `cli` and `config` modules provide the command-line user interface powered by **Typer**. They handle preset management, upfront option validation, `.srt` and `.md` target auto-detection, execution time formatting, interactive mode entry (`-i` / `--interactive`), and the interactive recovery loop on pipeline errors.

---

## 🎯 Preset Configurations

`config.py` defines five hardware and speed presets (`-p` / `--preset`):

| Preset | Whisper Model | Device | Compute Type | Batch Size | LLM Model |
|---|---|---|---|---|---|
| `cpu` (Default) | `large-v3-turbo` | `cpu` | `int8` | 2 | `LiquidAI/lfm2.5-1.2b-instruct` |
| `fast` | `small` | `cpu` | `int8` | 2 | `LiquidAI/lfm2.5-1.2b-instruct` |
| `gpu` | `large-v3-turbo` | `cuda` | `float16` | 4 | `gemma4:e2b` |
| `cuda` | `large-v3-turbo` | `cuda` | `float16` | 4 | `gemma4:e2b` |
| `accurate` | `large-v3` | `cuda` | `float16` | 4 | `gemma4:e2b` |

---

## 🛠️ Key Implementation Features

### Upfront Option Verification & Target Auto-Detection
Before downloading media or initializing neural models, `resolve_cli_config()` validates all command-line arguments:
- Verifies `--whisper-batch-size >= 1`.
- Verifies `--whisper-device` is valid (`cpu`, `cuda`, `gpu`, `auto`).
- Rejects GPU-only compute types (`float16`, `int8_float16`, `bfloat16`) if running on CPU (`-p cpu`), informing the user of valid CPU types (`int8`, `float32`, `default`).
- Rejects mutually exclusive flags (`--resume` + `--force`, `--video` + `--meeting`, `--gpu` + `--fast`, `--transcribe-only` + `--summarize-only`, `--transcribe-only` + `--interactive`).
- **Auto-Detection by File Extension**:
  - Passing a `.srt` target automatically enables `summarize_only=True`.
  - Passing an existing `.md` target (`*_ata.md` or `*_resume.md`) automatically enables `interactive=True` and `interactive_only=True`, skipping the batch pipeline and resolving its companion `.srt` file directly.

---

### `format_time_hhmmssmm(seconds: float) -> str`
Formats execution durations and audio lengths in `hh:mm:ss:mm` (hour, minute, second, centisecond) format.

---

### Interactive Model & Pipeline Recovery Loop
If a model load or execution fails during pipeline processing (e.g. invalid model name, missing weights, or out of VRAM), `cli.py` catches the error and launches an interactive CLI menu via `questionary`:

```text
Model or pipeline failure detected. Choose recovery option:
  1. Enter a new LLM model name (current: LiquidAI/lfm2.5-1.2b-instruct)
  2. Enter a new Whisper model name (current: large-v3-turbo)
  3. Change Whisper compute type (current: int8)
  4. Change Whisper device (current: cpu)
  5. Retry pipeline with current configuration
  6. Exit
```

This allows users to change parameters or enter corrected model names without losing downloaded YouTube audio or restarting completed steps (`effective_resume = True`).
