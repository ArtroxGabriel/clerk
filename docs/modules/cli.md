# Module: `src/clerk/cli.py`

The `cli` module provides the command-line user interface powered by **Typer**. It handles preset management, upfront option validation, execution time formatting, and the interactive recovery loop on pipeline errors.

---

## 🎯 Preset Configurations

`cli.py` defines five hardware and speed presets (`-p` / `--preset`):

| Preset | Whisper Model | Device | Compute Type | Batch Size | LLM Model |
|---|---|---|---|---|---|
| `cpu` (Default) | `large-v3` | `cpu` | `int8` | 2 | `LiquidAI/lfm2.5-1.2b-instruct` |
| `fast` | `small` | `cpu` | `int8` | 2 | `LiquidAI/lfm2.5-1.2b-instruct` |
| `gpu` | `large-v3` | `cuda` | `float16` | 8 | `llama3.1:8b` |
| `cuda` | `large-v3` | `cuda` | `float16` | 8 | `llama3.1:8b` |
| `accurate` | `large-v3` | `cuda` | `float16` | 4 | `llama3.1:8b` |

---

## 🛠️ Key Implementation Features

### Upfront Option Verification
Before downloading media or initializing neural models, `cli.py` validates all command-line arguments:
- Verifies `--whisper-batch-size >= 1`.
- Verifies `--whisper-device` is valid (`cpu`, `cuda`, `gpu`, `auto`).
- Rejects GPU-only compute types (`float16`, `int8_float16`, `bfloat16`) if running on CPU (`-p cpu`), informing the user of valid CPU types (`int8`, `float32`, `default`).

---

### `format_time_hhmmssmm(seconds: float) -> str`
Formats execution durations and audio lengths in `hh:mm:ss:mm` (hour, minute, second, centisecond) format.

**Constants**:
- `CS_PER_HOUR = 360_000`
- `CS_PER_MINUTE = 6_000`
- `CS_PER_SECOND = 100`

---

### Interactive Model & Pipeline Recovery Loop
If a model load or execution fails during pipeline processing (e.g. invalid model name, missing weights, or out of VRAM), `cli.py` catches the error and launches an interactive CLI menu:

```text
⚠️  Pipeline error: ...
Model or pipeline failure detected. Choose recovery option:
  [1] Enter a new LLM model name (current: LiquidAI/lfm2.5-1.2b-instruct)
  [2] Enter a new Whisper model name (current: large-v3)
  [3] Change Whisper compute type (current: float16)
  [4] Change Whisper device (current: cpu)
  [5] Retry pipeline with current configuration
  [6] Exit
Select option [1-6] [6]:
```

This allows users to change parameters or enter corrected model names without losing downloaded YouTube audio or restarting from scratch.
