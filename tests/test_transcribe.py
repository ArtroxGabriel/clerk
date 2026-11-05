from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clerk.transcribe import (
    format_timestamp,
    preload_cuda_libraries,
    segments_to_srt,
    transcribe_file,
)


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(61.5) == "00:01:01,500"
    assert format_timestamp(3661.123) == "01:01:01,123"


def test_segments_to_srt() -> None:
    mock_seg_1 = MagicMock()
    mock_seg_1.start = 0.0
    mock_seg_1.end = 2.5
    mock_seg_1.text = " Hello world. "

    mock_seg_2 = MagicMock()
    mock_seg_2.start = 3.0
    mock_seg_2.end = 5.0
    mock_seg_2.text = "This is SRT test."

    plain, srt = segments_to_srt([mock_seg_1, mock_seg_2])
    assert plain == "Hello world.\nThis is SRT test."
    expected_srt = (
        "1\n00:00:00,000 --> 00:00:02,500\nHello world.\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nThis is SRT test."
    )
    assert srt == expected_srt


def test_transcribe_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_file(Path("nonexistent.wav"))


def test_transcribe_success(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio raw data")

    # Mock segments returned by transcribe
    mock_segment_1 = MagicMock()
    mock_segment_1.start = 0.0
    mock_segment_1.end = 2.0
    mock_segment_1.text = " Hello world. "

    mock_segment_2 = MagicMock()
    mock_segment_2.start = 2.0
    mock_segment_2.end = 4.0
    mock_segment_2.text = "   "

    mock_segment_3 = MagicMock()
    mock_segment_3.start = 4.0
    mock_segment_3.end = 6.0
    mock_segment_3.text = "This is a test."

    mock_segments = [mock_segment_1, mock_segment_2, mock_segment_3]

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0
    mock_info.duration_after_vad = 8.5

    mock_model_instance = MagicMock()
    mock_batched_instance = MagicMock()
    mock_batched_instance.transcribe.return_value = (mock_segments, mock_info)

    with patch("clerk.transcribe.WhisperModel", return_value=mock_model_instance) as mock_whisper_class, \
         patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched_instance) as mock_batched_class:
        plain_text, srt_text, metadata = transcribe_file(
            audio_path,
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en"
        )

        mock_whisper_class.assert_called_once_with("tiny", device="cpu", compute_type="int8", download_root=None)
        mock_batched_class.assert_called_once_with(model=mock_model_instance)
        mock_batched_instance.transcribe.assert_called_once_with(
            str(audio_path),
            language="en",
            vad_filter=True,
            word_timestamps=False,
            batch_size=2,
            log_progress=True,
        )

        assert plain_text == "Hello world.\nThis is a test."
        assert "1\n00:00:00,000 --> 00:00:02,000\nHello world." in srt_text
        assert "2\n00:00:04,000 --> 00:00:06,000\nThis is a test." in srt_text
        assert metadata["language"] == "en"
        assert metadata["language_probability"] == 0.99
        assert metadata["duration"] == 10.0
        assert metadata["duration_after_vad"] == 8.5


def test_transcribe_empty_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio raw data")

    mock_segments = []
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0
    mock_info.duration_after_vad = 0.0

    mock_model_instance = MagicMock()
    mock_batched_instance = MagicMock()
    mock_batched_instance.transcribe.return_value = (mock_segments, mock_info)

    with (
        patch("clerk.transcribe.WhisperModel", return_value=mock_model_instance),
        patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched_instance),
        pytest.raises(RuntimeError, match="empty transcript"),
    ):
        transcribe_file(audio_path)


def test_transcribe_gpu_int8_mapping(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio raw data")

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    mock_segment.text = "GPU test."

    mock_info = MagicMock()
    mock_info.language = "pt"
    mock_info.language_probability = 1.0
    mock_info.duration = 1.0
    mock_info.duration_after_vad = 1.0

    mock_model_instance = MagicMock()
    mock_batched_instance = MagicMock()
    mock_batched_instance.transcribe.return_value = ([mock_segment], mock_info)

    with patch("clerk.transcribe.WhisperModel", return_value=mock_model_instance) as mock_whisper_class, \
         patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched_instance):
        transcribe_file(
            audio_path,
            model_name="small",
            device="cuda",
            compute_type="int8",
        )
        mock_whisper_class.assert_called_once_with("small", device="cuda", compute_type="int8_float16", download_root=None)


def test_resolve_download_root_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from clerk.transcribe import _resolve_download_root

    unwritable_dir = tmp_path / "unwritable_hf_cache"
    unwritable_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(unwritable_dir))
    custom_tmp = tmp_path / "custom_temp"

    # Mock touch to raise PermissionError and verify tempfile.gettempdir() is used
    with (
        patch("pathlib.Path.touch", side_effect=PermissionError("Permission denied")),
        patch("clerk.transcribe.tempfile.gettempdir", return_value=str(custom_tmp)),
    ):
        fallback = _resolve_download_root()
        assert fallback == str(custom_tmp / "huggingface_cache")


def test_transcribe_auto_language(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio raw data")

    mock_segment = MagicMock(start=0.0, end=1.0, text="Auto detected lang text.")
    mock_info = MagicMock(language="es", language_probability=0.95, duration=2.0, duration_after_vad=2.0)

    mock_model_instance = MagicMock()
    mock_batched_instance = MagicMock()
    mock_batched_instance.transcribe.return_value = ([mock_segment], mock_info)

    with patch("clerk.transcribe.WhisperModel", return_value=mock_model_instance), \
         patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched_instance):
        _plain_text, _srt_text, metadata = transcribe_file(audio_path, language="auto")
        assert metadata["language"] == "es"
        # Language should have been passed as None to Whisper transcribe call
        mock_batched_instance.transcribe.assert_called_once()
        assert mock_batched_instance.transcribe.call_args[1]["language"] is None


def test_transcribe_releases_vram_via_gc_collect(tmp_path: Path) -> None:
    # Arrange
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio raw data")
    mock_segment = MagicMock(start=0.0, end=1.0, text="VRAM release test.")
    mock_info = MagicMock(language="pt", language_probability=0.99, duration=1.0, duration_after_vad=1.0)
    mock_model_instance = MagicMock()
    mock_batched_instance = MagicMock()
    mock_batched_instance.transcribe.return_value = ([mock_segment], mock_info)

    # Act & Assert
    with (
        patch("clerk.transcribe.WhisperModel", return_value=mock_model_instance),
        patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched_instance),
        patch("clerk.transcribe.gc.collect") as mock_gc,
    ):
        transcribe_file(audio_path, device="cuda")
        mock_gc.assert_called_once()


def test_preload_cuda_libraries_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    fake_lib_dir = tmp_path / "nvidia" / "cublas" / "lib"
    fake_lib_dir.mkdir(parents=True)
    fake_so = fake_lib_dir / "libcublas.so.12"
    fake_so.touch()

    fake_pkg = MagicMock(__path__=[str(fake_lib_dir)])
    monkeypatch.setenv("LD_LIBRARY_PATH", "")

    import builtins
    from collections.abc import Mapping, Sequence
    orig_import = builtins.__import__

    def mock_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "nvidia.cublas.lib":
            return fake_pkg
        if name == "nvidia.cudnn.lib":
            raise ImportError(f"No module named {name}")
        return orig_import(name, globals, locals, fromlist, level)

    # Act
    with (
        patch("builtins.__import__", side_effect=mock_import),
        patch("ctypes.CDLL") as mock_cdll,
    ):
        loaded = preload_cuda_libraries()

    # Assert
    assert loaded is True
    mock_cdll.assert_called_once()
    assert str(fake_lib_dir) in os.environ.get("LD_LIBRARY_PATH", "")


def test_preload_cuda_libraries_missing_packages() -> None:
    # Arrange
    import builtins
    from collections.abc import Mapping, Sequence
    orig_import = builtins.__import__

    def mock_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("nvidia."):
            raise ImportError(f"No module named {name}")
        return orig_import(name, globals, locals, fromlist, level)

    # Act
    with patch("builtins.__import__", side_effect=mock_import):
        loaded = preload_cuda_libraries()

    # Assert
    assert loaded is False


def test_transcribe_triggers_preload_cuda_on_cuda(tmp_path: Path) -> None:
    # Arrange
    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("audio")
    mock_segment = MagicMock(start=0.0, end=1.0, text="text")
    mock_info = MagicMock(language="pt", language_probability=1.0, duration=1.0, duration_after_vad=1.0)
    mock_model = MagicMock()
    mock_batched = MagicMock()
    mock_batched.transcribe.return_value = ([mock_segment], mock_info)

    # Act & Assert
    with (
        patch("clerk.transcribe.WhisperModel", return_value=mock_model),
        patch("clerk.transcribe.BatchedInferencePipeline", return_value=mock_batched),
        patch("clerk.transcribe.preload_cuda_libraries") as mock_preload,
    ):
        transcribe_file(audio_path, device="cuda")
        mock_preload.assert_called_once()



