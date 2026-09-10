from __future__ import annotations

import httpx
import pytest

from clerk.backends.base import LLMBackend
from clerk.backends.factory import get_backend
from clerk.backends.ollama import OllamaBackend


class FakeHttpResponse:
    """Fake HTTP response simulating httpx.Response."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or ("" if json_data is None else str(json_data))

    def json(self) -> dict:
        return self._json_data


class FakeHttpClient:
    """Fake HTTP client simulating httpx.Client for testing OllamaBackend."""

    def __init__(self, responses: list[FakeHttpResponse | Exception]) -> None:
        self._responses = list(responses)
        self.posted_payloads: list[dict] = []
        self.posted_urls: list[str] = []

    def __enter__(self) -> FakeHttpClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass

    def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> FakeHttpResponse:
        self.posted_urls.append(url)
        self.posted_payloads.append(json or {})
        if not self._responses:
            return FakeHttpResponse(status_code=200)
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def test_ollama_backend_satisfies_protocol() -> None:
    # Arrange & Act
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Assert
    assert isinstance(backend, LLMBackend)


def test_ollama_backend_generates_response_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    fake_client = FakeHttpClient([FakeHttpResponse(200, {"response": "Structured summary."})])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: fake_client)
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Act
    output = backend.generate("Meeting transcript")

    # Assert
    assert output == "Structured summary."
    assert fake_client.posted_urls == ["/api/generate"]
    assert fake_client.posted_payloads[0]["model"] == "test-model"
    assert fake_client.posted_payloads[0]["prompt"] == "Meeting transcript"


def test_ollama_backend_auto_pulls_missing_model_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    fake_client = FakeHttpClient([
        FakeHttpResponse(404, text="model 'test-model' not found"),
        FakeHttpResponse(200, {"status": "success"}),
        FakeHttpResponse(200, {"response": "Retried summary."}),
    ])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: fake_client)
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Act
    output = backend.generate("Meeting transcript")

    # Assert
    assert output == "Retried summary."
    assert fake_client.posted_urls == ["/api/generate", "/api/pull", "/api/generate"]
    assert fake_client.posted_payloads[1]["name"] == "test-model"


def test_ollama_backend_raises_clear_error_when_server_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    connect_error = httpx.ConnectError("Connection refused")
    fake_client = FakeHttpClient([connect_error])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: fake_client)
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Act & Assert
    with pytest.raises(RuntimeError, match="Could not connect to local Ollama server at http://127.0.0.1:11434"):
        backend.generate("Meeting transcript")


def test_ollama_backend_raises_clear_error_when_status_not_200(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    fake_client = FakeHttpClient([FakeHttpResponse(500, text="Internal server error")])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: fake_client)
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Act & Assert
    with pytest.raises(RuntimeError, match="ollama request failed: 500 Internal server error"):
        backend.generate("Meeting transcript")


def test_ollama_backend_cleanup_unloads_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    fake_client = FakeHttpClient([FakeHttpResponse(200, {})])
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: fake_client)
    backend = OllamaBackend(model_name="test-model", base_url="http://127.0.0.1:11434")

    # Act
    backend.cleanup()

    # Assert
    assert fake_client.posted_urls == ["/api/generate"]
    assert fake_client.posted_payloads[0] == {"model": "test-model", "keep_alive": 0}


def test_get_backend_factory_returns_ollama_backend() -> None:
    # Arrange & Act
    backend = get_backend("ollama", model_name="custom-model")

    # Assert
    assert isinstance(backend, OllamaBackend)
    assert backend.model_name == "custom-model"


def test_get_backend_factory_raises_error_for_unsupported_backend() -> None:
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Unsupported backend: 'unsupported'"):
        get_backend("unsupported")
