from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from meeting_pipeline.summarize import summarize_transcript


def test_summarize_empty_transcript() -> None:
    with pytest.raises(ValueError, match="transcript is empty"):
        summarize_transcript("   ")


def test_summarize_success() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "## Pontos principais\n- Reunião produtiva"}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_transcript("Some transcript content")
        assert result == "## Pontos principais\n- Reunião produtiva"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        # Verify the request payload has correct structure
        payload = call_args[1]["json"]
        assert payload["model"] == "gemma:2b"
        assert "Some transcript content" in payload["prompt"]


def test_summarize_http_error() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="ollama request failed"):
            summarize_transcript("Some transcript content")


def test_summarize_empty_response_from_ollama() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": ""}

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="empty summary"):
            summarize_transcript("Some transcript content")
