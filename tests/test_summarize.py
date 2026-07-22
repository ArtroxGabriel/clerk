from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from meeting_pipeline.summarize import (
    parse_summary_sections,
    split_transcript_by_words,
    summarize_transcript,
)


def test_split_transcript_by_words() -> None:
    text = "word1 word2 word3\nword4 word5 word6"
    chunks = split_transcript_by_words(text, max_words=3)
    assert len(chunks) == 2
    assert chunks[0] == "word1 word2 word3"
    assert chunks[1] == "word4 word5 word6"


def test_parse_summary_sections() -> None:
    summary_text = """
## Pontos principais
- Ponto 1
* Ponto 2

## Decisões
1. Decisão A

## Ações
- Nenhuma registrada.

## Pendências
- Pendência X
"""
    sections = parse_summary_sections(summary_text)
    assert sections["Pontos principais"] == ["Ponto 1", "Ponto 2"]
    assert sections["Decisões"] == ["Decisão A"]
    assert sections["Ações"] == []
    assert sections["Pendências"] == ["Pendência X"]


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
        payload = call_args[1]["json"]
        assert payload["model"] == "LiquidAI/lfm2.5-1.2b-instruct"
        assert "Some transcript content" in payload["prompt"]


def test_summarize_multi_chunk() -> None:
    # 6 words total, max_words_per_chunk=3 -> 2 chunks
    long_transcript = "one two three\nfour five six"

    # Mock Ollama responses: 2 chunk summaries + 4 consolidation responses
    chunk_1_resp = MagicMock()
    chunk_1_resp.status_code = 200
    chunk_1_resp.json.return_value = {"response": "## Pontos principais\n- Point 1"}

    chunk_2_resp = MagicMock()
    chunk_2_resp.status_code = 200
    chunk_2_resp.json.return_value = {"response": "## Pontos principais\n- Point 2"}

    cons_resp = MagicMock()
    cons_resp.status_code = 200
    cons_resp.json.return_value = {"response": "- Consolidated Point"}

    mock_client = MagicMock()
    mock_client.post.side_effect = [chunk_1_resp, chunk_2_resp, cons_resp]
    mock_client.__enter__.return_value = mock_client

    with patch("httpx.Client", return_value=mock_client):
        result = summarize_transcript(long_transcript, max_words_per_chunk=3)
        assert "## Pontos principais" in result
        assert "- Consolidated Point" in result
        assert mock_client.post.call_count == 3



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

