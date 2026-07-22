from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Você receberá a transcrição de uma reunião.
Retorne em português, de forma objetiva, usando apenas o conteúdo explícito da transcrição.

Formato obrigatório:
## Pontos principais
- ...

## Decisões
- ...

## Ações
- ...

## Pendências
- ...

Não invente fatos ausentes.
Transcrição:
{transcript}
""".strip()


def summarize_transcript(
    transcript: str,
    model_name: str = "gemma:2b",
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 300.0,
) -> str:
    if not transcript.strip():
        logger.error("Transcript is empty")
        raise ValueError("transcript is empty")

    payload = {
        "model": model_name,
        "prompt": PROMPT_TEMPLATE.format(transcript=transcript),
        "stream": False,
    }

    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        response = client.post("/api/generate", json=payload)

    if response.status_code != 200:
        logger.error("Ollama request failed: %s %s", response.status_code, response.text)
        raise RuntimeError("ollama request failed")

    data = response.json()
    content = data.get("response", "").strip()

    if content:
        return content

    logger.error("Ollama returned empty response")
    raise RuntimeError("empty summary")
