from __future__ import annotations

import logging
import httpx

from clerk.prompts import clean_llm_output

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "LiquidAI/lfm2.5-1.2b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 300.0
UNLOAD_TIMEOUT_SECONDS = 10.0
PULL_TIMEOUT_SECONDS = 600.0


class OllamaBackend:
    """Inference backend communicating with a local Ollama daemon.

    Example:
        >>> backend = OllamaBackend("LiquidAI/lfm2.5-1.2b-instruct")
        >>> output = backend.generate("Hello world")
        >>> backend.cleanup()
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        pull_timeout_seconds: float = PULL_TIMEOUT_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.pull_timeout_seconds = pull_timeout_seconds

    def _attempt_auto_pull(self, client: httpx.Client) -> None:
        """Pulls missing model from Ollama library."""
        logger.info("Model '%s' not found locally. Pulling...", self.model_name)
        try:
            resp = client.post(
                "/api/pull",
                json={"name": self.model_name, "stream": False},
                timeout=self.pull_timeout_seconds,
            )
            if resp.status_code != 200:
                logger.error("Failed to pull model '%s': %s", self.model_name, resp.text)
                raise RuntimeError(
                    f"ollama request failed: Model '{self.model_name}' not found and auto-pull failed. "
                    f"Run 'ollama pull {self.model_name}'."
                )
            logger.info("Model '%s' successfully pulled.", self.model_name)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as err:
            logger.error("Network error pulling model '%s': %s", self.model_name, err)
            raise RuntimeError(
                f"ollama request failed: Model '{self.model_name}' not found and cannot be pulled while offline."
            ) from err

    def _execute_generate_post(self, client: httpx.Client, payload: dict) -> str:
        """Posts generate request and handles 404 auto-pull fallback."""
        resp = client.post("/api/generate", json=payload)
        if resp.status_code != 200 and ("not found" in resp.text.lower() or resp.status_code == 404):
            self._attempt_auto_pull(client)
            resp = client.post("/api/generate", json=payload)

        if resp.status_code != 200:
            logger.error("Ollama request failed: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"ollama request failed: {resp.status_code} {resp.text.strip()}")

        data = resp.json()
        return clean_llm_output(data.get("response", "").strip())

    def generate(self, prompt: str) -> str:
        """Generates text from prompt using Ollama."""
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds) as client:
                return self._execute_generate_post(client, payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as err:
            logger.error("Could not connect to Ollama at %s: %s", self.base_url, err)
            raise RuntimeError(
                f"ollama request failed: Could not connect to local Ollama server at {self.base_url}. "
                "Ensure Ollama is running ('ollama serve')."
            ) from err

    def cleanup(self) -> None:
        """Unloads model from Ollama memory via keep_alive: 0."""
        payload = {"model": self.model_name, "keep_alive": 0}
        try:
            with httpx.Client(base_url=self.base_url, timeout=UNLOAD_TIMEOUT_SECONDS) as client:
                client.post("/api/generate", json=payload)
            logger.info("Unloaded Ollama model '%s' from memory", self.model_name)
        except Exception as err:
            logger.warning("Failed to unload Ollama model '%s': %s", self.model_name, err)
