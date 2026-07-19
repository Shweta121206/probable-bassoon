"""Client for generating text with a local Ollama server."""

from __future__ import annotations

import os
import logging
from typing import Any
from xml.parsers.expat import model
from xml.parsers.expat import model

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Small client for Ollama's /api/generate endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            model: Ollama model name to use for generation.
            base_url: Base URL of the local Ollama server.
            timeout_seconds: Request timeout for local model generation.
        """
        if model is None:
            model = os.getenv(
                "OLLAMA_MODEL",
                "llama3.2:3b"
            )

        self.model = model

        if base_url is None:
            base_url = os.getenv(
                "OLLAMA_HOST",
                "http://localhost:11434"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Ollama.

        Args:
            prompt: Prompt text to send to the model.

        Returns:
            The generated response text only.

        Raises:
            TypeError: If prompt is not a string.
            ValueError: If prompt is empty or the response payload is malformed.
            RuntimeError: If the server cannot be reached, times out, or returns
                an HTTP error.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt cannot be empty")

        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": cleaned_prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }

        logger.info(
            "Sending generation request to Ollama model %s with %d second timeout",
            self.model,
            self.timeout_seconds,
        )

        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            logger.exception("Could not connect to Ollama at %s", url)
            raise RuntimeError(f"Could not connect to Ollama at {url}") from exc
        except requests.Timeout as exc:
            logger.exception("Ollama request timed out")
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except requests.HTTPError as exc:
            logger.exception("Ollama returned an HTTP error")
            raise RuntimeError(f"Ollama returned an HTTP error: {exc}") from exc
        except requests.RequestException as exc:
            logger.exception("Ollama request failed")
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.exception("Ollama returned malformed JSON")
            raise ValueError("Ollama returned malformed JSON") from exc

        generated_text = data.get("response")
        if not isinstance(generated_text, str):
            logger.error("Ollama response did not contain generated text")
            raise ValueError("Ollama response did not contain generated text")

        logger.info("Received %d character(s) from Ollama", len(generated_text))
        return generated_text
