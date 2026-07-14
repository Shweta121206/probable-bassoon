"""Smoke test script for the Ollama client."""

from __future__ import annotations

import logging

from app.llm import OllamaClient


def main() -> None:
    """Send a simple prompt to Ollama and print the response."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    client = OllamaClient()
    response = client.generate("Hi, my name is shweta")
    print(response)


if __name__ == "__main__":
    main()
