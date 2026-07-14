from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3


@dataclass(frozen=True)
class ClassificationResult:
    classification_code: str
    classification_name: str
    classification_type: str
    confidence: int
    reason: str
    raw_response: dict[str, Any] | None = None


class GeminiClassifier:
    """Classify transactions using Google Gemini."""

    def __init__(self, api_key: str | None = None, model_name: str = GEMINI_MODEL_NAME) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = self._create_client()

    def _create_client(self) -> genai.Client:
        if not self.api_key:
            raise ValueError("Gemini API key is not configured")
        return genai.Client(api_key=self.api_key)

    def classify_transaction(self, prompt: str) -> ClassificationResult:
        logger.info("Classifying transaction with Gemini model %s", self.model_name)
        last_exception: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return self._parse_response(response)
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "Gemini classification attempt %d failed: %s",
                    attempt,
                    exc,
                )
                time.sleep(1)

        logger.error("Gemini classification failed after %d attempts", MAX_RETRIES)
        raise RuntimeError("Gemini classification failed") from last_exception

    def classify_transactions(self, prompts: list[str]) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        for prompt in prompts:
            try:
                results.append(self.classify_transaction(prompt))
            except Exception as exc:
                logger.warning("Skipping failed Gemini classification: %s", exc)
        return results

    def _parse_response(self, response: Any) -> ClassificationResult:
        try:
            text = self._extract_text(response)
            parsed = json.loads(text)
        except ValueError as exc:
            logger.warning("Gemini returned invalid JSON: %s", exc)
            raise

        result = ClassificationResult(
            classification_code=str(parsed.get("classification_code", "")).strip(),
            classification_name=str(parsed.get("classification_name", "")).strip(),
            classification_type=str(parsed.get("classification_type", "")).strip(),
            confidence=int(parsed.get("confidence", 0)),
            reason=str(parsed.get("reason", "")).strip(),
            raw_response=self._extract_raw_response(response),
        )

        if not result.classification_code or not result.classification_name or not result.classification_type:
            raise ValueError("Gemini response did not contain required classification fields")

        return result

    def _extract_raw_response(self, response: Any) -> dict[str, Any] | None:
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return dumped
        if hasattr(response, "__dict__"):
            return {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
        return None

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise ValueError("Gemini response is None")

        if hasattr(response, "parsed") and response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, dict):
                return json.dumps(parsed)
            if hasattr(parsed, "model_dump"):
                dumped = parsed.model_dump()
                if isinstance(dumped, dict):
                    return json.dumps(dumped)
                if isinstance(dumped, str):
                    return dumped

        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            return response.text

        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                candidate_text = self._extract_text_from_content(candidate)
                if candidate_text.strip():
                    return candidate_text

        if hasattr(response, "parts") and response.parts:
            parts_text = "".join(self._extract_text_from_content(part) for part in response.parts)
            if parts_text.strip():
                return parts_text

        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            if isinstance(dumped, dict) and "text" in dumped and isinstance(dumped["text"], str):
                return dumped["text"]
            return json.dumps(dumped)

        raise ValueError("Unable to extract text from Gemini response")

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                return content["text"]
            if "parts" in content and isinstance(content["parts"], list):
                return "".join(self._extract_text_from_content(part) for part in content["parts"])
            return json.dumps(content)

        if hasattr(content, "text") and isinstance(getattr(content, "text"), str):
            return getattr(content, "text")

        if hasattr(content, "parts") and getattr(content, "parts") is not None:
            return "".join(self._extract_text_from_content(part) for part in getattr(content, "parts"))

        if hasattr(content, "content"):
            return self._extract_text_from_content(getattr(content, "content"))

        if hasattr(content, "model_dump"):
            dumped = content.model_dump()
            if isinstance(dumped, dict):
                return self._extract_text_from_content(dumped)
            return str(dumped)

        if isinstance(content, list):
            return "".join(self._extract_text_from_content(item) for item in content)

        return str(content)
