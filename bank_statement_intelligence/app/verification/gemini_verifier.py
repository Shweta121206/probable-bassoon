from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 80


def _is_gemini_enabled() -> bool:
    value = os.getenv("ENABLE_GEMINI_VERIFICATION", "true")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class GeminiVerifier:
    """Verify a retrieval classification with Gemini when confidence is low."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = GEMINI_MODEL_NAME,
        high_confidence_threshold: int = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.high_confidence_threshold = high_confidence_threshold
        self.client = self._create_client()

    def _create_client(self) -> Any | None:
        if not self.api_key:
            logger.warning("Gemini verification is unavailable because no API key is configured.")
            return None
        if genai is None:
            logger.warning("Gemini verification is unavailable because the Google GenAI package is not installed.")
            return None

        try:
            return genai.Client(api_key=self.api_key)
        except Exception as exc:
            logger.warning("Gemini verification could not be initialized: %s", exc)
            return None

    def verify_transaction(self, transaction: dict[str, Any]) -> dict[str, Any]:
        confidence = int(transaction.get("confidence", 0))
        if confidence >= self.high_confidence_threshold:
            return self._return_retrieval_result(transaction, reason="Gemini skipped")

        if not _is_gemini_enabled():
            logger.info("Gemini verification is disabled by configuration; using retrieval classification.")
            return self._return_retrieval_result(transaction, reason="Gemini disabled")

        if self.client is None:
            logger.warning("Gemini verification unavailable; using retrieval classification.")
            return self._return_retrieval_result(transaction, reason="Gemini unavailable")

        prompt = self._build_prompt(transaction)
        try:
            result = self._call_gemini(prompt)
        except Exception as exc:
            logger.warning("Gemini verification failed unexpectedly: %s", exc)
            return self._return_retrieval_result(transaction, reason="Gemini unavailable")

        return self._build_verification_result(transaction, result)

    def _call_gemini(self, prompt: str) -> dict[str, Any]:
        """Call Gemini API and parse the response."""
        try:
            if self.client is None:
                raise RuntimeError("Gemini client is not initialized")
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return self._parse_response(response)
        except Exception as exc:
            error_str = str(exc).lower()
            if "model" in error_str and "not found" in error_str:
                self._handle_model_not_found_error(exc)
            else:
                logger.warning("Gemini verification failed: %s", exc)
            return {
                "classification_code": "",
                "classification_name": "",
                "classification_type": "",
                "confidence": 0,
                "reason": "Gemini unavailable",
            }

    def _build_prompt(self, transaction: dict[str, Any]) -> str:
        return self._build_verification_prompt(transaction)

    def _build_verification_prompt(self, transaction: dict[str, Any]) -> str:
        original_transaction = json.dumps(transaction.get("original_transaction") or transaction, indent=2, default=str)
        top_candidates = json.dumps(transaction.get("top_candidates") or [], indent=2, default=str)
        retrieved_neighbors = json.dumps(transaction.get("retrieved_neighbors") or [], indent=2, default=str)

        return f"""You are an accounting classification assistant.
Review the transaction below and pick the single best accounting category.
Return ONLY valid JSON in this exact shape:
{{"classification_code":"","classification_name":"","classification_type":"","confidence":0,"reason":""}}

Transaction details:
{original_transaction}

Top retrieved candidates:
{top_candidates}

Retrieved neighbors / historical examples:
{retrieved_neighbors}

Use the historical evidence and the retrieval context to choose the most appropriate accounting category. Do not include markdown or extra text."""

    def _parse_response(self, response: Any) -> dict[str, Any]:
        text = self._extract_text(response)
        parsed = json.loads(text)
        return {
            "classification_code": str(parsed.get("classification_code", "")).strip(),
            "classification_name": str(parsed.get("classification_name", "")).strip(),
            "classification_type": str(parsed.get("classification_type", "")).strip(),
            "confidence": int(parsed.get("confidence", 0)),
            "reason": str(parsed.get("reason", "")).strip(),
        }

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise ValueError("Gemini response is None")
        if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                text = self._extract_text_from_content(candidate)
                if text.strip():
                    return text
        raise ValueError("Unable to extract text from Gemini response")

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                return content["text"]
            return json.dumps(content)
        if hasattr(content, "text") and isinstance(getattr(content, "text"), str):
            return getattr(content, "text")
        if hasattr(content, "parts") and getattr(content, "parts") is not None:
            return "".join(self._extract_text_from_content(part) for part in getattr(content, "parts"))
        return str(content)

    def _build_verification_result(self, transaction: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        merged = dict(transaction)
        merged.update({
            "classification_code": result.get("classification_code", transaction.get("classification_code", "")),
            "classification_name": result.get("classification_name", transaction.get("classification_name", "")),
            "classification_type": result.get("classification_type", transaction.get("classification_type", "")),
            "confidence": result.get("confidence", transaction.get("confidence", 0)),
            "reason": result.get("reason", transaction.get("reason", "")),
            "method": "Gemini Verification",
            "verification_method": "Gemini Verification",
            "verification_reason": result.get("reason", transaction.get("reason", "")),
            "verification_confidence": result.get("confidence", transaction.get("confidence", 0)),
        })
        return merged

    def _return_retrieval_result(self, transaction: dict[str, Any], reason: str) -> dict[str, Any]:
        merged = dict(transaction)
        merged.update({
            "classification_code": transaction.get("classification_code", ""),
            "classification_name": transaction.get("classification_name", ""),
            "classification_type": transaction.get("classification_type", ""),
            "confidence": transaction.get("confidence", 0),
            "reason": transaction.get("reason", ""),
            "method": "Retrieval",
            "verification_method": "Retrieval Only" if reason != "Gemini skipped" else "Retrieval",
            "verification_reason": reason if reason != "Gemini skipped" else "",
            "verification_confidence": transaction.get("confidence", 0),
        })
        return merged

    def _handle_model_not_found_error(self, exc: Exception) -> None:
        """Handle model-not-found error by logging available models."""
        logger.error(
            "Configured Gemini model '%s' not found. Error: %s",
            self.model_name,
            exc,
        )
        logger.info("Attempting to list available models...")
        try:
            available_models = self.client.models.list()
            model_names = [model.name for model in available_models]
            logger.info("Available Gemini models: %s", ", ".join(model_names))
        except Exception as list_exc:
            logger.warning("Could not list available models: %s", list_exc)


def verify_transaction(transaction: dict[str, Any], verifier: GeminiVerifier | None = None) -> dict[str, Any]:
    active_verifier = verifier or GeminiVerifier()
    return active_verifier.verify_transaction(transaction)
