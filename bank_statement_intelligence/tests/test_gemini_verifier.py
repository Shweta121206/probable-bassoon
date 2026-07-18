from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.verification.gemini_verifier import GeminiVerifier, verify_transaction


class GeminiVerifierTests(unittest.TestCase):
    def test_returns_retrieval_result_when_confidence_is_high(self) -> None:
        verifier = GeminiVerifier(api_key="test-key", high_confidence_threshold=80)
        transaction = {
            "confidence": 90,
            "classification_name": "Office Supplies",
            "classification_code": "OS",
            "classification_type": "Expense",
        }

        result = verifier.verify_transaction(transaction)

        self.assertEqual(result["method"], "Retrieval")
        self.assertEqual(result["classification_name"], "Office Supplies")

    def test_calls_gemini_for_low_confidence(self) -> None:
        verifier = GeminiVerifier(api_key="test-key", high_confidence_threshold=80)
        transaction = {
            "confidence": 40,
            "original_transaction": {"description": "Office supplies purchase"},
            "top_candidates": [{"classification_name": "Office Supplies"}],
            "retrieved_neighbors": [{"historical_description": "Office supplies purchase"}],
        }

        fake_response = Mock()
        fake_response.text = '{"classification_code":"OS","classification_name":"Office Supplies","classification_type":"Expense","confidence":91,"reason":"Strong match"}'
        with patch.object(verifier, "_call_gemini", return_value={
            "classification_code": "OS",
            "classification_name": "Office Supplies",
            "classification_type": "Expense",
            "confidence": 91,
            "reason": "Strong match",
        }) as mock_call:
            result = verifier.verify_transaction(transaction)

        self.assertEqual(result["method"], "Gemini Verification")
        self.assertEqual(result["classification_name"], "Office Supplies")
        mock_call.assert_called_once()

    def test_helper_function_uses_default_verifier(self) -> None:
        with patch("app.verification.gemini_verifier.GeminiVerifier") as mock_verifier_cls:
            instance = mock_verifier_cls.return_value
            instance.verify_transaction.return_value = {"method": "Gemini Verification"}

            result = verify_transaction({"confidence": 40})

        self.assertEqual(result["method"], "Gemini Verification")
        mock_verifier_cls.assert_called_once()

    def test_returns_retrieval_fallback_when_gemini_raises(self) -> None:
        verifier = GeminiVerifier(api_key="test-key", high_confidence_threshold=80)
        transaction = {
            "confidence": 40,
            "classification_name": "Office Supplies",
            "classification_code": "OS",
            "classification_type": "Expense",
        }

        with patch.object(verifier, "_call_gemini", side_effect=RuntimeError("403 forbidden")):
            result = verifier.verify_transaction(transaction)

        self.assertEqual(result["method"], "Retrieval")
        self.assertEqual(result["classification_name"], "Office Supplies")
        self.assertEqual(result["verification_method"], "Retrieval Only")
        self.assertEqual(result["verification_reason"], "Gemini unavailable")
        self.assertEqual(result["verification_confidence"], 40)


if __name__ == "__main__":
    unittest.main()
