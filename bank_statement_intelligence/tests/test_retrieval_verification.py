from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.classification.example_loader import HistoricalExample, HistoricalExampleSimilarity
from app.retrieval.retrieval_classifier import RetrievalClassifier


class RetrievalVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RetrievalClassifier.__new__(RetrievalClassifier)
        self.classifier.examples = [HistoricalExample(
            description="Office purchase",
            debit=150.0,
            credit=None,
            name="Office Depot",
            item_split_account="RS",
            transaction_type="Expense",
            classification_name="Retrieval Category",
            classification_type="Expense",
        )]
        self.classifier.index = Mock()
        self.classifier.index.is_ready.return_value = True
        self.classifier.embedding_builder = Mock()
        self.classifier.embedding_builder.build_embeddings.return_value = [[0.1]]
        self.classifier.index.search.return_value = [([0], [0.9])]
        self.classifier.candidate_ranker = Mock()
        self.classifier.candidate_ranker.rank_candidates.return_value = [{
            "classification_code": "RS",
            "classification_name": "Retrieval Category",
            "classification_type": "Expense",
            "confidence": 70,
            "method": "Retrieval",
            "score": 0.70,
            "average_similarity": 0.70,
            "maximum_similarity": 0.70,
            "frequency": 1,
            "merchant_match_count": 1,
            "debit_credit_match": 0.5,
            "description_overlap": 0.5,
            "direction_match": 0.5,
        }]
        self.classifier.confidence_engine = Mock()
        self.classifier.confidence_engine.calculate_confidence.return_value = 70
        self.classifier.verification_threshold = 80
        self.classifier.confidence_engine.determine_status.return_value = "Medium"
        self.classifier.confidence_engine.build_confidence_breakdown.return_value = {"confidence": 70}
        self.classifier.verifier = Mock()
        self.classifier.top_k = 5
        self.classifier._retrieve_examples = Mock(return_value=[HistoricalExampleSimilarity(
            example=HistoricalExample(
                description="Office purchase",
                debit=150.0,
                credit=None,
                name="Office Depot",
                item_split_account="RS",
                transaction_type="Expense",
                classification_name="Retrieval Category",
                classification_type="Expense",
            ),
            similarity=0.9,
        )])
        self.classifier._build_query_text = Mock(return_value="query")
        self.classifier._build_ranking_explanation = Mock(return_value="explanation")

    def test_high_confidence_retrieval_skips_gemini(self) -> None:
        self.classifier.confidence_engine.calculate_confidence.return_value = 85
        self.classifier.candidate_ranker.rank_candidates.return_value = [{
            "classification_code": "RS",
            "classification_name": "Retrieval Category",
            "classification_type": "Expense",
            "confidence": 85,
            "method": "Retrieval",
            "score": 0.85,
            "average_similarity": 0.85,
            "maximum_similarity": 0.85,
            "frequency": 1,
            "merchant_match_count": 1,
            "debit_credit_match": 0.5,
            "description_overlap": 0.5,
            "direction_match": 0.5,
        }]

        with patch("app.retrieval.retrieval_classifier.verify_transaction") as verify_mock:
            records = self.classifier.classify_transactions([{"description": "Office purchase"}])

        self.assertEqual(records[0]["method"], "Retrieval")
        verify_mock.assert_not_called()

    def test_low_confidence_retrieval_invokes_gemini(self) -> None:
        with patch("app.retrieval.retrieval_classifier.verify_transaction", return_value={
            "classification_code": "GS",
            "classification_name": "Gemini Category",
            "classification_type": "Expense",
            "confidence": 60,
            "reason": "Gemini reason",
            "method": "Gemini Verification",
        }) as verify_mock:
            self.classifier.classify_transactions([{"description": "Office purchase"}])

        verify_mock.assert_called_once()

    def test_gemini_overrides_retrieval(self) -> None:
        with patch("app.retrieval.retrieval_classifier.verify_transaction", return_value={
            "classification_code": "GS",
            "classification_name": "Gemini Category",
            "classification_type": "Expense",
            "confidence": 90,
            "reason": "Gemini reason",
            "method": "Gemini Verification",
        }):
            records = self.classifier.classify_transactions([{"description": "Office purchase"}])

        self.assertEqual(records[0]["classification_name"], "Gemini Category")
        self.assertEqual(records[0]["method"], "Gemini Verification")
        self.assertEqual(records[0]["verification_method"], "Gemini Verification")
        self.assertEqual(records[0]["verification_confidence"], 90)
        self.assertEqual(records[0]["verification_reason"], "Gemini reason")

    def test_gemini_rejects_override(self) -> None:
        with patch("app.retrieval.retrieval_classifier.verify_transaction", return_value={
            "classification_code": "GS",
            "classification_name": "Gemini Category",
            "classification_type": "Expense",
            "confidence": 60,
            "reason": "Gemini reason",
            "method": "Gemini Verification",
        }):
            records = self.classifier.classify_transactions([{"description": "Office purchase"}])

        self.assertEqual(records[0]["classification_name"], "Retrieval Category")
        self.assertEqual(records[0]["method"], "Retrieval")
        self.assertEqual(records[0]["verification_method"], "Retrieval")
        self.assertEqual(records[0]["verification_confidence"], 70)

    def test_gemini_unavailable_uses_retrieval_metadata(self) -> None:
        with patch("app.retrieval.retrieval_classifier.verify_transaction", side_effect=RuntimeError("quota exceeded")):
            records = self.classifier.classify_transactions([{"description": "Office purchase"}])

        self.assertEqual(records[0]["classification_name"], "Retrieval Category")
        self.assertEqual(records[0]["method"], "Retrieval")
        self.assertEqual(records[0]["verification_method"], "Retrieval Only")
        self.assertEqual(records[0]["verification_reason"], "Gemini unavailable")
        self.assertEqual(records[0]["verification_confidence"], 70)

    def test_output_json_contains_verification_metadata(self) -> None:
        with patch("app.retrieval.retrieval_classifier.verify_transaction", return_value={
            "classification_code": "GS",
            "classification_name": "Gemini Category",
            "classification_type": "Expense",
            "confidence": 90,
            "reason": "Gemini reason",
            "method": "Gemini Verification",
        }):
            records = self.classifier.classify_transactions([{"description": "Office purchase"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "classified.json"
            self.classifier.save_classified_transactions(records, output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("verification_reason", saved[0])
        self.assertEqual(saved[0]["verification_method"], "Gemini Verification")
        self.assertEqual(saved[0]["verification_confidence"], 90)


if __name__ == "__main__":
    unittest.main()
