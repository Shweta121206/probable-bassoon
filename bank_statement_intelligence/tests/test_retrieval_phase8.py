from __future__ import annotations

import unittest

from app.classification.example_loader import HistoricalExample, HistoricalExampleSimilarity
from app.retrieval.candidate_ranker import CandidateRanker
from app.retrieval.confidence_engine import ConfidenceEngine
from app.retrieval.direction_utils import DirectionUtils
from app.retrieval.merchant_extractor import MerchantExtractor


class MerchantExtractorTests(unittest.TestCase):
    def test_extract_merchant_normalizes_common_patterns(self) -> None:
        self.assertEqual(MerchantExtractor.extract_merchant("POS PURCHASE SQ *STARBUCKS 1234"), "starbucks")
        self.assertEqual(MerchantExtractor.extract_merchant("AMAZON MKTPLACE ORDER"), "amazon")
        self.assertEqual(MerchantExtractor.extract_merchant("BKOFAMERICA MOBILE PAYMENT"), "bank of america")


class DirectionUtilsTests(unittest.TestCase):
    def test_detect_direction_handles_debit_credit_and_type(self) -> None:
        self.assertEqual(DirectionUtils.detect_direction(debit=25.0, credit=None, transaction_type="Expense"), "Withdrawal")
        self.assertEqual(DirectionUtils.detect_direction(debit=None, credit=1000.0, transaction_type="Deposit"), "Deposit")
        self.assertEqual(DirectionUtils.detect_direction(debit=100.0, credit=50.0, transaction_type="Transfer"), "Transfer")
        self.assertEqual(DirectionUtils.detect_direction(debit=100.0, credit=100.0, transaction_type="Expense"), "Unknown")


class CandidateRankerTests(unittest.TestCase):
    def test_rank_candidates_prefers_merchant_and_direction_matches(self) -> None:
        transaction = {
            "description": "Office Depot office supplies",
            "transaction_type": "Expense",
            "debit": 150.0,
            "credit": None,
        }
        examples = [
            HistoricalExampleSimilarity(
                example=HistoricalExample(
                    description="Office Depot purchase",
                    debit=150.0,
                    credit=None,
                    name="Office Depot",
                    item_split_account="Office Supplies",
                    transaction_type="Expense",
                    classification_name="Office Supplies",
                    classification_type="Expense",
                ),
                similarity=0.74,
            ),
            HistoricalExampleSimilarity(
                example=HistoricalExample(
                    description="Airline ticket purchase",
                    debit=300.0,
                    credit=None,
                    name="Delta Airlines",
                    item_split_account="Travel",
                    transaction_type="Expense",
                    classification_name="Travel",
                    classification_type="Expense",
                ),
                similarity=0.80,
            ),
        ]

        ranked = CandidateRanker().rank_candidates(transaction, examples)

        self.assertEqual(ranked[0]["classification_name"], "Office Supplies")


class ConfidenceEngineTests(unittest.TestCase):
    def test_confidence_increases_for_strong_merchant_and_direction_alignment(self) -> None:
        candidate = {
            "average_similarity": 0.80,
            "maximum_similarity": 0.80,
            "frequency": 10,
            "merchant_match_count": 8,
            "debit_credit_match": 1.0,
            "description_overlap": 0.5,
            "direction_match": 1.0,
        }

        confidence = ConfidenceEngine().calculate_confidence(candidate)

        self.assertGreater(confidence, 75)


if __name__ == "__main__":
    unittest.main()
