from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.classification.example_loader import HistoricalExampleSimilarity
from app.retrieval.direction_utils import DirectionUtils
from app.retrieval.merchant_extractor import MerchantExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankedCandidate:
    classification_code: str
    classification_name: str
    classification_type: str
    average_similarity: float
    maximum_similarity: float
    frequency: int
    merchant_match_count: int
    debit_credit_match: float
    description_overlap: float
    direction_match: float
    score: float


class CandidateRanker:
    """Rank retrieval candidates using similarity, frequency, and metadata matches."""

    def rank_candidates(
        self,
        transaction: dict[str, Any],
        examples: list[HistoricalExampleSimilarity],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if not examples:
            return []

        transaction_description = str(transaction.get("description", "")).lower()
        transaction_type = str(transaction.get("transaction_type", "")).lower()
        debit = self._normalize_amount(transaction.get("debit"))
        credit = self._normalize_amount(transaction.get("credit"))
        transaction_merchant = MerchantExtractor.extract_merchant(transaction.get("merchant") or transaction.get("name") or transaction.get("description"))
        transaction_direction = DirectionUtils.detect_direction(debit, credit, transaction_type)

        candidate_groups = self._group_by_classification(examples)
        ranked_candidates: list[RankedCandidate] = []

        for classification_key, group in candidate_groups.items():
            frequency = len(group)
            average_similarity = sum(item.similarity for item in group) / frequency
            maximum_similarity = max(item.similarity for item in group)
            merchant_match_count = self._merchant_match_count(transaction_merchant, transaction_description, group)
            description_overlap = self._text_overlap(transaction_description, [item.example.description for item in group])
            debit_credit_match = self._debit_credit_match(debit, credit, group)
            direction_match = self._direction_match(transaction_direction, group)
            score = self._calculate_score(
                average_similarity=average_similarity,
                maximum_similarity=maximum_similarity,
                frequency=frequency,
                merchant_match_ratio=merchant_match_count / frequency,
                debit_credit_match=debit_credit_match,
                description_overlap=description_overlap,
                direction_match=direction_match,
            )
            example = group[0].example
            ranked_candidates.append(
                RankedCandidate(
                    classification_code=example.item_split_account or example.name or example.classification_name or "",
                    classification_name=example.classification_name,
                    classification_type=example.classification_type,
                    average_similarity=average_similarity,
                    maximum_similarity=maximum_similarity,
                    frequency=frequency,
                    merchant_match_count=merchant_match_count,
                    debit_credit_match=debit_credit_match,
                    description_overlap=description_overlap,
                    direction_match=direction_match,
                    score=score,
                )
            )

        ranked_candidates.sort(key=lambda candidate: (-candidate.score, -candidate.maximum_similarity, -candidate.average_similarity, -candidate.frequency))
        top_candidates = ranked_candidates[:top_n]

        return [
            {
                "classification_code": candidate.classification_code,
                "classification_name": candidate.classification_name,
                "classification_type": candidate.classification_type,
                "confidence": int(round(candidate.score * 100)),
                "method": "Retrieval",
                "score": round(candidate.score, 4),
                "average_similarity": round(candidate.average_similarity, 4),
                "maximum_similarity": round(candidate.maximum_similarity, 4),
                "frequency": candidate.frequency,
                "merchant_match_count": candidate.merchant_match_count,
                "debit_credit_match": round(candidate.debit_credit_match, 4),
                "description_overlap": round(candidate.description_overlap, 4),
                "direction_match": round(candidate.direction_match, 4),
            }
            for candidate in top_candidates
        ]

    def _group_by_classification(
        self,
        examples: list[HistoricalExampleSimilarity],
    ) -> dict[str, list[HistoricalExampleSimilarity]]:
        groups: dict[str, list[HistoricalExampleSimilarity]] = {}
        for example in examples:
            classification_code = example.example.item_split_account or example.example.name or example.example.classification_name or ""
            key = f"{classification_code}|{example.example.classification_name}|{example.example.classification_type}"
            groups.setdefault(key, []).append(example)
        return groups

    def _text_overlap(self, transaction_description: str, historical_descriptions: list[str]) -> float:
        if not transaction_description:
            return 0.0

        transaction_tokens = set(self._normalize_tokens(transaction_description))
        historical_tokens = set(token for description in historical_descriptions for token in self._normalize_tokens(description))
        if not transaction_tokens or not historical_tokens:
            return 0.0

        overlap = transaction_tokens & historical_tokens
        return len(overlap) / max(len(transaction_tokens), 1)

    def _merchant_match_count(self, transaction_merchant: str, transaction_description: str, examples: list[HistoricalExampleSimilarity]) -> int:
        normalized_merchant = MerchantExtractor.extract_merchant(transaction_merchant)
        if not normalized_merchant and transaction_description:
            normalized_merchant = MerchantExtractor.extract_merchant(transaction_description)
        if not normalized_merchant:
            return 0

        merchant_matches = 0
        for item in examples:
            historical_merchant = MerchantExtractor.extract_merchant(item.example.name)
            if historical_merchant and historical_merchant == normalized_merchant:
                merchant_matches += 1
        return merchant_matches

    def _debit_credit_match(
        self,
        debit: float | None,
        credit: float | None,
        examples: list[HistoricalExampleSimilarity],
    ) -> float:
        if debit is None and credit is None:
            return 0.0

        matches = 0
        for item in examples:
            if debit is not None and item.example.debit is not None and abs(debit - item.example.debit) < 0.01:
                matches += 1
            if credit is not None and item.example.credit is not None and abs(credit - item.example.credit) < 0.01:
                matches += 1
        return matches / max(len(examples), 1)

    def _direction_match(self, transaction_direction: str, examples: list[HistoricalExampleSimilarity]) -> float:
        if not transaction_direction or transaction_direction == "Unknown":
            return 0.0

        matches = 0
        for item in examples:
            historical_direction = DirectionUtils.detect_direction(item.example.debit, item.example.credit, item.example.transaction_type)
            if historical_direction == transaction_direction:
                matches += 1
        return matches / max(len(examples), 1)

    def _calculate_score(
        self,
        average_similarity: float,
        maximum_similarity: float,
        frequency: int,
        merchant_match_ratio: float,
        debit_credit_match: float,
        description_overlap: float,
        direction_match: float,
    ) -> float:
        frequency_factor = min(frequency / 20.0, 1.0)
        score = (
            0.20 * average_similarity +
            0.15 * maximum_similarity +
            0.10 * frequency_factor +
            0.25 * merchant_match_ratio +
            0.15 * debit_credit_match +
            0.05 * description_overlap +
            0.10 * direction_match
        )
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        normalized = text.lower()
        return [token for token in normalized.replace("/", " ").split() if token]

    @staticmethod
    def _normalize_amount(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
