from __future__ import annotations

from typing import Any


class ConfidenceEngine:
    """Calculate confidence and status for retrieval-based classification."""

    def calculate_confidence(self, candidate: dict[str, Any]) -> int:
        if candidate is None:
            return 0

        average_similarity = float(candidate.get("average_similarity", 0.0))
        maximum_similarity = float(candidate.get("maximum_similarity", 0.0))
        frequency = int(candidate.get("frequency", 0))
        merchant_match_count = int(candidate.get("merchant_match_count", 0))
        debit_credit_match = float(candidate.get("debit_credit_match", 0.0))
        description_overlap = float(candidate.get("description_overlap", 0.0))
        direction_match = float(candidate.get("direction_match", 0.0))

        frequency_factor = min(frequency / 20.0, 1.0)
        merchant_match_ratio = min(merchant_match_count / max(frequency, 1), 1.0)

        raw_score = (
            0.20 * average_similarity +
            0.15 * maximum_similarity +
            0.10 * frequency_factor +
            0.25 * merchant_match_ratio +
            0.15 * debit_credit_match +
            0.05 * description_overlap +
            0.10 * direction_match
        )
        confidence = int(round(min(max(raw_score, 0.0), 1.0) * 100))
        return confidence

    def determine_status(self, confidence: int) -> str:
        if confidence >= 95:
            return "Certain"
        if confidence >= 90:
            return "Very High"
        if confidence >= 80:
            return "High"
        if confidence >= 70:
            return "Medium"
        return "Low"

    def build_confidence_breakdown(self, candidate: dict[str, Any]) -> dict[str, Any]:
        average_similarity = float(candidate.get("average_similarity", 0.0))
        maximum_similarity = float(candidate.get("maximum_similarity", 0.0))
        frequency = int(candidate.get("frequency", 0))
        merchant_match_count = int(candidate.get("merchant_match_count", 0))
        debit_credit_match = float(candidate.get("debit_credit_match", 0.0))
        description_overlap = float(candidate.get("description_overlap", 0.0))
        direction_match = float(candidate.get("direction_match", 0.0))

        frequency_factor = min(frequency / 20.0, 1.0)
        merchant_match_ratio = min(merchant_match_count / max(frequency, 1), 1.0)

        weighted_score = (
            0.20 * average_similarity +
            0.15 * maximum_similarity +
            0.10 * frequency_factor +
            0.25 * merchant_match_ratio +
            0.15 * debit_credit_match +
            0.05 * description_overlap +
            0.10 * direction_match
        )
        confidence = int(round(min(max(weighted_score, 0.0), 1.0) * 100))

        return {
            "average_similarity": round(average_similarity, 4),
            "maximum_similarity": round(maximum_similarity, 4),
            "frequency_factor": round(frequency_factor, 4),
            "merchant_match_ratio": round(merchant_match_ratio, 4),
            "debit_credit_match": round(debit_credit_match, 4),
            "description_overlap": round(description_overlap, 4),
            "direction_match": round(direction_match, 4),
            "weighted_score": round(weighted_score, 4),
            "confidence": confidence,
        }
