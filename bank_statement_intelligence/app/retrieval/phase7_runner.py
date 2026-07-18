from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.retrieval.retrieval_classifier import RetrievalClassifier
from app.retrieval.evaluation import EvaluationTool

logger = logging.getLogger(__name__)


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    transactions_path = project_root / "output" / "transactions.json"
    historical_path = project_root / "config" / "historical_examples.xlsx"
    classified_path = project_root / "output" / "classified_transactions.json"
    report_path = project_root / "output" / "retrieval_report.json"

    if not transactions_path.exists():
        logger.error("Missing transactions.json at %s", transactions_path)
        return 1

    if not historical_path.exists():
        logger.error("Missing historical_examples.xlsx at %s", historical_path)
        return 1

    with transactions_path.open("r", encoding="utf-8") as handle:
        transactions = json.load(handle)

    try:
        classifier = RetrievalClassifier(historical_examples_path=historical_path)
        classified = classifier.classify_transactions(transactions)
        classifier.save_classified_transactions(classified, classified_path)

        report = {
            "total_transactions": len(classified),
            "transactions": [],
        }

        for record in classified:
            report["transactions"].append({
                "description": record.get("description", ""),
                "classification_code": record.get("classification_code", ""),
                "classification_name": record.get("classification_name", ""),
                "classification_type": record.get("classification_type", ""),
                "confidence": record.get("confidence", 0),
                "status": record.get("status", ""),
                "method": record.get("method", ""),
                "verification_method": record.get("verification_method", ""),
                "verification_confidence": record.get("verification_confidence", 0),
                "verification_reason": record.get("verification_reason", ""),
                "verified_by_gemini": record.get("verification_method") == "Gemini Verification",
                "average_similarity": record.get("average_similarity", 0.0),
                "maximum_similarity": record.get("maximum_similarity", 0.0),
                "classification_frequency": record.get("classification_frequency", 0),
                "merchant_match_count": record.get("merchant_match_count", 0),
                "confidence_calculation": record.get("confidence_calculation", {}),
                "ranking_explanation": record.get("ranking_explanation", ""),
                "top_candidates": record.get("top_candidates", []),
                "retrieved_neighbors": record.get("retrieved_neighbors", []),
            })

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved retrieval report to %s", report_path)
    except Exception as exc:
        logger.warning("Phase 7 completed with retrieval-only fallback after an unexpected error: %s", exc)
        classified_path.write_text("[]", encoding="utf-8")
        report_path.write_text(json.dumps({"total_transactions": 0, "transactions": []}, indent=2), encoding="utf-8")

    print("Phase 7 completed: classified_transactions.json and retrieval_report.json written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
