from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.classification.chart_loader import ChartLoader
from app.classification.example_loader import ExampleLoader, HistoricalExampleSimilarity
from app.classification.gemini_classifier import GeminiClassifier, ClassificationResult
from app.classification.prompt_builder import PromptContext, build_classification_prompt
from app.memory.similarity_search import SimilaritySearch

logger = logging.getLogger(__name__)

CLASSIFIED_TRANSACTIONS_PATH = Path("output/classified_transactions.json")


class TransactionClassifier:
    """Classify transaction records using Gemini with historical memory."""

    def __init__(self, api_key: str | None = None) -> None:
        self._configure_logging()
        self.chart_loader = ChartLoader()
        self.example_loader = ExampleLoader()
        self.gemini = GeminiClassifier(api_key=api_key)
        self.similarity_search = SimilaritySearch(self.example_loader.get_examples())

    def _configure_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / "classification.log"

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")) == file_path
            for handler in logger.handlers
        ):
            handler = logging.FileHandler(file_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
            )
            logger.addHandler(handler)

    def classify_transactions(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        classified: list[dict[str, Any]] = []
        for transaction in transactions:
            try:
                record = self._classify_transaction(transaction)
                classified.append(record)
            except Exception as exc:
                logger.warning("Failed to classify transaction: %s", exc)
                classified.append({**transaction, "classification_code": "", "classification_name": "", "classification_type": "", "classification_confidence": 0, "classification_reason": "Classification failed"})
        return classified

    def _classify_transaction(self, transaction: dict[str, Any]) -> dict[str, Any]:
        amount = self._normalize_amount(transaction.get("amount"))
        debit = self._normalize_amount(transaction.get("debit"))
        credit = self._normalize_amount(transaction.get("credit"))

        if debit is None and credit is None and amount is not None:
            if amount < 0:
                debit = abs(amount)
                credit = 0.0
            else:
                debit = 0.0
                credit = amount

        prompt_context = PromptContext(
            transaction_date=str(transaction.get("date", "")),
            description=str(transaction.get("description", "")),
            debit=debit,
            credit=credit,
            balance=transaction.get("balance"),
            transaction_type=str(transaction.get("transaction_type", "")),
        )

        similar_examples = self.similarity_search.get_similar_examples(prompt_context.description)
        prompt = build_classification_prompt(
            transaction=prompt_context,
            similar_examples=similar_examples,
            accounts=self.chart_loader.get_accounts(),
        )

        classification = self.gemini.classify_transaction(prompt)
        logger.info(
            "Classified transaction %s with %s (%s) confidence=%d",
            prompt_context.description,
            classification.classification_name,
            classification.classification_code,
            classification.confidence,
        )

        result = {
            **transaction,
            "debit": prompt_context.debit,
            "credit": prompt_context.credit,
            "classification_code": classification.classification_code,
            "classification_name": classification.classification_name,
            "classification_type": classification.classification_type,
            "classification_confidence": classification.confidence,
            "classification_reason": classification.reason,
        }

        self._log_classification(prompt_context, similar_examples, classification)
        return result

    def _normalize_amount(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            if isinstance(value, str):
                return float(value.replace("$", "").replace(",", ""))
            return float(value)
        except (TypeError, ValueError):
            return None

    def _log_classification(
        self,
        transaction: PromptContext,
        similar_examples: list[HistoricalExampleSimilarity],
        classification: ClassificationResult,
    ) -> None:
        lines = [
            f"Transaction: {transaction.description}",
            f"Date: {transaction.transaction_date}",
            f"Debit: {transaction.debit}",
            f"Credit: {transaction.credit}",
            f"Balance: {transaction.balance}",
            f"Transaction Type: {transaction.transaction_type}",
            "Retrieved Similar Examples:",
        ]
        for example in similar_examples:
            lines.append(
                f"Similarity={example.similarity:.4f} | Description={example.example.description} | Classification={example.example.classification_name} ({example.example.classification_type})"
            )
        lines.append("Gemini Classification:")
        lines.append(json.dumps({
            "classification_code": classification.classification_code,
            "classification_name": classification.classification_name,
            "classification_type": classification.classification_type,
            "confidence": classification.confidence,
            "reason": classification.reason,
        }))
        logger.info("\n" + "\n".join(lines))
