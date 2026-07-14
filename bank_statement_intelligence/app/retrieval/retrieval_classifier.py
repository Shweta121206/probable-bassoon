from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.classification.example_loader import ExampleLoader, HistoricalExample, HistoricalExampleSimilarity
from app.memory.embedding_builder import EmbeddingBuilder
from app.memory.faiss_index import FaissIndex
from app.retrieval.candidate_ranker import CandidateRanker
from app.retrieval.confidence_engine import ConfidenceEngine

logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 20
INPUT_FILE = Path("output/transactions.json")
OUTPUT_FILE = Path("output/classified_transactions.json")


class RetrievalClassifier:
    """Classify transactions using retrieval over historical accounting examples."""

    def __init__(self, historical_examples_path: str | Path | None = None) -> None:
        self.example_loader = ExampleLoader(path=historical_examples_path)
        self.examples = self.example_loader.get_examples()
        self.embedding_builder = EmbeddingBuilder()
        self.index = FaissIndex()
        self.top_k = DEFAULT_TOP_K
        self.candidate_ranker = CandidateRanker()
        self.confidence_engine = ConfidenceEngine()

        if not self.index.is_ready():
            self._build_memory_index()
        
        #temp
        print("=" * 50)
        print("Examples loaded:", len(self.examples))
        print("Index ready:", self.index.is_ready())
        print("Index metadata:", len(self.index.metadata))
        print("=" * 50)

    def _build_memory_index(self) -> None:
        if not self.examples:
            logger.warning("No historical examples available to build retrieval index")
            return

        example_texts = [self._build_example_text(example) for example in self.examples]
        embeddings = self.embedding_builder.build_embeddings(example_texts)
        metadata = [self._example_metadata(example) for example in self.examples]
        self.index.build_index(embeddings, metadata)

    def classify_transactions(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not transactions:
            return []

        if not self.examples or not self.index.is_ready():
            logger.warning("Retrieval classifier has no historical examples or index")
            return [self._empty_classification(transaction) for transaction in transactions]

        embeddings = self.embedding_builder.build_embeddings(
            [self._build_query_text(transaction) for transaction in transactions]
        )
        search_results = self.index.search(embeddings, top_k=self.top_k)

        classified: list[dict[str, Any]] = []
        for transaction, (indices, similarities) in zip(transactions, search_results):
            top_examples = self._retrieve_examples(indices, similarities)
            candidates = self.candidate_ranker.rank_candidates(
                transaction=transaction,
                examples=top_examples,
            )
            top_candidates = candidates[:5]
            if top_candidates:
                best_candidate = dict(top_candidates[0])
            else:
                best_candidate = self._empty_classification(transaction)

            confidence = self.confidence_engine.calculate_confidence(best_candidate)
            status = self.confidence_engine.determine_status(confidence)
            explanation = self._build_ranking_explanation(best_candidate)

            best_candidate["classification_method"] = "Retrieval"
            best_candidate["confidence"] = confidence
            best_candidate["status"] = status
            best_candidate["top_candidates"] = top_candidates
            best_candidate["retrieved_neighbors"] = [
                {
                    "similarity": neighbor.similarity,
                    "classification_code": neighbor.example.item_split_account or neighbor.example.name,
                    "classification_name": neighbor.example.classification_name,
                    "classification_type": neighbor.example.classification_type,
                    "historical_description": neighbor.example.description,
                    "merchant": neighbor.example.name,
                    "transaction_type": neighbor.example.transaction_type,
                    "debit": neighbor.example.debit,
                    "credit": neighbor.example.credit,
                }
                for neighbor in top_examples
            ]
            best_candidate["confidence_calculation"] = self.confidence_engine.build_confidence_breakdown(best_candidate)
            best_candidate["ranking_explanation"] = explanation
            best_candidate["average_similarity"] = best_candidate.get("average_similarity", 0.0)
            best_candidate["maximum_similarity"] = best_candidate.get("maximum_similarity", 0.0)
            best_candidate["classification_frequency"] = best_candidate.get("frequency", 0)
            best_candidate["merchant_match_count"] = best_candidate.get("merchant_match_count", 0)
            classified.append({**transaction, **best_candidate})

        return classified

    def save_classified_transactions(self, records: list[dict[str, Any]], output_path: str | Path | None = None) -> Path:
        target = Path(output_path) if output_path is not None else OUTPUT_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info("Saved classified transactions to %s", target)
        return target

    def _retrieve_examples(self, indices: list[int], similarities: list[float]) -> list[HistoricalExampleSimilarity]:
        result: list[HistoricalExampleSimilarity] = []
        for index, similarity in zip(indices, similarities):
            if index < 0 or index >= len(self.examples):
                continue
            result.append(HistoricalExampleSimilarity(example=self.examples[index], similarity=float(similarity)))
        return result

    def _build_ranking_explanation(self, candidate: dict[str, Any]) -> str:
        return (
            f"Confidence is based on average similarity {candidate.get('average_similarity', 0.0):.4f}, "
            f"maximum similarity {candidate.get('maximum_similarity', 0.0):.4f}, "
            f"frequency {candidate.get('frequency', 0)}, "
            f"merchant matches {candidate.get('merchant_match_count', 0)}, "
            f"debit/credit match {candidate.get('debit_credit_match', 0.0):.4f}, "
            f"description overlap {candidate.get('description_overlap', 0.0):.4f}."
        )

    def _example_metadata(self, example: HistoricalExample) -> dict[str, Any]:
        return {
            "description": example.description,
            "debit": example.debit,
            "credit": example.credit,
            "merchant": example.name,
            "direction": self._transaction_direction(example.debit, example.credit),
            "classification_code": example.item_split_account or example.name or example.classification_name or "",
            "classification_name": example.classification_name,
            "classification_type": example.classification_type,
            "transaction_type": example.transaction_type,
        }

    def _build_example_text(self, example: HistoricalExample) -> str:
        return self._build_structured_text(
            merchant=example.name,
            description=example.description,
            debit=example.debit,
            credit=example.credit,
            transaction_type=example.transaction_type,
        )

    def _build_query_text(self, transaction: dict[str, Any]) -> str:
        return self._build_structured_text(
            merchant=transaction.get("merchant") or transaction.get("name"),
            description=transaction.get("description"),
            debit=transaction.get("debit"),
            credit=transaction.get("credit"),
            transaction_type=transaction.get("transaction_type"),
        )

    @staticmethod
    def _build_structured_text(
        merchant: Any,
        description: Any,
        debit: Any,
        credit: Any,
        transaction_type: Any,
    ) -> str:
        merchant_text = str(merchant or "").strip()
        description_text = str(description or "").strip()
        direction_text = RetrievalClassifier._transaction_direction(debit, credit)
        debit_text = "" if debit is None else str(debit)
        credit_text = "" if credit is None else str(credit)

        return "\n".join(
            section
            for section in [
                f"Merchant:", merchant_text,
                f"Description:", description_text,
                f"Direction:", direction_text,
                f"Debit:", debit_text,
                f"Credit:", credit_text,
                f"Transaction Type:", str(transaction_type or "").strip(),
            ]
            if section is not None and str(section).strip()
        )

    @staticmethod
    def _transaction_direction(debit: Any, credit: Any) -> str:
        if debit is not None and credit is None:
            return "Withdrawal"
        if credit is not None and debit is None:
            return "Deposit"
        return "Unknown"

    @staticmethod
    def _empty_classification(transaction: dict[str, Any]) -> dict[str, Any]:
        return {
            "classification_code": "",
            "classification_name": "",
            "classification_type": "",
            "confidence": 0,
            "status": "Low",
            "method": "Retrieval",
            "average_similarity": 0.0,
            "maximum_similarity": 0.0,
            "classification_frequency": 0,
            "merchant_match_count": 0,
            "debit_credit_match": 0.0,
            "description_overlap": 0.0,
            "confidence_calculation": {
                "average_similarity": 0.0,
                "maximum_similarity": 0.0,
                "frequency_factor": 0.0,
                "merchant_match_ratio": 0.0,
                "debit_credit_match": 0.0,
                "description_overlap": 0.0,
                "weighted_score": 0.0,
                "confidence": 0,
            },
            "ranking_explanation": "No retrieval candidates found.",
        }
