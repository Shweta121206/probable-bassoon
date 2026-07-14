from __future__ import annotations

import logging
from typing import Iterable

from app.classification.example_loader import ExampleLoader, HistoricalExample, HistoricalExampleSimilarity
from app.memory.embedding_builder import EmbeddingBuilder
from app.memory.faiss_index import FaissIndex

logger = logging.getLogger(__name__)
DEFAULT_TOP_K = 10


class SimilaritySearch:
    """Search historical examples by transaction description similarity."""

    def __init__(
        self,
        examples: list[HistoricalExample] | None = None,
        builder: EmbeddingBuilder | None = None,
        index: FaissIndex | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.examples = examples if examples is not None else ExampleLoader().get_examples()
        self.builder = builder or EmbeddingBuilder()
        self.index = index or FaissIndex()
        self.top_k = top_k

        if not self.index.is_ready():
            self.build_index()

    def build_index(self) -> None:
        if not self.examples:
            logger.warning("No historical examples available to build FAISS index")
            return

        descriptions = [example.description for example in self.examples]
        embeddings = self.builder.build_embeddings(descriptions)
        metadata = [self._example_metadata(example) for example in self.examples]
        self.index.build_index(embeddings, metadata)

    def get_similar_examples(
        self,
        description: str,
        top_k: int | None = None,
    ) -> list[HistoricalExampleSimilarity]:
        if not self.index.is_ready():
            self.build_index()

        if not self.examples or self.index is None or not self.index.is_ready():
            logger.warning("No memory index available to perform similarity search")
            return []

        search_k = top_k if top_k is not None else self.top_k
        if not description:
            return []

        embedding = self.builder.build_embedding(description)
        search_results = self.index.search([embedding], search_k)
        indices, similarities = search_results[0]

        examples: list[HistoricalExampleSimilarity] = []
        for item_index, similarity in zip(indices, similarities):
            if item_index < 0 or item_index >= len(self.examples):
                continue
            examples.append(
                HistoricalExampleSimilarity(
                    example=self.examples[item_index],
                    similarity=float(similarity),
                )
            )

        return examples

    @staticmethod
    def _example_metadata(example: HistoricalExample) -> dict[str, object]:
        return {
            "description": example.description,
            "debit": example.debit,
            "credit": example.credit,
            "name": example.name,
            "item_split_account": example.item_split_account,
            "transaction_type": example.transaction_type,
            "classification_name": example.classification_name,
            "classification_type": example.classification_type,
        }
