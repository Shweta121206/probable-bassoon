from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.classification.example_loader import ExampleLoader, HistoricalExample
from app.memory.embedding_builder import EmbeddingBuilder
from app.memory.faiss_index import FaissIndex
from app.memory.similarity_search import SimilaritySearch

logger = logging.getLogger(__name__)

HISTORICAL_INDEX_PATH = Path("historical_index.faiss")
HISTORICAL_METADATA_PATH = Path("historical_metadata.pkl")


class MemoryManager:
    """Manage the historical transaction memory lifecycle."""

    def __init__(
        self,
        example_path: str | Path | None = None,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.example_loader = ExampleLoader(path=example_path)
        self.builder = EmbeddingBuilder()
        self.index = FaissIndex(index_path=index_path, metadata_path=metadata_path)
        self.similarity_search = SimilaritySearch(
            examples=self.example_loader.get_examples(),
            builder=self.builder,
            index=self.index,
        )

    def build_memory(self) -> None:
        if self.index.is_ready():
            logger.info("Historical memory index already available")
            return

        examples = self.example_loader.get_examples()
        if not examples:
            logger.warning("No historical examples found to build memory")
            return

        descriptions = [example.description for example in examples]
        embeddings = self.builder.build_embeddings(descriptions)
        metadata = [self._example_metadata(example) for example in examples]
        self.index.build_index(embeddings, metadata)
        logger.info("Built historical memory index with %d examples", len(descriptions))

    def get_similar_examples(self, description: str, top_k: int = 10) -> list[HistoricalExample]:
        similarities = self.similarity_search.get_similar_examples(description, top_k=top_k)
        return [item.example for item in similarities]

    @staticmethod
    def _example_metadata(example: HistoricalExample) -> dict[str, Any]:
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
