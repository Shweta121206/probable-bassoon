from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingBuilder:
    """Build sentence embeddings for historical transaction descriptions."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer | None:
        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers is not installed; using fallback embeddings"
            )
            return None

        try:
            model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model %s", self.model_name)
            return model
        except Exception as exc:
            logger.warning(
                "Failed to load sentence-transformers model %s: %s; using fallback embeddings",
                self.model_name,
                exc,
            )
            return None

    def build_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        logger.info("Building embeddings for %d historical examples", len(texts))
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=False)
            return [list(map(float, embedding)) for embedding in embeddings]

        return [self._simple_embedding(text) for text in texts]

    def build_embedding(self, text: str) -> list[float]:
        return self.build_embeddings([text])[0]

    @staticmethod
    def _simple_embedding(text: str) -> list[float]:
        normalized = text.lower().strip()
        vector = [0.0] * 32
        for index, token in enumerate(normalized.split()):
            value = sum(ord(char) for char in token) / 0xFFFF
            for i in range(len(vector)):
                vector[i] += value * (1.0 / (index + 1)) * ((i + 1) / len(vector))
        return [float(v) for v in vector]
