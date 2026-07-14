from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np

logger = logging.getLogger(__name__)
INDEX_PATH = Path("historical_index.faiss")
METADATA_PATH = Path("historical_metadata.pkl")


class FaissIndex:
    """Manage a FAISS index for historical transaction embeddings."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.index_path = Path(index_path) if index_path is not None else INDEX_PATH
        self.metadata_path = Path(metadata_path) if metadata_path is not None else METADATA_PATH
        self.index: Any | None = None
        self.metadata: list[dict[str, Any]] = []
        self._use_faiss = faiss is not None
        self._load_index()

    def _load_index(self) -> None:
        #temp
        print("INDEX PATH:", self.index_path)
        print("INDEX EXISTS:", self.index_path.exists())
        print("METADATA PATH:", self.metadata_path)
        print("METADATA EXISTS:", self.metadata_path.exists())

        if self._use_faiss and self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "rb") as handle:
                    self.metadata = pickle.load(handle)
                logger.info("Loaded FAISS index from %s", self.index_path)
                return
            #temp
            # except Exception as exc:
            #    logger.warning("Failed to load FAISS index: %s", exc)
            except Exception as exc:
                import traceback

                print("=" * 60)
                print("FAISS LOAD FAILED")
                print(type(exc))
                print(exc)
                traceback.print_exc()
                print("=" * 60)

            self.index = None
            self.metadata = []

        self.index = None
        self.metadata = []

    def build_index(self, embeddings: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        if self.index is not None:
            logger.info("FAISS index already exists; skipping rebuild")
            return

        if not embeddings:
            raise ValueError("No embeddings provided to build FAISS index")

        vectors = self._to_float32_vectors(embeddings)
        normalized_vectors = self._normalize_vectors(vectors)
        if self._use_faiss:
            dim = normalized_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(normalized_vectors)
            #temp self._save_index()
            self.metadata = list(metadata)
        else:
            self.index = normalized_vectors

        #temp self.metadata = list(metadata)
        self._save_index()
        logger.info(
            "Built historical embedding index with %d vectors using %s",
            normalized_vectors.shape[0],
            "faiss" if self._use_faiss else "fallback search",
        )

    def search(self, embeddings: list[list[float]], top_k: int = 10) -> list[tuple[list[int], list[float]]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")

        vectors = self._to_float32_vectors(embeddings)
        normalized_vectors = self._normalize_vectors(vectors)
        if self._use_faiss:
            distances, indices = self.index.search(normalized_vectors, top_k)
            return [
                (list(map(int, row_indices)), list(map(float, row_distances)))
                for row_indices, row_distances in zip(indices, distances)
            ]

        if not isinstance(self.index, np.ndarray):
            raise RuntimeError("Fallback index is corrupted")

        similarity_matrix = self._cosine_similarity(normalized_vectors, self.index)
        results: list[tuple[list[int], list[float]]] = []
        for row in similarity_matrix:
            top_indices = list(np.argsort(row)[::-1][:top_k])
            top_similarities = [float(row[idx]) for idx in top_indices]
            results.append((top_indices, top_similarities))
        return results

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        return vectors / safe_norms

    def _save_index(self) -> None:
        if not self._use_faiss:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as handle:
            pickle.dump(self.metadata, handle)

    @staticmethod
    def _to_float32_vectors(embeddings: list[list[float]]) -> "np.ndarray":
        return np.array(embeddings, dtype="float32")

    @staticmethod
    def _cosine_similarity(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        if vectors_a.ndim == 1:
            vectors_a = vectors_a.reshape(1, -1)
        if vectors_b.ndim == 1:
            vectors_b = vectors_b.reshape(1, -1)

        a_norm = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(vectors_b, axis=1, keepdims=True)
        denom = np.maximum(a_norm * b_norm.T, 1e-12)
        return (vectors_a @ vectors_b.T) / denom

    def is_ready(self) -> bool:
        return (
        self.index is not None
        and len(self.metadata) > 0
        and self.index.ntotal == len(self.metadata)
        )

    def get_metadata(self, index: int) -> dict[str, Any]:
        try:
            return self.metadata[index]
        except IndexError as exc:
            raise IndexError("Metadata index out of range") from exc

    def clear(self) -> None:
        self.index = None
        self.metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        logger.info("Cleared FAISS index and metadata")
