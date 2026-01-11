import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from coreason_archive.models import CachedThought
from coreason_archive.utils.logger import logger


class VectorStore:
    """
    An in-memory Vector Store using NumPy for cosine similarity.
    Persists data to disk as a JSON file.
    """

    def __init__(self) -> None:
        """Initialize an empty Vector Store."""
        self.thoughts: List[CachedThought] = []
        # cache the vectors as a numpy array for faster search
        # We'll rebuild this lazily or incrementally if needed,
        # but for MVP, rebuilding on add or search is acceptable logic.
        # To avoid complexity, we'll keep a list and convert to array on search.
        self._vectors: List[List[float]] = []

    def add(self, thought: CachedThought) -> None:
        """
        Adds a CachedThought to the store.

        Args:
            thought: The thought object to store.
        """
        self.thoughts.append(thought)
        self._vectors.append(thought.vector)
        logger.debug(f"Added thought {thought.id} to VectorStore.")

    def search(
        self, query_vector: List[float], limit: int = 10, min_score: float = 0.0
    ) -> List[Tuple[CachedThought, float]]:
        """
        Performs a cosine similarity search against stored thoughts.

        Args:
            query_vector: The embedding vector to search with.
            limit: Maximum number of results to return.
            min_score: Minimum similarity score (0.0 to 1.0) to include.

        Returns:
            A list of tuples (CachedThought, score), sorted by score descending.
        """
        if not self.thoughts:
            return []

        # Convert to numpy arrays
        # Shape: (N, D)
        candidates = np.array(self._vectors)
        # Shape: (D,)
        query = np.array(query_vector)

        # Norm calculation
        # axis=1 for candidates (norm of each row)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        query_norm = np.linalg.norm(query)

        # Avoid division by zero
        if query_norm == 0:
            logger.warning("Query vector has zero norm.")
            return []

        # Handle zero-norm candidates (rare for embeddings but possible in edge cases)
        # We replace 0 norms with 1 (or infinity) to avoid nan, resulting in 0 score.
        # simpler: just ignore division by zero warning or handle explicitly.
        candidate_norms[candidate_norms == 0] = 1e-10

        # Dot product
        # (N, D) dot (D,) -> (N,)
        dot_products = np.dot(candidates, query)

        # Cosine similarity
        scores = dot_products / (candidate_norms * query_norm)

        # Zip with thoughts
        results: List[Tuple[CachedThought, float]] = []
        for thought, score in zip(self.thoughts, scores, strict=False):
            # float(score) converts numpy float to python float
            s = float(score)
            if s >= min_score:
                results.append((thought, s))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def save(self, filepath: Path) -> None:
        """
        Persists the list of thoughts to a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        # Serialize list of models
        # Pydantic 2 syntax for list serialization: TypeAdapter or manual list
        # Simple manual list dump
        data = [json.loads(t.model_dump_json()) for t in self.thoughts]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"VectorStore saved {len(self.thoughts)} thoughts to {filepath}")

    def load(self, filepath: Path) -> None:
        """
        Loads thoughts from a JSON file.

        Args:
            filepath: Path to the JSON file.
        """
        if not filepath.exists():
            logger.warning(f"VectorStore file {filepath} not found. Starting empty.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.thoughts = [CachedThought.model_validate(item) for item in data]
        # Rebuild vector cache
        self._vectors = [t.vector for t in self.thoughts]

        logger.info(f"VectorStore loaded {len(self.thoughts)} thoughts from {filepath}")
