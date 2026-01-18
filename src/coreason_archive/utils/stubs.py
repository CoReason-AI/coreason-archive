# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import hashlib
from typing import List

from coreason_archive.interfaces import Embedder


class StubEmbedder(Embedder):
    """
    A deterministic stub embedder for local testing and CLI usage.
    Generates a pseudo-random vector based on the input text hash.
    """

    def __init__(self, dim: int = 1536) -> None:
        """
        Initialize the StubEmbedder.

        Args:
            dim: The dimension of the generated vectors.
        """
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """
        Generates a deterministic vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        # Create a seed from the text hash
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)

        # Use a simple Linear Congruential Generator for determinism
        vector = []
        current = seed
        for _ in range(self.dim):
            # Parameters from Numerical Recipes
            current = (current * 1664525 + 1013904223) % 2**32
            # Normalize to [-1, 1] approximately (using float division)
            val = (current / 2**32) * 2 - 1
            vector.append(val)

        # Normalize the vector to unit length (L2 norm)
        norm = sum(x * x for x in vector) ** 0.5
        if norm > 0:
            vector = [x / norm for x in vector]
        else:  # pragma: no cover
            vector = [0.0] * self.dim

        return vector
