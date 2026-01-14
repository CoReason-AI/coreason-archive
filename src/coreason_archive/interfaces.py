# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import Any, Coroutine, List, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for generating vector embeddings from text.
    """

    def embed(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    """
    Protocol for extracting entities from text.
    """

    async def extract(self, text: str) -> List[str]:
        """
        Extracts entities from the given text asynchronously.

        Args:
            text: The text to analyze.

        Returns:
            A list of entity strings in 'Type:Value' format.
        """
        ...


@runtime_checkable
class TaskRunner(Protocol):
    """
    Protocol for executing background tasks.
    Allows decoupling the archive from specific execution environments (e.g., asyncio, FastAPI).
    """

    def run(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Schedules a coroutine for execution in the background.

        Args:
            coro: The coroutine to execute.
        """
        ...
