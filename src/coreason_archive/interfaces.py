from typing import List, Protocol, runtime_checkable


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
