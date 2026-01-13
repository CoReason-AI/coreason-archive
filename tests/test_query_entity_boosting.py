from typing import List
from unittest.mock import AsyncMock

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.models import MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    """Simple mock that returns fixed vectors."""

    def embed(self, text: str) -> List[float]:
        # Return a dummy vector of 1536 dims
        # Using 0.1 allows us to have 'perfect' similarity if query also yields 0.1
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    """Mock extractor that returns predefined entities based on input text."""

    async def extract(self, text: str) -> List[str]:
        if "Apollo" in text:
            return ["Project:Apollo"]
        if "Alice" in text:
            return ["User:Alice"]
        if "Drug Z" in text:
            return ["Drug:Z"]
        return []


@pytest.mark.asyncio
async def test_query_entity_boosting_hit() -> None:
    """
    Test Case 1: Verify that a thought containing 'Drug:Z' receives a score boost
    when the search query also yields 'Drug:Z' via extraction.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Add a thought about Drug Z
    # Scope is USER, unrelated to any active project, to isolate query-boosting.
    thought_z = await archive.add_thought(
        prompt="Effects of Drug Z?",
        response="Drug Z causes drowsiness.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )
    # Manually ensure entities are set (since add_thought uses background task)
    thought_z.entities = ["Drug:Z"]

    # Add a control thought (identical vector, but different entities)
    thought_control = await archive.add_thought(
        prompt="Effects of something else",
        response="No effects.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )
    thought_control.entities = ["Drug:Y"]

    # 2. Query for "Drug Z"
    context = UserContext(user_id="user_123")
    # query contains "Drug Z", so MockEntityExtractor returns ["Drug:Z"]
    query = "What about Drug Z?"

    results = await archive.retrieve(
        query=query,
        context=context,
        limit=10,
        graph_boost_factor=2.0,  # High boost to make it obvious
    )

    # 3. Verification
    # Both thoughts have identical vector score (1.0).
    # thought_z should be boosted by 2.0 because "Drug:Z" matches.
    # thought_control should NOT be boosted.

    assert len(results) == 2
    top_thought, top_score, top_metadata = results[0]
    second_thought, second_score, second_metadata = results[1]

    assert top_thought.id == thought_z.id
    assert top_metadata.get("is_boosted") is True
    assert top_score > second_score
    # Approx 2x difference (ignoring slight temporal decay diff if any)
    assert top_score >= 1.9  # 1.0 * 2.0 * decay (~1.0)


@pytest.mark.asyncio
async def test_query_entity_boosting_no_hit() -> None:
    """
    Test Case 2: Verify that no boost is applied if the query entity
    does not match the thought's entities.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Thought has "Project:Apollo"
    thought = await archive.add_thought(
        prompt="Apollo info",
        response="Apollo is a rocket.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )
    thought.entities = ["Project:Apollo"]

    # Query has "Drug Z" (via mock)
    query = "Tell me about Drug Z"
    context = UserContext(user_id="user_123")

    results = await archive.retrieve(query, context, graph_boost_factor=2.0)

    assert len(results) == 1
    t, score, meta = results[0]

    # Should NOT be boosted
    assert meta.get("is_boosted") is False
    # Score should be base score (approx 1.0 * decay)
    assert score < 1.1


@pytest.mark.asyncio
async def test_retrieve_graceful_extractor_failure() -> None:
    """
    Test Case 3: Ensure graceful handling if entity_extractor fails
    during retrieval.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()

    # Extractor that fails
    extractor = AsyncMock()
    extractor.extract.side_effect = Exception("NLP Service Down")

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    await archive.add_thought("q", "r", MemoryScope.USER, "user_123", "user_123")
    context = UserContext(user_id="user_123")

    # Should not raise exception
    results = await archive.retrieve("query", context)

    assert len(results) == 1
    # Check logs if possible, but mainly ensure no crash
