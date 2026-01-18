# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import asyncio
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.matchmaker import MatchStrategy
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        # Return a dummy vector of 1536 dims
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    async def extract(self, text: str) -> List[str]:
        return ["Project:Apollo", "User:Alice"]


@pytest.mark.asyncio
async def test_add_thought_flow() -> None:
    """Test the full ingestion flow with mocks."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    thought = await archive.add_thought(
        prompt="Who is Alice?",
        response="Alice is on Project Apollo.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )

    # Wait for background processing
    if archive._background_tasks:
        await asyncio.gather(*archive._background_tasks)

    # Verify Vector Store
    assert len(v_store.thoughts) == 1
    stored_thought = v_store.thoughts[0]
    assert stored_thought.id == thought.id
    assert stored_thought.vector == [0.1] * 1536
    assert stored_thought.entities == ["Project:Apollo", "User:Alice"]

    # Verify Graph Store
    # Should have nodes: Thought:ID, Project:Apollo, User:Alice
    assert g_store.graph.has_node(f"Thought:{thought.id}")
    assert g_store.graph.has_node("Project:Apollo")
    assert g_store.graph.has_node("User:Alice")

    # Verify Edges
    # Project:Apollo -> Thought:ID
    assert g_store.graph.has_edge("Project:Apollo", f"Thought:{thought.id}")


@pytest.mark.asyncio
async def test_add_thought_no_extractor() -> None:
    """Test ingestion without an entity extractor."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()

    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    thought = await archive.add_thought(
        prompt="Test",
        response="Test",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )

    # Vector store has it
    assert len(v_store.thoughts) == 1
    # Entities empty
    assert thought.entities == []
    # Graph store is NOT empty now due to synchronous structural linking (User, Thought, Scope)
    # Expect: User:user_123, Thought:<id>
    # Since Scope is USER and scope_id is user_123, Scope entity is also User:user_123
    assert g_store.graph.has_node("User:user_123")
    assert g_store.graph.has_node(f"Thought:{thought.id}")


@pytest.mark.asyncio
async def test_extraction_failure_graceful() -> None:
    """Test that extraction failure doesn't crash ingestion."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()

    # Extractor that raises exception
    extractor = Mock()
    extractor.extract = AsyncMock(side_effect=Exception("NLP Error"))

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Should not raise
    thought = await archive.add_thought(
        prompt="Test",
        response="Test",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_id="user_123",
    )

    assert len(v_store.thoughts) == 1
    assert thought.entities == []


@pytest.mark.asyncio
async def test_process_entities_no_extractor() -> None:
    """Test that process_entities does nothing if no extractor is set."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # Should just return without error
    await archive.process_entities(Mock(), "text")
    assert len(g_store.graph.nodes) == 0


@pytest.mark.asyncio
async def test_retrieve_flow() -> None:
    """Test the full retrieval flow."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    # No extractor needed for this test, we construct thoughts manually or simple

    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # Setup thoughts
    # 1. User thought (Accessible)
    t1 = await archive.add_thought("q1", "r1", MemoryScope.USER, "user_123", "user_123")
    # 2. Other User thought (Inaccessible)
    t2 = await archive.add_thought("q2", "r2", MemoryScope.USER, "user_456", "user_456")
    # 3. Project thought (Accessible, Boosted)
    t3 = await archive.add_thought("q3", "r3", MemoryScope.PROJECT, "apollo", "user_123")
    # Manually add entities to t3 to simulate boosting
    t3.entities = ["Project:apollo"]

    # 4. Old thought (Decayed)
    await archive.add_thought("q4", "r4", MemoryScope.USER, "user_123", "user_123")
    # Set t4 to be old manually (since created_at is set in add_thought)
    # Assuming we can modify it (it's in memory)
    # Actually, TemporalRanker handles decay.
    # Let's not mess with datetime mocking for now, just verify filtering
    # and boosting relative to each other if scores equal.
    # Since MockEmbedder returns constant [0.1...], all vector scores will be 1.0 (identical vectors).

    context = UserContext(user_id="user_123", project_ids=["apollo"])

    # Execute Retrieve
    results = await archive.retrieve("query", context, limit=10)

    # Expected:
    # t2 should be filtered out (wrong user).
    # t1, t3, t4 remain.
    # t3 should be boosted because "Project:apollo" is in context.project_ids.
    # t1 and t4 are recent USER thoughts.

    # Verify t2 is gone
    ids = [t.id for t, s, _ in results]
    assert t2.id not in ids

    # Verify t3 is top ranked (boosted)
    # Since all have vector score 1.0 (identical mock vectors),
    # t3 gets * 1.1 -> 1.1
    # t1 gets * 1.0 -> 1.0
    assert results[0][0].id == t3.id
    assert results[0][1] > 1.0

    # Verify t1 is present
    assert t1.id in ids


@pytest.mark.asyncio
async def test_retrieve_empty() -> None:
    """Test retrieving from empty store."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    context = UserContext(user_id="user_123")
    results = await archive.retrieve("test", context)
    assert results == []


@pytest.mark.asyncio
async def test_smart_lookup_exact_hit() -> None:
    """Test Smart Lookup returning EXACT_HIT."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    await archive.add_thought("q", "r", MemoryScope.USER, "user_123", "user_123")
    context = UserContext(user_id="user_123")

    # With identical vector, score is 1.0. Default threshold is 0.99.
    result = await archive.smart_lookup("q", context)

    assert result.strategy == MatchStrategy.EXACT_HIT
    assert result.content["source"] == "cache_hit"
    assert result.thought is not None


@pytest.mark.asyncio
async def test_smart_lookup_semantic_hint() -> None:
    """Test Smart Lookup returning SEMANTIC_HINT."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    await archive.add_thought("q", "trace", MemoryScope.USER, "user_123", "user_123")
    context = UserContext(user_id="user_123")

    # Force threshold higher than 1.0 to fail exact match (not possible naturally unless boosted)
    # But since MockEmbedder gives 1.0, and decay/boost might alter it.
    # Let's set exact_threshold to 1.1 (unreachable without boost) and hint to 0.9.
    result = await archive.smart_lookup("q", context, exact_threshold=1.1, hint_threshold=0.9)

    assert result.strategy == MatchStrategy.SEMANTIC_HINT
    assert "hint" in result.content
    assert result.content["source"] == "semantic_hint"


@pytest.mark.asyncio
async def test_smart_lookup_standard() -> None:
    """Test Smart Lookup returning STANDARD_RETRIEVAL."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    await archive.add_thought("q", "r", MemoryScope.USER, "user_123", "user_123")
    context = UserContext(user_id="user_123")

    # Set both thresholds very high
    result = await archive.smart_lookup("q", context, exact_threshold=2.0, hint_threshold=2.0)

    assert result.strategy == MatchStrategy.STANDARD_RETRIEVAL
    assert "top_thoughts" in result.content


@pytest.mark.asyncio
async def test_smart_lookup_no_results() -> None:
    """Test Smart Lookup with no results."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)
    context = UserContext(user_id="user_123")

    result = await archive.smart_lookup("q", context)

    assert result.strategy == MatchStrategy.STANDARD_RETRIEVAL
    assert result.content["message"] == "No relevant memories found."


@pytest.mark.asyncio
async def test_define_entity_relationship() -> None:
    """Test explicitly defining entity relationships for hierarchy."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    # Define a hierarchy link
    archive.define_entity_relationship(
        source="Project:Apollo", target="Department:RnD", relation=GraphEdgeType.BELONGS_TO
    )

    # Verify in GraphStore
    related = g_store.get_related_entities("Project:Apollo", GraphEdgeType.BELONGS_TO, direction="outgoing")
    assert len(related) == 1
    assert related[0][0] == "Department:RnD"
