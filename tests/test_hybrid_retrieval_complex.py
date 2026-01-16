# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import List, Tuple
from uuid import uuid4

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    """Mock embedder providing controllable vector output."""

    def embed(self, text: str) -> List[float]:
        if "query" in text:
            return [1.0, 0.0]  # Query
        if "match" in text:
            return [0.9, 0.1]  # High Sim
        if "miss" in text:
            return [0.0, 1.0]  # Low Sim (Orthogonal)
        return [0.5, 0.5]  # Default


class MockEntityExtractor(EntityExtractor):
    """Mock extractor returning specific entities."""

    def __init__(self, entities: List[str] | None = None):
        self.entities = entities or []

    async def extract(self, text: str) -> List[str]:
        return self.entities


@pytest.fixture
def base_archive() -> Tuple[VectorStore, GraphStore, MockEmbedder]:
    v = VectorStore()
    g = GraphStore()
    e = MockEmbedder()
    return v, g, e


@pytest.mark.asyncio
async def test_orphaned_graph_node(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Edge Case: The Graph contains a 'Thought:UUID' node, but the thought is NOT in VectorStore.
    Expected: Retrieve should ignore it and not crash.
    """
    v_store, g_store, embedder = base_archive
    extractor = MockEntityExtractor(["Entity:Key"])
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Setup Graph with Orphaned Node
    orphan_id = str(uuid4())
    g_store.add_relationship("Entity:Key", f"Thought:{orphan_id}", GraphEdgeType.RELATED_TO)

    # 2. Add a valid thought (just to have something in store)
    await archive.add_thought("p", "r", MemoryScope.USER, "u1", "u1")

    # 3. Retrieve
    context = UserContext(user_id="u1")
    await archive.retrieve("query", context)

    # Should succeed. Results might contain the valid thought (if vector matches default).
    # Specifically, we verify no error was raised about missing ID.
    assert True


@pytest.mark.asyncio
async def test_security_filtering_on_graph_sourced_thought(
    base_archive: Tuple[VectorStore, GraphStore, MockEmbedder],
) -> None:
    """
    Security Test: A thought is found ONLY via Graph Sourcing (low vector sim),
    but the user does NOT have permission to see it.
    Expected: It is filtered out by FederationBroker.
    """
    v_store, g_store, embedder = base_archive
    extractor = MockEntityExtractor(["Entity:Secret"])
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Add "Secret" Thought
    # Use "miss" to ensure low vector similarity (so it depends on Graph Sourcing)
    secret_thought = await archive.add_thought("miss", "secret info", MemoryScope.PROJECT, "RestrictedProject", "admin")
    secret_thought.entities = ["Entity:Secret"]  # Manually link

    # Ensure Graph Link exists (since we skipped background extraction or manually set entities)
    g_store.add_relationship("Entity:Secret", f"Thought:{secret_thought.id}", GraphEdgeType.RELATED_TO)

    # 2. User Context: NO access to "RestrictedProject"
    context = UserContext(user_id="u1", project_ids=["PublicProject"])

    # 3. Retrieve
    # min_score=0.5 -> "miss" ([0,1] vs [1,0]) sim=0.0 -> Excluded by Vector Search
    # Graph Sourcing pulls it in via Entity:Secret.
    # Federation Filter should drop it.
    results = await archive.retrieve("query", context, min_score=0.5)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_duplicate_candidate_handling(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Complex Scenario: Thought is found by Vector Search AND Graph Sourcing.
    Expected: It appears only once in results.
    """
    v_store, g_store, embedder = base_archive
    extractor = MockEntityExtractor(["Entity:Dual"])
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Add Thought
    # "match" -> High Sim -> Found by Vector Search
    thought = await archive.add_thought("match", "content", MemoryScope.USER, "u1", "u1")
    thought.entities = ["Entity:Dual"]

    # Graph Link
    g_store.add_relationship("Entity:Dual", f"Thought:{thought.id}", GraphEdgeType.RELATED_TO)

    context = UserContext(user_id="u1")

    # 2. Retrieve
    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    # 3. Verify
    # Should be in results once
    matching_results = [r for r in results if r[0].id == thought.id]
    assert len(matching_results) == 1

    # Check score
    # Base sim approx 0.9 (match vs query). Boost 2.0.
    # Score should be ~1.8 (ignoring decay)
    assert matching_results[0][1] > 1.5
    assert matching_results[0][2]["is_boosted"] is True


@pytest.mark.asyncio
async def test_zero_entity_query_fallback(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Edge Case: No entities extracted, no active project.
    Expected: Standard Vector Search results.
    """
    v_store, g_store, embedder = base_archive
    extractor = MockEntityExtractor([])  # Returns empty list
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Add thoughts
    await archive.add_thought("match", "good", MemoryScope.USER, "u1", "u1")

    context = UserContext(user_id="u1")

    results = await archive.retrieve("query", context)

    assert len(results) > 0
    # Top result should be the match
    assert "match" in results[0][0].prompt_text
