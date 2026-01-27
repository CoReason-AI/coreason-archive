# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import List
from unittest.mock import AsyncMock

import pytest
from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.models import GraphEdgeType, MemoryScope
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
    user_ctx = UserContext(user_id="user_123", email="test@example.com")
    thought_z = await archive.add_thought(
        prompt="Effects of Drug Z?",
        response="Drug Z causes drowsiness.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_context=user_ctx,
    )
    # Manually ensure entities are set (since add_thought uses background task)
    thought_z.entities = ["Drug:Z"]

    # Add a control thought (identical vector, but different entities)
    thought_control = await archive.add_thought(
        prompt="Effects of something else",
        response="No effects.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_context=user_ctx,
    )
    thought_control.entities = ["Drug:Y"]

    # 2. Query for "Drug Z"
    context = UserContext(user_id="user_123", email="test@example.com")
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
    user_ctx = UserContext(user_id="user_123", email="test@example.com")
    thought = await archive.add_thought(
        prompt="Apollo info",
        response="Apollo is a rocket.",
        scope=MemoryScope.USER,
        scope_id="user_123",
        user_context=user_ctx,
    )
    thought.entities = ["Project:Apollo"]

    # Query has "Drug Z" (via mock)
    query = "Tell me about Drug Z"
    context = UserContext(user_id="user_123", email="test@example.com")

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

    user_ctx = UserContext(user_id="user_123", email="test@example.com")
    await archive.add_thought("q", "r", MemoryScope.USER, "user_123", user_context=user_ctx)
    context = UserContext(user_id="user_123", email="test@example.com")

    # Should not raise exception
    results = await archive.retrieve("query", context)

    assert len(results) == 1
    # Check logs if possible, but mainly ensure no crash


@pytest.mark.asyncio
async def test_query_entity_expansion_boost() -> None:
    """
    Test Case 4: Test that a thought linked to a neighbor of the query entity is boosted.
    Scenario:
    - Graph: "Drug:Z" -> RELATED_TO -> "Concept:Cisplatin"
    - Thought: Contains "Concept:Cisplatin" (but NOT "Drug:Z")
    - Query: "Drug Z" (Extracts "Drug:Z")

    Expectation:
    - "Drug:Z" is extracted from query.
    - Archive expands "Drug:Z" to find neighbor "Concept:Cisplatin".
    - Thought with "Concept:Cisplatin" is boosted.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Setup Graph: Drug:Z -> Concept:Cisplatin
    g_store.add_relationship("Drug:Z", "Concept:Cisplatin", GraphEdgeType.RELATED_TO)

    # 2. Add Thought with Concept:Cisplatin
    user_ctx = UserContext(user_id="user_1", email="test@example.com")
    thought = await archive.add_thought(
        prompt="Chemo protocols",
        response="Use Cisplatin for efficacy.",
        scope=MemoryScope.USER,
        scope_id="user_1",
        user_context=user_ctx,
    )
    # Manually inject entities (skipping background extractor for determinism)
    thought.entities = ["Concept:Cisplatin"]

    # 3. Query for "Drug Z"
    context = UserContext(user_id="user_1", email="test@example.com")
    results = await archive.retrieve(query="Tell me about Drug Z", context=context, limit=1, graph_boost_factor=2.0)

    assert len(results) > 0
    top_thought, top_score, meta = results[0]

    # Verification
    assert meta.get("is_boosted") is True, "Thought should be boosted via 1-hop graph expansion of query entity"


@pytest.mark.asyncio
async def test_hybrid_retrieval_low_vector_similarity() -> None:
    """
    Test Case 5: Verify "User Story A" - Hybrid Retrieval.
    Scenario:
    - Thought T1 has very low vector similarity to query (excluded from vector search results).
    - But T1 is strongly linked via Graph to the Query Entity.
    - Expectation: T1 is retrieved despite low vector score.
    """
    v_store = VectorStore()
    g_store = GraphStore()

    # Custom Embedder that makes query and thought orthogonal (0 similarity)
    class OrthogonalEmbedder(Embedder):
        def embed(self, text: str) -> List[float]:
            if "query" in text:
                return [1.0, 0.0]  # Query Vector
            return [0.0, 1.0]  # Thought Vector

    embedder = OrthogonalEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Add Thought T1 about "Drug Z" (but text yields orthogonal vector)
    user_ctx = UserContext(user_id="user_1", email="test@example.com")
    t1 = await archive.add_thought(
        prompt="Different context",
        response="Some content.",
        scope=MemoryScope.USER,
        scope_id="user_1",
        user_context=user_ctx,
    )
    # T1 vector will be [0, 1]
    # Set entities manually to link it
    t1.entities = ["Drug:Z"]
    # Ensure graph node exists for T1 (add_thought does it via process_entities usually, do manually here)
    g_store.add_entity(f"Thought:{t1.id}")
    g_store.add_relationship("Drug:Z", f"Thought:{t1.id}", GraphEdgeType.RELATED_TO)

    # 2. Add Thought T2 (Control) - also orthogonal, no link
    t2 = await archive.add_thought(
        prompt="Noise",
        response="Noise.",
        scope=MemoryScope.USER,
        scope_id="user_1",
        user_context=user_ctx,
    )
    # T2 vector [0, 1]

    # 3. Query "query about Drug Z"
    # Query vector [1, 0]
    # Similarity to T1 and T2 is 0.0.
    # Vector Search with min_score=0.1 would exclude them.
    query = "query about Drug Z"
    context = UserContext(user_id="user_1", email="test@example.com")

    # We set min_score=0.1 to prove T1 would be lost in standard retrieval
    results = await archive.retrieve(
        query=query,
        context=context,
        limit=10,
        min_score=0.1,
        graph_boost_factor=10.0,  # High boost
    )

    # 4. Verify
    # T2 should be missing (score 0.0 < 0.1, no graph link)
    # T1 should be present (sourced via graph, even if base score is 0.0)
    # Wait, if base score is 0.0, boost * 0.0 = 0.0.
    # So it might still be ranked low, but it should be PRESENT in the list if we don't filter graph results.
    # My implementation does NOT filter graph-sourced items by min_score before merging.
    # However, it filters `filtered_candidates` by `filter_fn` (RBAC).
    # And then sorts.
    # Does it apply min_score at the end? No. `retrieve` signature has `min_score` used in `vector_store.search`.

    # So T1 should be in results with score 0.0 (or boosted 0.0).
    # IF dot product is 0.
    # Let's adjust vector slightly to have non-zero base score but < min_score?
    # Or rely on boost being additive? No, boost is multiplicative (PRD).
    # If base score is 0, boost does nothing.
    # "Story A" implies "Low similarity" (not zero).
    # Let's make Embedder return [0.1, 0.9] for thought.
    # Query [1.0, 0.0].
    # Sim = 0.1 / (1 * ~0.9) ~= 0.11.
    # Let's make it simpler.
    # Query: [1, 0]. Thought: [0.05, 0.99...]. Sim ~ 0.05.
    # min_score = 0.1.
    # Thought excluded by Vector Search.
    # Retrieved by Graph.
    # Boosted: 0.05 * 10 = 0.5.
    # Result: 0.5.

    # Redefine embedder
    class LowSimEmbedder(Embedder):
        def embed(self, text: str) -> List[float]:
            if "query" in text:
                return [1.0, 0.0]
            # Thought vector: small component in x, large in y
            return [0.05, 0.998]  # norm ~1.0

    archive.embedder = LowSimEmbedder()
    # Update thought vectors (hacky, but easier than re-adding)
    t1.vector = [0.05, 0.998]
    t2.vector = [0.05, 0.998]
    # Update VectorStore cache if needed (it uses .vector attribute in search, or cached _vectors list)
    # VectorStore implementation caches _vectors list. We must update it.
    v_store._vectors = [t.vector for t in v_store.thoughts]

    results = await archive.retrieve(
        query=query,
        context=context,
        limit=10,
        min_score=0.1,  # Threshold higher than 0.05
        graph_boost_factor=10.0,
    )

    t1_result = next((r for r in results if r[0].id == t1.id), None)
    t2_result = next((r for r in results if r[0].id == t2.id), None)

    assert t1_result is not None, "T1 should be retrieved via Graph Sourcing"
    assert t2_result is None, "T2 should be excluded by Vector Search min_score and has no graph link"

    # Check score
    # Base ~0.05. Boost 10. Final ~0.5.
    assert t1_result[1] > 0.4
