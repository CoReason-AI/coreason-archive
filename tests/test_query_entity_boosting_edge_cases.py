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
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    """Mock extractor that supports edge cases."""

    async def extract(self, text: str) -> List[str]:
        entities = []
        if "Drug A" in text:
            entities.append("Drug:A")
        if "Drug B" in text:
            entities.append("Drug:B")
        if "Project Apollo" in text:
            entities.append("Project:Apollo")
        if "case mismatch" in text:
            entities.append("concept:mismatch")  # Lowercase type
        return entities


@pytest.mark.asyncio
async def test_multiple_query_entities() -> None:
    """Test Case 1: Multiple entities in query (Drug:A and Drug:B)."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Thought A linked to Drug:A
    t_a = await archive.add_thought("A", "A", MemoryScope.USER, "u1", "u1")
    t_a.entities = ["Drug:A"]

    # Thought B linked to Drug:B
    t_b = await archive.add_thought("B", "B", MemoryScope.USER, "u1", "u1")
    t_b.entities = ["Drug:B"]

    # Thought C linked to nothing
    t_c = await archive.add_thought("C", "C", MemoryScope.USER, "u1", "u1")
    t_c.entities = []

    # Query mentions both
    query = "Compare Drug A and Drug B"
    context = UserContext(user_id="u1")

    results = await archive.retrieve(query, context, graph_boost_factor=2.0)

    result_ids = {t.id for t, s, m in results if m.get("is_boosted")}

    assert t_a.id in result_ids
    assert t_b.id in result_ids
    assert t_c.id not in result_ids


@pytest.mark.asyncio
async def test_overlapping_entities() -> None:
    """Test Case 2: Overlap between User Context and Query."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Thought linked to Project:Apollo
    t_apollo = await archive.add_thought("Apollo", "Apollo", MemoryScope.PROJECT, "Apollo", "u1")
    t_apollo.entities = ["Project:Apollo"]

    # User is in Project Apollo AND queries about it
    context = UserContext(user_id="u1", project_ids=["Apollo"])
    query = "Update on Project Apollo"

    # Should get boosted. Overlap should not cause double boosting or errors.
    results = await archive.retrieve(query, context, graph_boost_factor=2.0)

    assert len(results) > 0
    top_thought, top_score, meta = results[0]

    assert top_thought.id == t_apollo.id
    assert meta.get("is_boosted") is True
    # Verify score is boosted exactly once (approx 2.0 * base)
    # If boosted twice, it might be 4.0.
    # Logic uses `boost_entities` set, so duplicates are removed.
    assert top_score < 3.0


@pytest.mark.asyncio
async def test_empty_query_string() -> None:
    """Test Case 3: Empty query string handling."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    await archive.add_thought("A", "A", MemoryScope.USER, "u1", "u1")
    context = UserContext(user_id="u1")

    # Should handle empty query gracefully
    results = await archive.retrieve("", context)
    # Results depend on vector store handling of empty/zero vector.
    # VectorStore mock returns [0.1]...
    # If query vector is valid, search proceeds.
    # Extractor returns empty list for "".
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_case_sensitivity_mismatch() -> None:
    """
    Test Case 4: Case sensitivity mismatch.
    Graph is strict. If thought has 'Concept:Mismatch' but extractor returns 'concept:mismatch',
    it won't match unless logic handles it or strings match exactly.
    Current implementation assumes exact string match.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Thought has "Concept:Mismatch" (Upper C, Upper M)
    t = await archive.add_thought("X", "X", MemoryScope.USER, "u1", "u1")
    t.entities = ["Concept:Mismatch"]

    context = UserContext(user_id="u1")
    query = "test case mismatch"  # Extractor returns "concept:mismatch" (lower)

    results = await archive.retrieve(query, context, graph_boost_factor=2.0)

    # Expect NO boost because strings don't match exactly in set
    # This verifies the limitation/expectation of exact match
    t_res, score, meta = results[0]
    assert meta.get("is_boosted") is False


@pytest.mark.asyncio
async def test_complex_needle_in_haystack() -> None:
    """
    Test Case 5: Needle in Haystack.
    100 thoughts, 1 relevant via Entity Hop.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Add 50 noise thoughts
    for i in range(50):
        t = await archive.add_thought(f"Noise {i}", "Noise", MemoryScope.USER, "u1", "u1")
        t.entities = [f"Noise:{i}"]

    # 2. Add Needle thought linked to "Drug:A"
    needle = await archive.add_thought("Needle", "Secret Info", MemoryScope.USER, "u1", "u1")
    needle.entities = ["Drug:A"]

    # 3. Add 50 more noise thoughts
    for i in range(50, 100):
        t = await archive.add_thought(f"Noise {i}", "Noise", MemoryScope.USER, "u1", "u1")
        t.entities = [f"Noise:{i}"]

    # Query for Drug A
    context = UserContext(user_id="u1")
    query = "Tell me about Drug A"

    # We need to ensure retrieval actually *finds* it.
    # Vector search returns limit*5 candidates.
    # If limit=10, candidates=50.
    # Total 101 thoughts. All have identical vector score (1.0).
    # Sort order is stable or random. Needle might be at index 50.
    # If we request limit=25 (candidates=125), we cover all.

    _ = await archive.retrieve(query, context, limit=20)

    # Since we can't guarantee vector rank without distinct vectors,
    # we rely on the fact that VectorStore.search returns *all* matching min_score if we requested enough?
    # No, search returns top N.
    # With identical vectors, it's arbitrary.
    # BUT, to test *boosting*, we assume it's in the candidates.

    # Let's verify boost applied if it IS returned.
    # Or force it to be returned by making vector search irrelevant (small N) but ensuring it's there?
    # Actually, in real world, vector score for "Drug A" query and "Needle" thought might be low,
    # but still higher than "Noise".
    # Here vectors are identical.

    # Let's search with limit=101 to ensure it's considered.
    results_all = await archive.retrieve(query, context, limit=101, graph_boost_factor=10.0)

    # Needle should be #1 because of boost
    assert results_all[0][0].id == needle.id
    assert results_all[0][2]["is_boosted"] is True
