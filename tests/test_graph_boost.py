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

from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


@pytest.mark.asyncio
async def test_indirect_graph_boosting() -> None:
    """
    Scenario:
    Thought T1 has entity 'Concept:A'.
    'Concept:A' is linked to 'Project:Apollo' in the Graph.
    User Context includes 'Project:Apollo'.

    Current behavior (without change): T1 is NOT boosted.
    Desired behavior: T1 IS boosted because it is 1 hop away.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # 1. Add Thought T1 with entity 'Concept:A'
    # Use 'Project:Apollo' in the prompt/response so it might get picked up if we had an extractor,
    # but here we manually set entities to ensure we test the graph link, not text matching.
    user_ctx = UserContext(user_id="user_1", email="test@example.com")
    t1 = await archive.add_thought(
        prompt="Discussing Concept A",
        response="Concept A is great.",
        scope=MemoryScope.USER,
        scope_id="user_1",
        user_context=user_ctx,
    )
    # Manually set entities (simulating extraction)
    t1.entities = ["Concept:A"]

    # 2. Add Thought T2 with NO relevant entities (Control group)
    t2 = await archive.add_thought(
        prompt="Discussing Concept B",
        response="Concept B is okay.",
        scope=MemoryScope.USER,
        scope_id="user_1",
        user_context=user_ctx,
    )
    t2.entities = ["Concept:B"]

    # 3. Define Graph Relationship: Concept:A <-> Project:Apollo
    # We add relationship in both directions or just one?
    # Usually "Concept:A BELONGS_TO Project:Apollo" or "Project:Apollo RELATED_TO Concept:A".
    # The requirement is "1 hop traversal".
    # Let's say "Concept:A" BELONGS_TO "Project:Apollo".
    g_store.add_relationship("Concept:A", "Project:Apollo", GraphEdgeType.BELONGS_TO)
    # Ensure reciprocal link or check if retrieve looks at incoming edges?
    # PRD says: "Graph Traversal -> Boost score if Candidate is linked to Active Project Node."
    # If we start at Active Project ("Project:Apollo"), we need to find "Concept:A".
    # If edge is Concept:A -> Project:Apollo, then from Project:Apollo it is an INCOMING edge.

    # 4. Retrieval Context
    context = UserContext(
        user_id="user_1",
        email="test@example.com",
        groups=["Apollo"],  # Matches "Project:Apollo"
    )

    # 5. Retrieve
    # Base score for both will be identical (since vectors are identical [0.1]*1536).
    # If boosted, T1 score > T2 score.
    # graph_boost_factor default is 1.1
    results = await archive.retrieve(
        query="query",
        context=context,
        limit=10,
        graph_boost_factor=1.5,  # Make it obvious
    )

    # Extract thoughts and scores
    t1_result = next((r for r in results if r[0].id == t1.id), None)
    t2_result = next((r for r in results if r[0].id == t2.id), None)

    assert t1_result is not None
    assert t2_result is not None

    score_t1 = t1_result[1]
    score_t2 = t2_result[1]

    # This assertion is expected to FAIL until we implement the traversal
    assert score_t1 > score_t2, f"T1 ({score_t1}) should be boosted higher than T2 ({score_t2})"
