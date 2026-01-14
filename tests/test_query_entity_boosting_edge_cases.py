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

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    def __init__(self, entities_to_return: List[str] | None = None):
        self.entities = entities_to_return or []

    async def extract(self, text: str) -> List[str]:
        return self.entities


@pytest.fixture
def base_archive() -> Tuple[VectorStore, GraphStore, MockEmbedder]:
    v = VectorStore()
    g = GraphStore()
    e = MockEmbedder()
    return v, g, e


@pytest.mark.asyncio
async def test_two_hop_exclusion(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Verify that boosting is strictly 1-hop.
    Graph: Entity:Query -> Entity:Link1 -> Entity:Link2
    Thought linked to Entity:Link2 should NOT be boosted.
    """
    v, g, e = base_archive
    extractor = MockEntityExtractor(["Entity:Query"])
    archive = CoreasonArchive(v, g, e, extractor)

    # Setup Graph
    g.add_relationship("Entity:Query", "Entity:Link1", GraphEdgeType.RELATED_TO)
    g.add_relationship("Entity:Link1", "Entity:Link2", GraphEdgeType.RELATED_TO)

    # Thought linked to Link2 (2 hops away)
    thought = await archive.add_thought("prompt", "response", MemoryScope.USER, "u1", "u1")
    thought.entities = ["Entity:Link2"]

    context = UserContext(user_id="u1")
    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    assert len(results) > 0
    _, _, meta = results[0]

    assert meta.get("is_boosted") is False, "2-hop neighbor should NOT be boosted"


@pytest.mark.asyncio
async def test_circular_graph_stability(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Verify 1-hop expansion handles loops gracefully.
    Graph: Node:A <-> Node:B
    Query: Node:A
    Thought: Node:B (Should be boosted)
    """
    v, g, e = base_archive
    extractor = MockEntityExtractor(["Node:A"])
    archive = CoreasonArchive(v, g, e, extractor)

    # Circular loop
    g.add_relationship("Node:A", "Node:B", GraphEdgeType.RELATED_TO)
    g.add_relationship("Node:B", "Node:A", GraphEdgeType.RELATED_TO)

    thought = await archive.add_thought("p", "r", MemoryScope.USER, "u1", "u1")
    thought.entities = ["Node:B"]

    context = UserContext(user_id="u1")
    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    assert len(results) > 0
    _, _, meta = results[0]
    assert meta.get("is_boosted") is True


@pytest.mark.asyncio
async def test_multiple_query_entities_complex(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Scenario:
    Query yields "Type:E1" and "Type:E2".
    Type:E1 -> Node:N1
    Type:E2 -> Node:N2
    Thought 1 linked to Node:N1
    Thought 2 linked to Node:N2
    Thought 3 linked to Type:E3 (unrelated)
    """
    v, g, e = base_archive
    extractor = MockEntityExtractor(["Type:E1", "Type:E2"])
    archive = CoreasonArchive(v, g, e, extractor)

    g.add_relationship("Type:E1", "Node:N1", GraphEdgeType.RELATED_TO)
    g.add_relationship("Type:E2", "Node:N2", GraphEdgeType.RELATED_TO)

    t1 = await archive.add_thought("1", "1", MemoryScope.USER, "u1", "u1")
    t1.entities = ["Node:N1"]

    t2 = await archive.add_thought("2", "2", MemoryScope.USER, "u1", "u1")
    t2.entities = ["Node:N2"]

    t3 = await archive.add_thought("3", "3", MemoryScope.USER, "u1", "u1")
    t3.entities = ["Type:E3"]

    context = UserContext(user_id="u1")
    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    res_map = {r[0].id: r for r in results}

    assert res_map[t1.id][2]["is_boosted"] is True
    assert res_map[t2.id][2]["is_boosted"] is True
    assert res_map[t3.id][2]["is_boosted"] is False


@pytest.mark.asyncio
async def test_graph_direction_both_ways(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    Verify expansion works for incoming edges too.
    Graph: Node:Child -> BELONGS_TO -> Node:Parent(QueryEntity)
    Query: Node:Parent
    Thought linked to Node:Child.
    """
    v, g, e = base_archive
    extractor = MockEntityExtractor(["Node:Parent"])
    archive = CoreasonArchive(v, g, e, extractor)

    g.add_relationship("Node:Child", "Node:Parent", GraphEdgeType.BELONGS_TO)

    thought = await archive.add_thought("p", "r", MemoryScope.USER, "u1", "u1")
    thought.entities = ["Node:Child"]

    context = UserContext(user_id="u1")
    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    assert len(results) > 0
    _, _, meta = results[0]
    assert meta.get("is_boosted") is True


@pytest.mark.asyncio
async def test_overlapping_active_context_and_query(base_archive: Tuple[VectorStore, GraphStore, MockEmbedder]) -> None:
    """
    User is in Project P.
    Query also extracts Project P.
    Graph: Project:P -> Node:Neighbor
    Thought linked to Node:Neighbor.
    """
    v, g, e = base_archive
    extractor = MockEntityExtractor(["Project:P"])
    archive = CoreasonArchive(v, g, e, extractor)

    g.add_relationship("Project:P", "Node:Neighbor", GraphEdgeType.RELATED_TO)

    # Use a different user scope to avoid auto-project context if we weren't explicit,
    # but here we set project scope on thought.
    # scope_id should be just "P" to match context.project_ids=["P"]
    thought = await archive.add_thought("p", "r", MemoryScope.PROJECT, "P", "u1")
    thought.entities = ["Node:Neighbor"]

    # Context puts user in Project:P
    context = UserContext(user_id="u1", project_ids=["P"])

    results = await archive.retrieve("query", context, graph_boost_factor=2.0)

    assert len(results) > 0
    _, _, meta = results[0]
    assert meta.get("is_boosted") is True
