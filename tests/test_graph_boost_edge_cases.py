from typing import List, Tuple

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


@pytest.fixture
def archive_setup() -> Tuple[CoreasonArchive, VectorStore, GraphStore]:
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)
    return archive, v_store, g_store


@pytest.mark.asyncio
async def test_boosting_bidirectional_links(archive_setup: Tuple[CoreasonArchive, VectorStore, GraphStore]) -> None:
    """
    Test that traversal works for both incoming and outgoing edges.
    1. ProjectA -> RELATED_TO -> Concept:Out (Outgoing)
    2. Concept:In -> BELONGS_TO -> ProjectA (Incoming)
    Thoughts with Concept:Out or Concept:In should both be boosted.
    """
    archive, v_store, g_store = archive_setup

    # Setup Graph
    # Project:A -> Outgoing -> Concept:Out
    g_store.add_relationship("Project:A", "Concept:Out", GraphEdgeType.RELATED_TO)
    # Concept:In -> Incoming -> Project:A
    g_store.add_relationship("Concept:In", "Project:A", GraphEdgeType.BELONGS_TO)

    # Thoughts
    # T1: Linked via Outgoing
    t1 = await archive.add_thought("1", "1", MemoryScope.USER, "u1", "u1")
    t1.entities = ["Concept:Out"]

    # T2: Linked via Incoming
    t2 = await archive.add_thought("2", "2", MemoryScope.USER, "u1", "u1")
    t2.entities = ["Concept:In"]

    # T3: Control (No link)
    t3 = await archive.add_thought("3", "3", MemoryScope.USER, "u1", "u1")
    t3.entities = ["Concept:None"]

    context = UserContext(user_id="u1", project_ids=["A"])

    results = await archive.retrieve("q", context, limit=10, graph_boost_factor=2.0)

    scores = {r.id: s for r, s, _ in results}

    # T1 and T2 should be boosted (~2.0 * decay), T3 should be base (~1.0 * decay)
    # Since created_at is identical (ms diff), decay is negligible.
    assert scores[t1.id] > scores[t3.id] * 1.5
    assert scores[t2.id] > scores[t3.id] * 1.5

    # T1 and T2 should have roughly similar scores (both boosted once)
    assert abs(scores[t1.id] - scores[t2.id]) < 0.1


@pytest.mark.asyncio
async def test_boosting_multiple_active_projects(
    archive_setup: Tuple[CoreasonArchive, VectorStore, GraphStore],
) -> None:
    """
    Test aggregation of neighbors from multiple active projects.
    User in Project A and Project B.
    Thought linked to A should boost.
    Thought linked to B should boost.
    """
    archive, v_store, g_store = archive_setup

    g_store.add_relationship("Project:A", "Concept:A", GraphEdgeType.RELATED_TO)
    g_store.add_relationship("Project:B", "Concept:B", GraphEdgeType.RELATED_TO)

    t_a = await archive.add_thought("A", "A", MemoryScope.USER, "u1", "u1")
    t_a.entities = ["Concept:A"]

    t_b = await archive.add_thought("B", "B", MemoryScope.USER, "u1", "u1")
    t_b.entities = ["Concept:B"]

    t_none = await archive.add_thought("N", "N", MemoryScope.USER, "u1", "u1")
    t_none.entities = ["Concept:None"]

    context = UserContext(user_id="u1", project_ids=["A", "B"])

    results = await archive.retrieve("q", context, graph_boost_factor=2.0)
    scores = {r.id: s for r, s, _ in results}

    assert scores[t_a.id] > scores[t_none.id] * 1.5
    assert scores[t_b.id] > scores[t_none.id] * 1.5


@pytest.mark.asyncio
async def test_no_boost_disconnected(archive_setup: Tuple[CoreasonArchive, VectorStore, GraphStore]) -> None:
    """
    Test that a thought containing an entity that is in the graph but NOT linked
    to the active project does NOT get boosted.
    """
    archive, v_store, g_store = archive_setup

    # Graph has Project:A and Concept:Disconnected, but no edge between them
    g_store.add_entity("Project:A")
    g_store.add_entity("Concept:Disconnected")

    # Thought has the disconnected entity
    t1 = await archive.add_thought("1", "1", MemoryScope.USER, "u1", "u1")
    t1.entities = ["Concept:Disconnected"]

    # Control thought
    t2 = await archive.add_thought("2", "2", MemoryScope.USER, "u1", "u1")
    t2.entities = ["Concept:Other"]

    context = UserContext(user_id="u1", project_ids=["A"])

    results = await archive.retrieve("q", context, graph_boost_factor=2.0)
    scores = {r.id: s for r, s, _ in results}

    # Scores should be identical (roughly) as neither is boosted
    assert abs(scores[t1.id] - scores[t2.id]) < 0.001


@pytest.mark.asyncio
async def test_boost_factor_control(archive_setup: Tuple[CoreasonArchive, VectorStore, GraphStore]) -> None:
    """
    Test that setting boost factor to 1.0 results in no change.
    """
    archive, v_store, g_store = archive_setup

    g_store.add_relationship("Project:A", "Concept:A", GraphEdgeType.RELATED_TO)

    t1 = await archive.add_thought("1", "1", MemoryScope.USER, "u1", "u1")
    t1.entities = ["Concept:A"]  # Linked

    t2 = await archive.add_thought("2", "2", MemoryScope.USER, "u1", "u1")
    t2.entities = ["Concept:B"]  # Not Linked

    context = UserContext(user_id="u1", project_ids=["A"])

    # Factor 1.0 -> No boost
    results = await archive.retrieve("q", context, graph_boost_factor=1.0)
    scores = {r.id: s for r, s, _ in results}

    # Scores should be effectively equal
    assert abs(scores[t1.id] - scores[t2.id]) < 0.001
