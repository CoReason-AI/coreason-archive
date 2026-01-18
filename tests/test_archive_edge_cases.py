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
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.relocation import CoreasonRelocationManager
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


@pytest.fixture
def components() -> Tuple[CoreasonArchive, CoreasonRelocationManager, VectorStore, GraphStore]:
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)
    manager = CoreasonRelocationManager(v_store, g_store)
    return archive, manager, v_store, g_store


def test_define_relationship_invalid_format(
    components: Tuple[CoreasonArchive, CoreasonRelocationManager, VectorStore, GraphStore],
) -> None:
    archive, _, _, _ = components

    # Missing colon in source
    with pytest.raises(ValueError, match="Type:Value"):
        archive.define_entity_relationship("InvalidEntity", "Department:A", GraphEdgeType.BELONGS_TO)

    # Missing colon in target
    with pytest.raises(ValueError, match="Type:Value"):
        archive.define_entity_relationship("Project:A", "InvalidTarget", GraphEdgeType.BELONGS_TO)


def test_define_relationship_idempotency(
    components: Tuple[CoreasonArchive, CoreasonRelocationManager, VectorStore, GraphStore],
) -> None:
    archive, _, _, g_store = components

    source = "Project:A"
    target = "Department:A"
    relation = GraphEdgeType.BELONGS_TO

    # Add first time
    archive.define_entity_relationship(source, target, relation)

    # Verify edge exists
    related = g_store.get_related_entities(source, relation, "outgoing")
    assert len(related) == 1
    assert related[0][0] == target

    # Add second time
    archive.define_entity_relationship(source, target, relation)

    # Verify still only one unique relationship of this type/value
    # NetworkX MultiDiGraph allows multiple edges, but our GraphStore.add_relationship uses a key.
    # So it should be overwritten, not duplicated.
    related_after = g_store.get_related_entities(source, relation, "outgoing")
    assert len(related_after) == 1
    assert related_after[0][0] == target


def test_define_relationship_self_loop(
    components: Tuple[CoreasonArchive, CoreasonRelocationManager, VectorStore, GraphStore],
) -> None:
    archive, _, _, g_store = components

    entity = "Project:Infinite"
    archive.define_entity_relationship(entity, entity, GraphEdgeType.RELATED_TO)

    related = g_store.get_related_entities(entity, GraphEdgeType.RELATED_TO, "outgoing")
    assert len(related) == 1
    assert related[0][0] == entity


@pytest.mark.asyncio
async def test_full_relocation_flow_with_defined_hierarchy(
    components: Tuple[CoreasonArchive, CoreasonRelocationManager, VectorStore, GraphStore],
) -> None:
    archive, manager, v_store, _ = components

    user_id = "agent_smith"
    old_dept = "MatrixOps"
    new_dept = "Resistance"

    # 1. Ingest Thought (manually specifying entity for simplicity,
    # as mock extraction isn't the focus here, but we can rely on add_thought flow)
    # We'll use add_thought but since we don't have an extractor in the fixture,
    # we manually inject the entity into the graph/thought after creation
    # OR we use a thought with manually assigned entities if we access the store directly.
    # Using archive.add_thought is cleaner, but entities won't be populated without extractor.
    # Let's populate entities manually after add_thought for the test scenario.

    thought = await archive.add_thought(
        prompt="Update code", response="Modifying the kernel", scope=MemoryScope.USER, scope_id=user_id, user_id=user_id
    )
    # Manually attach entity
    thought.entities = ["Project:Zion"]
    # Update vector store to reflect this change (since it's in-memory object reference, it might be fine,
    # but let's be safe. Actually v_store.thoughts stores the object, so modifying `thought` here modifies it in store)

    # 2. Define Hierarchy: Project:Zion BELONGS_TO Department:MatrixOps
    archive.define_entity_relationship("Project:Zion", f"Department:{old_dept}", GraphEdgeType.BELONGS_TO)

    # 3. Verify Setup
    assert len(v_store.get_by_scope(MemoryScope.USER, user_id)) == 1

    # 4. Perform Transfer
    await manager.on_dept_transfer(user_id, old_dept, new_dept)

    # 5. Verify Deletion
    # The thought contained "Project:Zion" which belongs to "Department:MatrixOps".
    # User left "MatrixOps". Thought should be gone.
    remaining = v_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 0
