from datetime import datetime, timezone
from uuid import uuid4

import pytest

from coreason_archive.graph_store import GraphStore
from coreason_archive.models import CachedThought, GraphEdgeType, MemoryScope
from coreason_archive.relocation import CoreasonRelocationManager
from coreason_archive.vector_store import VectorStore


@pytest.fixture
def vector_store() -> VectorStore:
    return VectorStore()


@pytest.fixture
def graph_store() -> GraphStore:
    return GraphStore()


@pytest.fixture
def manager(vector_store: VectorStore, graph_store: GraphStore) -> CoreasonRelocationManager:
    return CoreasonRelocationManager(vector_store, graph_store)


def create_thought(user_id: str, entities: list[str], content: str = "Test thought") -> CachedThought:
    return CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=entities,
        scope=MemoryScope.USER,
        scope_id=user_id,
        prompt_text=content,
        reasoning_trace=content,
        final_response=content,
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=[],
    )


@pytest.mark.asyncio
async def test_relocation_sanitization(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario:
    User has 2 thoughts:
    1. "I like coffee" (Entities: [Concept:Coffee]) -> Safe
    2. "Secret Project X details" (Entities: [Project:X]) -> Linked to Old Dept -> Unsafe

    Action: User transfers from DeptA to DeptB.
    Expectation: Thought 2 is deleted. Thought 1 remains.
    """
    user_id = "user_1"
    old_dept = "DeptA"
    new_dept = "DeptB"

    # Setup Graph
    # Project:X belongs to DeptA
    graph_store.add_relationship("Project:X", f"Department:{old_dept}", GraphEdgeType.BELONGS_TO)

    # Concept:Coffee is generic (no department link) or linked to global/other
    graph_store.add_entity("Concept:Coffee")

    # Setup Thoughts
    safe_thought = create_thought(user_id, ["Concept:Coffee"], "Coffee is good")
    unsafe_thought = create_thought(user_id, ["Project:X"], "Secret Project X")

    vector_store.add(safe_thought)
    vector_store.add(unsafe_thought)

    assert len(vector_store.thoughts) == 2

    # Run Transfer
    await manager.on_dept_transfer(user_id, old_dept, new_dept)

    # Verify
    remaining_thoughts = vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining_thoughts) == 1
    assert remaining_thoughts[0].id == safe_thought.id

    # Verify unsafe thought is gone
    assert unsafe_thought not in vector_store.thoughts


@pytest.mark.asyncio
async def test_relocation_mixed_contamination(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: Thought contains both Safe and Unsafe entities.
    Expectation: Thought is DELETED because it contains contaminated info.
    """
    user_id = "user_mixed"
    old_dept = "DeptA"

    # Setup Graph
    graph_store.add_relationship("Project:Unsafe", f"Department:{old_dept}", GraphEdgeType.BELONGS_TO)
    graph_store.add_entity("Project:Safe")

    # Thought has both
    thought = create_thought(user_id, ["Project:Safe", "Project:Unsafe"], "Mixed Content")
    vector_store.add(thought)

    # Run Transfer
    await manager.on_dept_transfer(user_id, old_dept, "DeptB")

    # Verify Deletion
    remaining = vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_entities_with_no_dept_link(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: Thought has entities, but those entities are NOT linked to the old department.
    Expectation: Kept.
    """
    user_id = "user_orphan"
    old_dept = "DeptA"

    # Setup Graph: Entity exists but has no BELONGS_TO edges
    graph_store.add_entity("Project:Orphan")
    # Entity linked to unrelated department
    graph_store.add_relationship("Project:Other", "Department:Other", GraphEdgeType.BELONGS_TO)

    thought = create_thought(user_id, ["Project:Orphan", "Project:Other"], "Safe Content")
    vector_store.add(thought)

    await manager.on_dept_transfer(user_id, old_dept, "DeptB")

    assert len(vector_store.thoughts) == 1


@pytest.mark.asyncio
async def test_idempotency(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: Run transfer twice.
    Expectation: Second run does nothing, no errors.
    """
    user_id = "user_idem"
    old_dept = "DeptA"

    graph_store.add_relationship("Project:X", f"Department:{old_dept}", GraphEdgeType.BELONGS_TO)
    thought = create_thought(user_id, ["Project:X"], "Unsafe")
    vector_store.add(thought)

    # First Run
    await manager.on_dept_transfer(user_id, old_dept, "DeptB")
    assert len(vector_store.thoughts) == 0

    # Second Run
    await manager.on_dept_transfer(user_id, old_dept, "DeptB")
    assert len(vector_store.thoughts) == 0


@pytest.mark.asyncio
async def test_relocation_no_effect_on_other_scopes(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Ensure thoughts in PROJECT or DEPT scope are not touched by this logic.
    """
    user_id = "user_1"
    old_dept = "DeptA"

    # Setup Graph
    graph_store.add_relationship("Project:X", f"Department:{old_dept}", GraphEdgeType.BELONGS_TO)

    # Setup Thought in PROJECT scope (even if linked to old dept entity, sanitization targets USER scope migration)
    # Why? Because DEPT scope thoughts naturally stay with the dept. PROJECT scope stays with project.
    # ONLY USER scope thoughts travel with the user and need cleaning.
    project_thought = CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=["Project:X"],
        scope=MemoryScope.PROJECT,  # NOT USER
        scope_id="project_x",
        prompt_text="Project stuff",
        reasoning_trace="trace",
        final_response="resp",
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=[],
    )

    vector_store.add(project_thought)

    await manager.on_dept_transfer(user_id, old_dept, "DeptB")

    assert len(vector_store.thoughts) == 1
    assert vector_store.thoughts[0].id == project_thought.id


@pytest.mark.asyncio
async def test_relocation_role_change_passive(manager: CoreasonRelocationManager) -> None:
    """Test that on_role_change is passive (no error)."""
    await manager.on_role_change("user_1", ["admin"])
    # If no exception, passed.
