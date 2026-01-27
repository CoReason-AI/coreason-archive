# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

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


def create_thought(
    user_id: str,
    entities: list[str],
    content: str = "Test thought",
    access_roles: list[str] | None = None,
) -> CachedThought:
    return CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=entities,
        scope=MemoryScope.USER,
        scope_id=user_id,
        prompt_text=content,
        reasoning_trace=content,
        final_response=content,
        owner_id=user_id,
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=access_roles or [],
    )


@pytest.mark.asyncio
async def test_relocation_multiple_departments_conservative(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: Entity 'Project:Joint' belongs to 'Department:A' AND 'Department:B'.
    User loses access to 'Department:A' but keeps 'Department:B'.
    Expectation: Thought is DELETED. (Conservative security: if it touches a restricted context, it's unsafe).
    """
    user_id = "user_joint"

    # Graph Setup
    graph_store.add_relationship("Project:Joint", "Department:A", GraphEdgeType.BELONGS_TO)
    graph_store.add_relationship("Project:Joint", "Department:B", GraphEdgeType.BELONGS_TO)

    thought = create_thought(user_id, ["Project:Joint"], "Joint project notes")
    vector_store.add(thought)

    # User keeps Role B, loses Role A
    new_roles = ["dept:B"]

    await manager.on_role_change(user_id, new_roles)

    remaining = vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 0, (
        "Thought linked to lost Department:A should be deleted despite having Department:B access"
    )


@pytest.mark.asyncio
async def test_relocation_case_sensitivity(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: 'Department:Sales'. User has role 'dept:sales' (lowercase).
    Expectation: Mismatch -> Deletion (Strict case sensitivity).
    """
    user_id = "user_case"

    graph_store.add_relationship("Project:Sales", "Department:Sales", GraphEdgeType.BELONGS_TO)

    thought = create_thought(user_id, ["Project:Sales"], "Sales notes")
    vector_store.add(thought)

    # Role is lowercase 'sales'
    # Logic checks: 'dept:Sales' in roles OR 'Sales' in roles.
    # 'dept:sales' != 'dept:Sales'.
    new_roles = ["dept:sales"]

    await manager.on_role_change(user_id, new_roles)

    remaining = vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 0, "Strict case sensitivity should cause deletion on mismatch"


@pytest.mark.asyncio
async def test_relocation_total_revocation(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario: User loses ALL roles.
    Expectation:
    1. Thoughts linked to ANY department -> Deleted.
    2. Thoughts linked to NO department -> Kept (Personal/Public).
    """
    user_id = "user_revoke"

    # Graph
    graph_store.add_relationship("Project:Secret", "Department:X", GraphEdgeType.BELONGS_TO)
    graph_store.add_entity("Project:Public")  # No Dept link

    t_secret = create_thought(user_id, ["Project:Secret"], "Secret")
    t_public = create_thought(user_id, ["Project:Public"], "Public")

    vector_store.add(t_secret)
    vector_store.add(t_public)

    # Lose all roles
    await manager.on_role_change(user_id, [])

    remaining = vector_store.get_by_scope(MemoryScope.USER, user_id)
    ids = [t.id for t in remaining]

    assert t_secret.id not in ids
    assert t_public.id in ids


@pytest.mark.asyncio
async def test_relocation_indirect_links_safe(
    manager: CoreasonRelocationManager, vector_store: VectorStore, graph_store: GraphStore
) -> None:
    """
    Scenario:
    Thought -> Entity:Task -> RELATED_TO -> Project:Secret -> BELONGS_TO -> Department:X.
    Implementation is 1-hop scan from thought entities.
    So checking 'Entity:Task' does not verify 'Project:Secret' ownership.
    Expectation: Thought is KEPT (because current logic doesn't do deep traversal).
    This documents the limitation/scope of the current "Atomic Unit".
    """
    user_id = "user_indirect"

    graph_store.add_relationship("Project:Secret", "Department:X", GraphEdgeType.BELONGS_TO)
    graph_store.add_relationship("Entity:Task", "Project:Secret", GraphEdgeType.RELATED_TO)

    # Thought only knows about "Entity:Task"
    thought = create_thought(user_id, ["Entity:Task"], "Task notes")
    vector_store.add(thought)

    # User has no roles
    await manager.on_role_change(user_id, [])

    remaining = vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 1, "Indirect 2-hop links are not currently sanitized"
