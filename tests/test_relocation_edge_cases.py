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
from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.relocation import CoreasonRelocationManager
from coreason_archive.vector_store import VectorStore


@pytest.fixture
def stores() -> tuple[VectorStore, GraphStore]:
    return VectorStore(), GraphStore()


@pytest.fixture
def manager(stores: tuple[VectorStore, GraphStore]) -> CoreasonRelocationManager:
    v_store, g_store = stores
    return CoreasonRelocationManager(v_store, g_store)


def create_thought(
    user_id: str,
    access_roles: list[str] | None = None,
) -> CachedThought:
    return CachedThought(
        id=uuid4(),
        vector=[0.0] * 1536,
        entities=[],
        scope=MemoryScope.USER,
        scope_id=user_id,
        prompt_text="test",
        reasoning_trace="test",
        final_response="test",
        owner_id=user_id,
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=access_roles or [],
    )


@pytest.mark.asyncio
async def test_role_access_or_logic_preservation(manager: CoreasonRelocationManager) -> None:
    """
    Edge Case: Thought requires Role A OR Role B.
    User has Role A. (Access OK).
    User switches to Role B. (Access OK).
    Expectation: Thought is preserved.
    """
    user_id = "user_switch"
    # access_roles=["A", "B"] means A OR B is required
    thought = create_thought(user_id, access_roles=["A", "B"])
    manager.vector_store.add(thought)

    # Initial state assumption: User had "A".
    # Action: User changes to "B".
    await manager.on_role_change(user_id, ["B"])

    remaining = manager.vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 1
    assert remaining[0].id == thought.id


@pytest.mark.asyncio
async def test_total_role_revocation(manager: CoreasonRelocationManager) -> None:
    """
    Edge Case: User loses ALL roles.
    Protected thought (requires "A") -> Deleted.
    Public thought (requires []) -> Preserved.
    """
    user_id = "user_revoked"
    t_protected = create_thought(user_id, access_roles=["A"])
    t_public = create_thought(user_id, access_roles=[])

    manager.vector_store.add(t_protected)
    manager.vector_store.add(t_public)

    # Action: User roles become empty
    await manager.on_role_change(user_id, [])

    remaining = manager.vector_store.get_by_scope(MemoryScope.USER, user_id)
    ids = [t.id for t in remaining]

    assert t_public.id in ids
    assert t_protected.id not in ids


@pytest.mark.asyncio
async def test_non_matching_roles(manager: CoreasonRelocationManager) -> None:
    """
    Edge Case: User gets roles that don't match requirements.
    User gets "C". Thought needs "A". -> Deleted.
    """
    user_id = "user_mismatch"
    t = create_thought(user_id, access_roles=["A"])
    manager.vector_store.add(t)

    await manager.on_role_change(user_id, ["C"])

    remaining = manager.vector_store.get_by_scope(MemoryScope.USER, user_id)
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_complex_bulk_sanitization(manager: CoreasonRelocationManager) -> None:
    """
    Complex Scenario: Bulk processing.
    50 thoughts needing "A" (Keep).
    50 thoughts needing "B" (Delete).
    50 thoughts needing "A" or "B" (Keep).
    User roles: ["A"].
    """
    user_id = "user_bulk"

    keep_a = []
    delete_b = []
    keep_ab = []

    for _ in range(50):
        t = create_thought(user_id, access_roles=["A"])
        manager.vector_store.add(t)
        keep_a.append(t)

    for _ in range(50):
        t = create_thought(user_id, access_roles=["B"])
        manager.vector_store.add(t)
        delete_b.append(t)

    for _ in range(50):
        t = create_thought(user_id, access_roles=["A", "B"])
        manager.vector_store.add(t)
        keep_ab.append(t)

    # Action: User has "A"
    await manager.on_role_change(user_id, ["A"])

    remaining = manager.vector_store.get_by_scope(MemoryScope.USER, user_id)
    remaining_ids = {t.id for t in remaining}

    # Verify "A" kept
    for t in keep_a:
        assert t.id in remaining_ids

    # Verify "B" deleted
    for t in delete_b:
        assert t.id not in remaining_ids

    # Verify "A or B" kept (because user has A)
    for t in keep_ab:
        assert t.id in remaining_ids

    assert len(remaining) == 100
