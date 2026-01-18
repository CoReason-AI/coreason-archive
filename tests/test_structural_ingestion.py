# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.utils.stubs import StubEmbedder
from coreason_archive.vector_store import VectorStore


@pytest.mark.asyncio
async def test_structural_ingestion_user_scope() -> None:
    """Verify CREATED and BELONGS_TO edges for USER scope."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "alice"
    scope_id = "alice"

    thought = await archive.add_thought(
        prompt="p", response="r", scope=MemoryScope.USER, scope_id=scope_id, user_id=user_id
    )

    thought_node = f"Thought:{thought.id}"
    user_node = f"User:{user_id}"

    # Verify CREATED edge: User -> Thought
    assert g_store.graph.has_edge(user_node, thought_node, key=GraphEdgeType.CREATED.value)

    # Verify BELONGS_TO edge: Thought -> User
    # For USER scope, the scope entity is the User node
    assert g_store.graph.has_edge(thought_node, user_node, key=GraphEdgeType.BELONGS_TO.value)


@pytest.mark.asyncio
async def test_structural_ingestion_project_scope() -> None:
    """Verify CREATED and BELONGS_TO edges for PROJECT scope."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "bob"
    scope_id = "apollo"

    thought = await archive.add_thought(
        prompt="p", response="r", scope=MemoryScope.PROJECT, scope_id=scope_id, user_id=user_id
    )

    thought_node = f"Thought:{thought.id}"
    user_node = f"User:{user_id}"
    scope_node = f"Project:{scope_id}"

    # Verify CREATED edge
    assert g_store.graph.has_edge(user_node, thought_node, key=GraphEdgeType.CREATED.value)

    # Verify BELONGS_TO edge: Thought -> Project
    assert g_store.graph.has_edge(thought_node, scope_node, key=GraphEdgeType.BELONGS_TO.value)


@pytest.mark.asyncio
async def test_structural_ingestion_dept_scope() -> None:
    """Verify CREATED and BELONGS_TO edges for DEPARTMENT scope."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "charlie"
    scope_id = "engineering"

    thought = await archive.add_thought(
        prompt="p", response="r", scope=MemoryScope.DEPARTMENT, scope_id=scope_id, user_id=user_id
    )

    thought_node = f"Thought:{thought.id}"
    user_node = f"User:{user_id}"
    scope_node = f"Department:{scope_id}"

    assert g_store.graph.has_edge(user_node, thought_node, key=GraphEdgeType.CREATED.value)
    assert g_store.graph.has_edge(thought_node, scope_node, key=GraphEdgeType.BELONGS_TO.value)


@pytest.mark.asyncio
async def test_structural_ingestion_client_scope() -> None:
    """Verify CREATED and BELONGS_TO edges for CLIENT scope."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "dave"
    scope_id = "acme_corp"

    thought = await archive.add_thought(
        prompt="p", response="r", scope=MemoryScope.CLIENT, scope_id=scope_id, user_id=user_id
    )

    thought_node = f"Thought:{thought.id}"
    user_node = f"User:{user_id}"
    scope_node = f"Client:{scope_id}"

    assert g_store.graph.has_edge(user_node, thought_node, key=GraphEdgeType.CREATED.value)
    assert g_store.graph.has_edge(thought_node, scope_node, key=GraphEdgeType.BELONGS_TO.value)


@pytest.mark.asyncio
async def test_structural_ingestion_special_characters() -> None:
    """Verify handling of special characters in IDs."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "user@example.com"
    scope_id = "Project X & Y"

    thought = await archive.add_thought(
        prompt="p", response="r", scope=MemoryScope.PROJECT, scope_id=scope_id, user_id=user_id
    )

    thought_node = f"Thought:{thought.id}"
    user_node = f"User:{user_id}"
    scope_node = f"Project:{scope_id}"

    assert g_store.graph.has_node(user_node)
    assert g_store.graph.has_node(scope_node)
    assert g_store.graph.has_edge(user_node, thought_node, key=GraphEdgeType.CREATED.value)
    assert g_store.graph.has_edge(thought_node, scope_node, key=GraphEdgeType.BELONGS_TO.value)


@pytest.mark.asyncio
async def test_hub_and_spoke_topology() -> None:
    """Verify multiple thoughts link to the same scope/user (Hub and Spoke)."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "hub_user"
    scope_id = "hub_project"

    # Add 3 thoughts
    t1 = await archive.add_thought("p1", "r1", MemoryScope.PROJECT, scope_id, user_id)
    t2 = await archive.add_thought("p2", "r2", MemoryScope.PROJECT, scope_id, user_id)
    t3 = await archive.add_thought("p3", "r3", MemoryScope.PROJECT, scope_id, user_id)

    user_node = f"User:{user_id}"
    scope_node = f"Project:{scope_id}"

    # Verify User -> [t1, t2, t3]
    for t in [t1, t2, t3]:
        t_node = f"Thought:{t.id}"
        assert g_store.graph.has_edge(user_node, t_node, key=GraphEdgeType.CREATED.value)
        assert g_store.graph.has_edge(t_node, scope_node, key=GraphEdgeType.BELONGS_TO.value)

    # Check degrees
    # User out-degree should be 3 (CREATED)
    assert g_store.graph.out_degree(user_node) == 3
    # Scope in-degree should be 3 (BELONGS_TO)
    assert g_store.graph.in_degree(scope_node) == 3


@pytest.mark.asyncio
async def test_mixed_scope_usage() -> None:
    """
    Verify a single user creating thoughts across different scopes.
    User -> Thought1 -> Project A
    User -> Thought2 -> Department B
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "multitasker"

    # 1. Add thought to Project
    t1 = await archive.add_thought("p1", "r1", MemoryScope.PROJECT, "project_alpha", user_id)

    # 2. Add thought to Department
    t2 = await archive.add_thought("p2", "r2", MemoryScope.DEPARTMENT, "dept_beta", user_id)

    user_node = f"User:{user_id}"
    t1_node = f"Thought:{t1.id}"
    t2_node = f"Thought:{t2.id}"
    proj_node = "Project:project_alpha"
    dept_node = "Department:dept_beta"

    # Verify User is connected to both thoughts
    assert g_store.graph.has_edge(user_node, t1_node, key=GraphEdgeType.CREATED.value)
    assert g_store.graph.has_edge(user_node, t2_node, key=GraphEdgeType.CREATED.value)

    # Verify Thoughts connected to respective scopes
    assert g_store.graph.has_edge(t1_node, proj_node, key=GraphEdgeType.BELONGS_TO.value)
    assert g_store.graph.has_edge(t2_node, dept_node, key=GraphEdgeType.BELONGS_TO.value)

    # Verify NO cross-contamination
    assert not g_store.graph.has_edge(t1_node, dept_node)
    assert not g_store.graph.has_edge(t2_node, proj_node)


@pytest.mark.asyncio
async def test_node_reuse() -> None:
    """
    Verify that ingesting multiple thoughts for the same user/scope reuses existing graph nodes.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    user_id = "reuse_user"
    scope_id = "reuse_project"

    # Add first thought
    await archive.add_thought("p1", "r1", MemoryScope.PROJECT, scope_id, user_id)

    # Snapshot node count
    initial_nodes = set(g_store.graph.nodes)
    assert f"User:{user_id}" in initial_nodes
    assert f"Project:{scope_id}" in initial_nodes

    # Add second thought with SAME user and scope
    await archive.add_thought("p2", "r2", MemoryScope.PROJECT, scope_id, user_id)

    final_nodes = set(g_store.graph.nodes)

    # The only NEW node should be the second thought itself
    # User and Project nodes should not be duplicated (e.g. no "User:reuse_user_1")
    new_nodes = final_nodes - initial_nodes
    assert len(new_nodes) == 1
    assert list(new_nodes)[0].startswith("Thought:")
