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
from coreason_archive.models import MemoryScope, GraphEdgeType
from coreason_archive.vector_store import VectorStore
from coreason_archive.graph_store import GraphStore
from coreason_archive.utils.stubs import StubEmbedder
from coreason_archive.utils.logger import logger

@pytest.mark.asyncio
async def test_synchronous_structural_ingestion_all_scopes() -> None:
    """
    Verifies that 'add_thought' synchronously creates the necessary structural edges
    (User->CREATED->Thought and Thought->BELONGS_TO->Scope) for ALL MemoryScope types,
    ensuring the graph is never disconnected even before background NLP runs.
    """

    # Scenarios to test: Scope Enum -> Expected Scope Node Prefix
    scenarios = [
        (MemoryScope.USER, "user_123", "User:user_123"),
        (MemoryScope.PROJECT, "proj_apollo", "Project:proj_apollo"),
        (MemoryScope.DEPARTMENT, "dept_rnd", "Department:dept_rnd"),
        (MemoryScope.CLIENT, "client_acme", "Client:client_acme"),
    ]

    for scope, scope_id, expected_scope_node in scenarios:
        logger.info(f"Testing synchronous ingestion for scope: {scope}")

        # Setup fresh components for each iteration to avoid ID collisions or state bleed
        v_store = VectorStore()
        g_store = GraphStore()
        embedder = StubEmbedder()
        # No extractor needed; we are testing the *structural* links that happen *before* extraction
        archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

        user_id = "test_user_main"

        # Action
        thought = await archive.add_thought(
            prompt="Test prompt",
            response="Test response",
            scope=scope,
            scope_id=scope_id,
            user_id=user_id
        )

        # Verification - Immediate Check (Synchronous)

        # 1. Check User -> CREATED -> Thought
        user_node = f"User:{user_id}"
        thought_node = f"Thought:{thought.id}"

        related_created = g_store.get_related_entities(user_node, relation=GraphEdgeType.CREATED, direction="outgoing")
        assert any(n == thought_node for n, _ in related_created), \
            f"Scope {scope}: Missing CREATED edge from {user_node} to {thought_node}"

        # 2. Check Thought -> BELONGS_TO -> Scope Entity
        related_belongs = g_store.get_related_entities(thought_node, relation=GraphEdgeType.BELONGS_TO, direction="outgoing")

        # We expect exactly one BELONGS_TO edge to the scope node
        # (unless there are other implicit ones, but here we only expect one structural one)
        found_scope_link = False
        for neighbor, _ in related_belongs:
            if neighbor == expected_scope_node:
                found_scope_link = True
                break

        assert found_scope_link, \
            f"Scope {scope}: Missing BELONGS_TO edge from {thought_node} to {expected_scope_node}. Found: {related_belongs}"

@pytest.mark.asyncio
async def test_synchronous_ingestion_null_ids() -> None:
    """
    Verifies robust handling of empty/None user_id or scope_id.
    They should be sanitized to 'Unknown' rather than crashing or creating invalid nodes.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # Action with empty strings (which might happen in some upstream error cases)
    thought = await archive.add_thought(
        prompt="Test",
        response="Response",
        scope=MemoryScope.USER,
        scope_id="", # Empty
        user_id=""   # Empty
    )

    thought_node = f"Thought:{thought.id}"

    # Verify User:Unknown -> CREATED -> Thought
    related_created = g_store.get_related_entities("User:Unknown", relation=GraphEdgeType.CREATED, direction="outgoing")
    assert any(n == thought_node for n, _ in related_created), "Missing link from User:Unknown"

    # Verify Thought -> BELONGS_TO -> User:Unknown
    related_belongs = g_store.get_related_entities(thought_node, relation=GraphEdgeType.BELONGS_TO, direction="outgoing")
    assert any(n == "User:Unknown" for n, _ in related_belongs), "Missing link to User:Unknown scope"

@pytest.mark.asyncio
async def test_synchronous_ingestion_special_characters() -> None:
    """
    Verifies that IDs containing special characters (especially colons used as separators)
    are handled correctly and do not corrupt the graph node parsing.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # User ID with colon (dangerous) and unicode
    user_id = "user:alice@example.com:mega_admin"
    # Project ID with spaces and symbols
    scope_id = "Project #1: Top Secret"

    thought = await archive.add_thought(
        prompt="Test",
        response="Response",
        scope=MemoryScope.PROJECT,
        scope_id=scope_id,
        user_id=user_id
    )

    # Expected Node Names
    # The logic is f"User:{user_id}" -> "User:user:alice@example.com:mega_admin"
    # GraphStore splits on *first* colon. Type="User", Value="user:alice@example.com:mega_admin".
    expected_user_node = f"User:{user_id}"
    expected_scope_node = f"Project:{scope_id}"
    thought_node = f"Thought:{thought.id}"

    # Verify Nodes Exist
    assert g_store.graph.has_node(expected_user_node)
    assert g_store.graph.has_node(expected_scope_node)

    # Verify User -> CREATED -> Thought
    related_created = g_store.get_related_entities(expected_user_node, relation=GraphEdgeType.CREATED, direction="outgoing")
    assert any(n == thought_node for n, _ in related_created)

    # Verify Thought -> BELONGS_TO -> Project
    related_belongs = g_store.get_related_entities(thought_node, relation=GraphEdgeType.BELONGS_TO, direction="outgoing")
    assert any(n == expected_scope_node for n, _ in related_belongs)

@pytest.mark.asyncio
async def test_complex_topology_hub_and_spoke() -> None:
    """
    Complex Scenario: Hub-and-Spoke Topology.
    A single User adds multiple thoughts to the same Project.
    Verifies that the User node acts as a hub (multiple outgoing edges)
    and the Project node acts as a sink (multiple incoming edges).
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    user_id = "hub_user"
    project_id = "sink_project"

    thoughts = []
    # Add 5 thoughts
    for i in range(5):
        t = await archive.add_thought(
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            scope=MemoryScope.PROJECT,
            scope_id=project_id,
            user_id=user_id
        )
        thoughts.append(t)

    user_node = f"User:{user_id}"
    project_node = f"Project:{project_id}"

    # Verify User has 5 outgoing CREATED edges
    created_edges = g_store.get_related_entities(user_node, relation=GraphEdgeType.CREATED, direction="outgoing")
    assert len(created_edges) == 5

    # Verify all thoughts are in the list
    thought_nodes = {f"Thought:{t.id}" for t in thoughts}
    found_nodes = {n for n, _ in created_edges}
    assert thought_nodes == found_nodes

    # Verify Project has 5 incoming BELONGS_TO edges
    # Note: get_related_entities(..., direction="incoming") returns (neighbor, relation)
    # where neighbor is the source of the edge (the Thought).
    belongs_edges = g_store.get_related_entities(project_node, relation=GraphEdgeType.BELONGS_TO, direction="incoming")
    assert len(belongs_edges) == 5

    found_sources = {n for n, _ in belongs_edges}
    assert thought_nodes == found_sources

@pytest.mark.asyncio
async def test_complex_topology_scope_switching() -> None:
    """
    Complex Scenario: Scope Switching.
    A single User adds thoughts to different scopes (Project A, Dept B).
    Verifies that the User connects to all thoughts, but thoughts connect to different scopes.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = StubEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    user_id = "multi_scope_user"

    # 1. Add to Project A
    t1 = await archive.add_thought("p1", "r1", MemoryScope.PROJECT, "ProjA", user_id)
    # 2. Add to Dept B
    t2 = await archive.add_thought("p2", "r2", MemoryScope.DEPARTMENT, "DeptB", user_id)

    user_node = f"User:{user_id}"
    proj_node = "Project:ProjA"
    dept_node = "Department:DeptB"
    t1_node = f"Thought:{t1.id}"
    t2_node = f"Thought:{t2.id}"

    # Verify User connects to both
    created = g_store.get_related_entities(user_node, relation=GraphEdgeType.CREATED, direction="outgoing")
    created_ids = {n for n, _ in created}
    assert t1_node in created_ids
    assert t2_node in created_ids

    # Verify T1 connects ONLY to Project A
    t1_scope = g_store.get_related_entities(t1_node, relation=GraphEdgeType.BELONGS_TO, direction="outgoing")
    assert len(t1_scope) == 1
    assert t1_scope[0][0] == proj_node

    # Verify T2 connects ONLY to Dept B
    t2_scope = g_store.get_related_entities(t2_node, relation=GraphEdgeType.BELONGS_TO, direction="outgoing")
    assert len(t2_scope) == 1
    assert t2_scope[0][0] == dept_node
