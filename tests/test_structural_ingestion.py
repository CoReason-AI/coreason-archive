import pytest
from uuid import uuid4
from coreason_archive.archive import CoreasonArchive
from coreason_archive.models import MemoryScope, GraphEdgeType
from coreason_archive.vector_store import VectorStore
from coreason_archive.graph_store import GraphStore
from coreason_archive.utils.stubs import StubEmbedder
from coreason_archive.utils.logger import logger

@pytest.mark.asyncio
async def test_synchronous_structural_ingestion_all_scopes():
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
async def test_synchronous_ingestion_null_ids():
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
