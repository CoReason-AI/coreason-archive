from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.vector_store import VectorStore


@pytest.fixture
def base_thought() -> CachedThought:
    return CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=[],
        scope=MemoryScope.USER,
        scope_id="user_1",
        prompt_text="test",
        reasoning_trace="trace",
        final_response="resp",
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=[],
        is_stale=False,
    )


def test_urn_substring_no_match(base_thought: CachedThought) -> None:
    """
    Edge Case: Ensure partial URN matches do not trigger the stale flag.
    'urn:123' should not match 'urn:1234'.
    """
    store = VectorStore()
    thought = base_thought.model_copy(update={"id": uuid4(), "source_urns": ["urn:1234"]})
    store.add(thought)

    count = store.mark_stale_by_urn("urn:123")

    assert count == 0
    assert store.thoughts[0].is_stale is False


def test_urn_case_sensitivity(base_thought: CachedThought) -> None:
    """
    Edge Case: Ensure URN matching is strict/case-sensitive.
    'urn:ABC' should not match 'urn:abc'.
    """
    store = VectorStore()
    thought = base_thought.model_copy(update={"id": uuid4(), "source_urns": ["urn:abc"]})
    store.add(thought)

    count = store.mark_stale_by_urn("urn:ABC")

    assert count == 0
    assert store.thoughts[0].is_stale is False


def test_multiple_urns_interaction(base_thought: CachedThought) -> None:
    """
    Edge Case: Thoughts with multiple source URNs.
    If one source is invalid, the thought is stale.
    Subsequent invalidation of other sources should be handled gracefully.
    """
    store = VectorStore()
    thought = base_thought.model_copy(update={"id": uuid4(), "source_urns": ["urn:A", "urn:B"]})
    store.add(thought)

    # 1. Invalidate first URN
    count_a = store.mark_stale_by_urn("urn:A")
    assert count_a == 1
    assert store.thoughts[0].is_stale is True

    # 2. Invalidate second URN
    # Should find the thought, but it's already stale, so count should not increment (implementation detail check)
    # The implementation returns count of thoughts *marked* (changed state).
    count_b = store.mark_stale_by_urn("urn:B")
    assert count_b == 0
    assert store.thoughts[0].is_stale is True


@pytest.mark.asyncio
async def test_complex_retrieval_flow(base_thought: CachedThought) -> None:
    """
    Complex Scenario: Add thoughts -> Retrieve (fresh) -> Invalidate -> Retrieve (stale).
    Verifies that the stale state propagates to search results.
    """
    # Setup Archive
    store = VectorStore()
    graph = MagicMock()
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 1536  # Mock embedding

    archive = CoreasonArchive(vector_store=store, graph_store=graph, embedder=embedder)

    # Add thoughts
    t1 = await archive.add_thought(
        prompt="foo", response="bar", scope=MemoryScope.USER, scope_id="u1", user_id="u1", source_urns=["urn:doc:1"]
    )

    context = UserContext(user_id="u1", roles=[])

    # 1. First Retrieval
    results_fresh = await archive.retrieve("foo", context)
    assert len(results_fresh) > 0
    assert results_fresh[0][0].id == t1.id
    assert results_fresh[0][0].is_stale is False

    # 2. Invalidate Source
    archive.invalidate_source("urn:doc:1")

    # 3. Second Retrieval
    results_stale = await archive.retrieve("foo", context)
    assert len(results_stale) > 0
    assert results_stale[0][0].id == t1.id
    assert results_stale[0][0].is_stale is True


def test_invalidation_performance_simulation(base_thought: CachedThought) -> None:
    """
    Complex Scenario: Batch operation simulation.
    Store has mixed thoughts, only linked ones should be touched.
    """
    store = VectorStore()

    # Create 100 thoughts:
    # 50 linked to urn:target
    # 50 linked to urn:other

    target_urn = "urn:target"
    other_urn = "urn:other"

    for i in range(100):
        urns = [target_urn] if i < 50 else [other_urn]
        t = base_thought.model_copy(update={"id": uuid4(), "source_urns": urns})
        store.add(t)

    # Invalidate target
    count = store.mark_stale_by_urn(target_urn)

    assert count == 50

    # Verify correctness
    stale_count = sum(1 for t in store.thoughts if t.is_stale)
    assert stale_count == 50

    # Verify the correct ones are stale
    for i, t in enumerate(store.thoughts):
        if i < 50:
            assert t.is_stale is True, f"Thought {i} should be stale"
            assert target_urn in t.source_urns
        else:
            assert t.is_stale is False, f"Thought {i} should NOT be stale"
            assert other_urn in t.source_urns
