import asyncio
from typing import List

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor
from coreason_archive.models import MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    def __init__(self, should_fail: bool = False, delay: float = 0.0) -> None:
        self.should_fail = should_fail
        self.delay = delay

    async def extract(self, text: str) -> List[str]:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise ValueError("Intentional Extraction Failure")
        return ["Project:AsyncTest"]


@pytest.mark.asyncio
async def test_concurrent_ingestion() -> None:
    """
    Test adding multiple thoughts rapidly to ensure background tasks
    handle concurrency correctly without race conditions.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    # Add a small delay to ensure tasks overlap
    extractor = MockEntityExtractor(delay=0.01)

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # 1. Fire off 50 additions concurrently
    tasks = []
    for i in range(50):
        tasks.append(
            archive.add_thought(
                prompt=f"p{i}",
                response=f"r{i}",
                scope=MemoryScope.USER,
                scope_id="u1",
                user_id="u1",
            )
        )

    await asyncio.gather(*tasks)

    # 2. Wait for background processing
    # We poll until tasks are done, or rely on gather if we had access.
    # We can access archive._background_tasks directly for testing.
    if archive._background_tasks:
        await asyncio.gather(*archive._background_tasks)

    # 3. Validation
    # VectorStore should have 50 items
    assert len(v_store.thoughts) == 50

    # GraphStore should have 50 Thought nodes + 1 Project node
    # Each thought extraction adds "Project:AsyncTest"
    # And "Thought:ID"
    # Total nodes = 50 (Thoughts) + 1 (Project) = 51
    # Check if "Project:AsyncTest" exists
    assert g_store.graph.has_node("Project:AsyncTest")

    # Check edges count. Each thought has 2 edges (Thought <-> Project)
    # Total edges = 100
    assert g_store.graph.number_of_edges() == 100


@pytest.mark.asyncio
async def test_task_cleanup() -> None:
    """
    Verify that _background_tasks set is cleaned up after tasks complete.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Add a thought
    await archive.add_thought("p", "r", MemoryScope.USER, "u1", "u1")

    # Assert task exists initially (or quickly grab it)
    # Since add_thought is async but create_task is immediate,
    # the task might be in set but could finish very fast.
    # Let's ensure we wait for it.
    assert len(archive._background_tasks) > 0 or len(v_store.thoughts) == 1

    # Wait for completion
    if archive._background_tasks:
        await asyncio.gather(*archive._background_tasks)

    # Allow event loop one more tick for the done callback to fire
    await asyncio.sleep(0)

    # Assert set is empty
    assert len(archive._background_tasks) == 0


@pytest.mark.asyncio
async def test_mixed_failure_success() -> None:
    """
    Test a batch where some extractions fail.
    Ensures system stability and partial success.
    """
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()

    # We need an extractor that fails for specific inputs
    # But current interface is generic.
    # Let's mock the extract method dynamically.
    extractor = MockEntityExtractor()
    # We'll use a side effect that checks the text
    original_extract = extractor.extract

    async def side_effect(text: str) -> List[str]:
        if "FAIL" in text:
            raise ValueError("Boom")
        return await original_extract(text)

    extractor.extract = side_effect  # type: ignore

    archive = CoreasonArchive(v_store, g_store, embedder, extractor)

    # Add 1 success, 1 fail, 1 success
    await archive.add_thought("p1", "SUCCESS 1", MemoryScope.USER, "u1", "u1")
    await archive.add_thought("p2", "FAIL THIS", MemoryScope.USER, "u1", "u1")
    await archive.add_thought("p3", "SUCCESS 2", MemoryScope.USER, "u1", "u1")

    # Wait for tasks
    if archive._background_tasks:
        # gather will raise if any task raised, unless return_exceptions=True
        await asyncio.gather(*archive._background_tasks, return_exceptions=True)

    # Verify VectorStore has all 3 (since vectorization happens before extraction)
    assert len(v_store.thoughts) == 3

    # Verify GraphStore
    # Should contain nodes for SUCCESS 1 and SUCCESS 2 thoughts.
    # And the Project entity.
    # Should NOT contain node for FAIL thought (or at least no edges if we fail before adding).
    # Logic:
    # 1. extract() -> 2. add thought node -> 3. add entities -> 4. add edges.
    # If extract fails, steps 2-4 don't run.
    # So FAIL thought should NOT have a graph node.

    # We need to find the IDs to check.
    t_fail = next(t for t in v_store.thoughts if "FAIL" in t.reasoning_trace)
    t_succ1 = next(t for t in v_store.thoughts if "SUCCESS 1" in t.reasoning_trace)

    assert not g_store.graph.has_node(f"Thought:{t_fail.id}")
    assert g_store.graph.has_node(f"Thought:{t_succ1.id}")
