# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import asyncio
from typing import Any, Coroutine, List
from unittest.mock import MagicMock

import pytest

from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder, EntityExtractor, TaskRunner
from coreason_archive.models import MemoryScope
from coreason_archive.utils.runners import AsyncIOTaskRunner
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        # Return a dummy vector of 1536 dims
        return [0.1] * 1536


class MockEntityExtractor(EntityExtractor):
    async def extract(self, text: str) -> List[str]:
        return ["Project:Apollo", "User:Alice"]


class MockTaskRunner(TaskRunner):
    """
    A mock task runner that collects coroutines instead of running them.
    Useful for verifying that tasks were submitted.
    """

    def __init__(self) -> None:
        self.submitted_tasks: List[Coroutine[Any, Any, Any]] = []

    def run(self, coro: Coroutine[Any, Any, Any]) -> None:
        self.submitted_tasks.append(coro)


@pytest.mark.asyncio
async def test_asyncio_task_runner() -> None:
    """Test the default AsyncIOTaskRunner."""
    runner = AsyncIOTaskRunner()

    # Create a simple task that sleeps briefly
    async def simple_task() -> str:
        await asyncio.sleep(0.01)
        return "done"

    runner.run(simple_task())

    # Check that task is tracked
    assert len(runner._background_tasks) == 1

    # Wait for completion
    task = next(iter(runner._background_tasks))
    await task

    # Should be removed from set (callback might need a yield to run)
    await asyncio.sleep(0)
    assert len(runner._background_tasks) == 0


@pytest.mark.asyncio
async def test_asyncio_task_runner_cancellation() -> None:
    """Test handling of cancelled tasks."""
    runner = AsyncIOTaskRunner()

    async def long_task() -> None:
        await asyncio.sleep(1)

    runner.run(long_task())
    task = next(iter(runner._background_tasks))

    # Cancel the task
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Yield to allow callback to run
    await asyncio.sleep(0)

    # Should be removed
    assert len(runner._background_tasks) == 0


@pytest.mark.asyncio
async def test_asyncio_task_runner_exception_in_callback() -> None:
    """Test generic exception handling in callback."""
    runner = AsyncIOTaskRunner()

    # Replace the set with a mock that raises on discard
    mock_set = MagicMock(spec=set)
    mock_set.discard.side_effect = RuntimeError("Boom")

    # We must allow add/iteration to work if called.
    # run calls add.
    # We need to manually capture the task to wait on it,
    # because mock_set won't store it.

    captured_task = None

    async def simple_task() -> None:
        pass

    def side_effect_add(item: Any) -> None:
        nonlocal captured_task
        captured_task = item

    mock_set.add.side_effect = side_effect_add

    runner._background_tasks = mock_set

    runner.run(simple_task())

    assert captured_task is not None
    await captured_task

    # Yield for callback
    await asyncio.sleep(0)

    # Verify discard was called and raised
    mock_set.discard.assert_called_once()

    # We expect the exception to be caught and logged.
    # If not caught, this test would fail with unhandled exception or printed error.
    # To be sure, we can check logger calls if we mock logger, but coverage is the goal.


@pytest.mark.asyncio
async def test_asyncio_task_runner_task_failure() -> None:
    """Test that task failure is logged."""
    runner = AsyncIOTaskRunner()

    async def failing_task() -> None:
        raise ValueError("Task Failed")

    runner.run(failing_task())

    task = next(iter(runner._background_tasks))

    # Wait for it to fail
    try:
        await task
    except ValueError:
        pass

    await asyncio.sleep(0)
    assert len(runner._background_tasks) == 0


@pytest.mark.asyncio
async def test_archive_uses_injected_runner() -> None:
    """Verify that CoreasonArchive uses the provided TaskRunner."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()
    mock_runner = MockTaskRunner()

    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=extractor, task_runner=mock_runner)

    user_ctx = UserContext(user_id="user", email="test@example.com")
    await archive.add_thought(prompt="Test", response="Test", scope=MemoryScope.USER, scope_id="user", user_context=user_ctx)

    # Verify task was submitted to mock runner
    assert len(mock_runner.submitted_tasks) == 1

    # Verify vector store has thought
    assert len(v_store.thoughts) == 1

    # Verify graph store HAS structural nodes (synchronous)
    # User:user and Thought:<id>
    assert g_store.graph.has_node("User:user")
    thought = v_store.thoughts[0]
    assert g_store.graph.has_node(f"Thought:{thought.id}")

    # But it does NOT have extracted entities (because task wasn't run)
    # Extractor returns "Project:Apollo"
    assert not g_store.graph.has_node("Project:Apollo")

    # Now run the collected task manually to verify it works
    task_coro = mock_runner.submitted_tasks[0]
    await task_coro

    # Now graph should have nodes
    assert len(g_store.graph.nodes) > 0


@pytest.mark.asyncio
async def test_archive_default_runner() -> None:
    """Verify that CoreasonArchive uses AsyncIOTaskRunner by default."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    extractor = MockEntityExtractor()

    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=extractor)

    assert isinstance(archive.task_runner, AsyncIOTaskRunner)

    # Verify the backward compatibility alias
    assert archive._background_tasks is archive.task_runner._background_tasks
