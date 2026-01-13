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
import random
from typing import Set

import pytest

from coreason_archive.utils.runners import AsyncIOTaskRunner


@pytest.mark.asyncio
async def test_concurrent_load() -> None:
    """
    Complex Scenario: "Thundering Herd"
    Submit many tasks concurrently with random durations.
    Verify that all tasks execute and the internal tracking set is cleaned up.
    """
    runner = AsyncIOTaskRunner()
    task_count = 50
    completed_tasks: Set[int] = set()

    async def worker(idx: int) -> None:
        # Sleep random amount to ensure out-of-order completion
        await asyncio.sleep(random.uniform(0.001, 0.01))
        completed_tasks.add(idx)

    # Submit all tasks
    for i in range(task_count):
        runner.run(worker(i))

    # Assert they are tracked
    # Note: Some might complete instantly, but unlikely all.
    # We can't strictly assert len == task_count unless we control the loop,
    # but we can assert > 0.
    assert len(runner._background_tasks) > 0

    # Wait for all tasks to finish
    # We clone the set because it changes during iteration
    all_tasks = list(runner._background_tasks)
    await asyncio.gather(*all_tasks)

    # Yield to allow callbacks to run
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # Double yield often needed for callback propagation

    # Verify Logic
    assert len(completed_tasks) == task_count
    assert len(runner._background_tasks) == 0


@pytest.mark.asyncio
async def test_mixed_results() -> None:
    """
    Complex Scenario: "Chaos Engineering"
    Submit a batch containing:
    - Successes
    - Failures (Exceptions)
    - Cancellations
    Verify robust cleanup.
    """
    runner = AsyncIOTaskRunner()

    # Define behaviors
    async def success_task() -> str:
        await asyncio.sleep(0.001)
        return "ok"

    async def fail_task() -> None:
        await asyncio.sleep(0.001)
        raise ValueError("Intentional Failure")

    async def long_task() -> None:
        await asyncio.sleep(1.0)  # Will be cancelled

    # Launch
    runner.run(success_task())
    runner.run(fail_task())
    runner.run(long_task())

    # Identification
    tasks = list(runner._background_tasks)
    assert len(tasks) == 3

    # Cancel the long task
    # We need to find it. But we don't have handles returned by run().
    # We inspect the set.
    # NOTE: In a real app, if we needed to cancel specific tasks,
    # run() should return the task. But Protocol returns None.
    # So we iterate and cancel the one that is still pending/long?
    # Or just cancel all remaining after a short sleep.

    # Let's wait for success/fail to likely finish
    await asyncio.sleep(0.01)

    # Now find the running one
    for t in runner._background_tasks:
        if not t.done():
            t.cancel()

    # Wait for everything
    # We use return_exceptions=True to avoid gathering crashing
    all_tasks = list(runner._background_tasks)
    if all_tasks:
        await asyncio.gather(*all_tasks, return_exceptions=True)

    await asyncio.sleep(0)

    # Verify cleanup
    # Even failed and cancelled tasks should be removed from the set
    assert len(runner._background_tasks) == 0


@pytest.mark.asyncio
async def test_recursive_submission() -> None:
    """
    Complex Scenario: "Chain Reaction"
    A background task submits another background task.
    Verify both run and cleanup.
    """
    runner = AsyncIOTaskRunner()

    result_order = []

    async def child_task() -> None:
        result_order.append("child")

    async def parent_task() -> None:
        result_order.append("parent_start")
        runner.run(child_task())
        result_order.append("parent_end")

    runner.run(parent_task())

    # Wait for parent
    # Wait for *any* tasks in runner until empty

    # Simple polling for test
    for _ in range(10):
        if len(runner._background_tasks) == 0 and len(result_order) >= 3:
            break
        await asyncio.sleep(0.01)

    # Wait explicitly for tasks if any remain (child might be running)
    tasks = list(runner._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)

    await asyncio.sleep(0)

    assert len(runner._background_tasks) == 0
    assert "parent_start" in result_order
    assert "child" in result_order
    # Child might finish before or after parent_end depending on scheduling
