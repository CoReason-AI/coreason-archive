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
from typing import Any, Coroutine, Set

from coreason_archive.interfaces import TaskRunner
from coreason_archive.utils.logger import logger


class AsyncIOTaskRunner(TaskRunner):
    """
    Default implementation of TaskRunner using asyncio.create_task.
    Manages a set of strong references to running tasks to prevent garbage collection.
    """

    def __init__(self) -> None:
        self._background_tasks: Set[asyncio.Task[Any]] = set()

    def run(self, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Schedules the coroutine and tracks the task.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        logger.debug(f"Scheduled background task: {task.get_name()}")

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """
        Callback to remove the task from the tracking set when done.
        """
        try:
            self._background_tasks.discard(task)
            # Retrieve exception to avoid "Task exception was never retrieved" warning
            exc = task.exception()
            if exc:
                logger.error(f"Background task failed: {exc}")
        except asyncio.CancelledError:
            logger.warning("Background task was cancelled.")
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
