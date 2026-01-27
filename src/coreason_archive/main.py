# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.extractors import RegexEntityExtractor
from coreason_archive.graph_store import GraphStore
from coreason_archive.models import MemoryScope
from coreason_archive.utils.logger import logger
from coreason_archive.utils.stubs import StubEmbedder
from coreason_archive.vector_store import VectorStore

DATA_DIR = Path("data")
VECTOR_STORE_PATH = DATA_DIR / "vector_store.json"
GRAPH_STORE_PATH = DATA_DIR / "graph_store.json"


def ensure_data_dir() -> None:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True)


def init_archive() -> CoreasonArchive:
    ensure_data_dir()

    # Initialize Stores
    v_store = VectorStore()
    if VECTOR_STORE_PATH.exists():
        v_store.load(VECTOR_STORE_PATH)

    g_store = GraphStore()
    if GRAPH_STORE_PATH.exists():
        g_store.load(GRAPH_STORE_PATH)

    # Initialize Components
    embedder = StubEmbedder()
    extractor = RegexEntityExtractor()

    return CoreasonArchive(v_store, g_store, embedder, extractor)


def save_archive(archive: CoreasonArchive) -> None:
    archive.vector_store.save(VECTOR_STORE_PATH)
    archive.graph_store.save(GRAPH_STORE_PATH)


async def add_thought(
    archive: CoreasonArchive,
    prompt: str,
    response: str,
    user_id: str,
    scope: str = "USER",
    project: Optional[str] = None,
) -> None:
    mem_scope = MemoryScope(scope)
    scope_id = user_id if mem_scope == MemoryScope.USER else (project or "default")

    # Construct minimal UserContext for CLI
    # If scope is PROJECT, user must be in that group for the check to pass
    groups = []
    if mem_scope in (MemoryScope.PROJECT, MemoryScope.DEPARTMENT, MemoryScope.CLIENT):
        groups.append(scope_id)

    # CLI doesn't provide email, so we mock it
    user_context = UserContext(user_id=user_id, email=f"{user_id}@coreason.ai", groups=groups)

    logger.info(f"Adding thought for user {user_id} in scope {scope}...")

    thought = await archive.add_thought(
        prompt=prompt,
        response=response,
        scope=mem_scope,
        scope_id=scope_id,
        user_context=user_context,
        access_roles=[],  # Default public within scope
    )

    # Wait for background extraction to finish for CLI purposes
    # In a real app, this would happen in background, but CLI exits immediately
    if archive.task_runner:
        # We need to access the task runner's active tasks if we want to await them.
        # But AsyncIOTaskRunner creates tasks on the loop.
        # We can just wait a bit or force processing if possible.
        # Since add_thought fires and forgets, we rely on the fact that we are in an async function
        # and the task is scheduled. We should probably gather all pending tasks.
        # However, AsyncIOTaskRunner stores them in _background_tasks.
        if hasattr(archive.task_runner, "_background_tasks"):
            tasks = archive.task_runner._background_tasks
            if tasks:
                await asyncio.gather(*tasks)

    print(f"Thought added successfully with ID: {thought.id}")
    print(f"Entities found: {thought.entities}")


async def search_thought(archive: CoreasonArchive, query: str, user_id: str, project: Optional[str] = None) -> None:
    logger.info(f"Searching for '{query}' as user {user_id}...")

    # Construct Context
    # For CLI demo, we assume user has access to the project if provided
    groups = []
    if project:
        groups.append(project)

    context = UserContext(user_id=user_id, email=f"{user_id}@coreason.ai", groups=groups)

    result = await archive.smart_lookup(query, context)

    print("\nSearch Result:")
    print(f"Strategy: {result.strategy.value}")
    print(f"Score: {result.score:.4f}")
    print(f"Content: {json.dumps(result.content, indent=2)}")


async def run_async_main() -> None:
    parser = argparse.ArgumentParser(description="CoReason Archive CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add Command
    add_parser = subparsers.add_parser("add", help="Add a new thought")
    add_parser.add_argument("--prompt", required=True, help="The user prompt")
    add_parser.add_argument("--response", required=True, help="The system response")
    add_parser.add_argument("--user", required=True, help="User ID")
    add_parser.add_argument("--scope", default="USER", choices=[s.value for s in MemoryScope], help="Memory Scope")
    add_parser.add_argument("--project", help="Project ID (required if scope is PROJECT)")

    # Search Command
    search_parser = subparsers.add_parser("search", help="Search for thoughts")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--user", required=True, help="User ID")
    search_parser.add_argument("--project", help="Active Project Context")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    archive = init_archive()

    try:
        if args.command == "add":
            if args.scope == "PROJECT" and not args.project:
                print("Error: --project is required for PROJECT scope.")
                return
            await add_thought(archive, args.prompt, args.response, args.user, args.scope, args.project)

        elif args.command == "search":
            await search_thought(archive, args.query, args.user, args.project)

    finally:
        save_archive(archive)


def main() -> None:
    asyncio.run(run_async_main())


if __name__ == "__main__":  # pragma: no cover
    main()
