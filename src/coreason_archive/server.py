# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, List, Optional, cast

from coreason_identity.models import UserContext
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from coreason_archive.archive import CoreasonArchive
from coreason_archive.main import init_archive, save_archive
from coreason_archive.matchmaker import SearchResult
from coreason_archive.models import MemoryScope
from coreason_archive.utils.logger import logger


class ThoughtRequest(BaseModel):
    prompt: str
    response: str
    user_id: str
    scope: str = "USER"
    project_id: Optional[str] = None
    source_urns: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str
    context: UserContext


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    logger.info("Initializing Coreason Archive Server...")
    # Initialize the archive (loads data)
    archive = init_archive()
    app.state.archive = archive
    yield
    # Shutdown
    logger.info("Saving Coreason Archive Server...")
    save_archive(archive)


app = FastAPI(lifespan=lifespan, title="CoReason Archive API")


def get_archive(request: Request) -> CoreasonArchive:
    if not hasattr(request.app.state, "archive"):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Archive not initialized")
    return cast(CoreasonArchive, request.app.state.archive)


@app.post("/thoughts", status_code=status.HTTP_201_CREATED)
async def add_thought(
    req: ThoughtRequest,
    archive: CoreasonArchive = Depends(get_archive),  # noqa: B008
) -> dict[str, str]:
    # Validate scope early
    try:
        mem_scope = MemoryScope(req.scope)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope: {req.scope}. Valid scopes: {[s.value for s in MemoryScope]}",
        ) from e

    try:
        scope_id = req.user_id if mem_scope == MemoryScope.USER else (req.project_id or "default")

        # In server mode, user_context usually comes from auth middleware.
        # Here we construct it from input or minimal.
        groups = []
        if mem_scope in (MemoryScope.PROJECT, MemoryScope.DEPARTMENT, MemoryScope.CLIENT):
            if scope_id != "default":
                groups.append(scope_id)

        user_context = UserContext(
            user_id=req.user_id,
            email=f"{req.user_id}@coreason.ai",
            groups=groups,
        )

        thought = await archive.add_thought(
            prompt=req.prompt,
            response=req.response,
            scope=mem_scope,
            scope_id=scope_id,
            user_context=user_context,
            source_urns=req.source_urns,
            access_roles=[],  # Default
        )
        return {"status": "success", "thought_id": str(thought.id)}
    except ValueError as e:
        # Catch specific ValueErrors from logic (if any) that should be 400
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error adding thought")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.post("/search")
async def search(
    req: SearchRequest,
    archive: CoreasonArchive = Depends(get_archive),  # noqa: B008
) -> SearchResult:
    try:
        result = await archive.smart_lookup(req.query, req.context)
        return result
    except Exception as e:
        logger.exception("Error searching")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.get("/health")
async def health(archive: CoreasonArchive = Depends(get_archive)) -> dict[str, Any]:  # noqa: B008
    v_size = len(archive.vector_store.thoughts) if hasattr(archive.vector_store, "thoughts") else "unknown"
    g_nodes = len(archive.graph_store.graph.nodes) if hasattr(archive.graph_store, "graph") else "unknown"
    return {
        "status": "healthy",
        "vector_store_size": v_size,
        "graph_nodes": g_nodes,
    }
