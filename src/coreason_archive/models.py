# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from datetime import datetime
from enum import Enum
from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    USER = "USER"
    PROJECT = "PROJECT"
    DEPARTMENT = "DEPT"
    CLIENT = "CLIENT"


class GraphEdgeType(str, Enum):
    CREATED = "CREATED"
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"


class CachedThought(BaseModel):
    """
    Represents a single cognitive state or reasoning trace stored in the archive.
    """

    id: UUID = Field(..., description="Unique identifier for the thought")

    # Neuro-Symbolic Data
    vector: List[float] = Field(..., description="1536-dim embedding of the thought")
    entities: List[str] = Field(..., description="List of entities extracted (e.g., 'Project:Apollo')")

    # Hierarchy
    scope: MemoryScope = Field(..., description="The scope level of this memory")
    scope_id: str = Field(..., description="Identifier for the scope (e.g., 'dept_oncology')")

    # Content
    prompt_text: str = Field(..., description="The original prompt that generated this thought")
    reasoning_trace: str = Field(..., description="The step-by-step reasoning process (The 'How')")
    final_response: str = Field(..., description="The final answer or conclusion (The 'What')")

    # Metadata
    source_urns: List[str] = Field(..., description="Links to source documents in MCP")
    is_stale: bool = Field(default=False, description="Flag indicating if the source information is outdated")
    created_at: datetime = Field(..., description="Timestamp of creation")
    ttl_seconds: int = Field(..., description="Time-to-live in seconds for decay calculation")
    access_roles: List[str] = Field(..., description="RBAC claims required to access this thought")
