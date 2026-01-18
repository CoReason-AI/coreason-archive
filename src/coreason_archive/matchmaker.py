# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from coreason_archive.models import CachedThought


class MatchStrategy(str, Enum):
    EXACT_HIT = "EXACT_HIT"
    SEMANTIC_HINT = "SEMANTIC_HINT"
    STANDARD_RETRIEVAL = "STANDARD_RETRIEVAL"
    ENTITY_HOP = "ENTITY_HOP"


class SearchResult(BaseModel):
    """
    Represents the result of a smart lookup.
    """

    strategy: MatchStrategy = Field(..., description="The strategy used (Exact, Hint, etc.)")
    thought: Optional[CachedThought] = Field(None, description="The primary thought found (if any)")
    score: float = Field(0.0, description="The similarity score")
    content: Dict[str, Any] = Field(..., description="The payload to be used by the agent")
