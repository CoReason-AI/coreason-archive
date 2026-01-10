from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_archive.models import CachedThought, MemoryScope


def test_cached_thought_valid_creation() -> None:
    """Test that a CachedThought can be created with valid data."""
    thought = CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=["Project:Apollo", "Drug:X"],
        scope=MemoryScope.DEPARTMENT,
        scope_id="dept_oncology",
        prompt_text="What is the dosing protocol?",
        reasoning_trace="Step 1: Check database...",
        final_response="Use 50mg.",
        source_urns=["urn:doc:123"],
        created_at=datetime.now(),
        ttl_seconds=3600,
        access_roles=["role:oncologist"],
    )
    assert thought.scope == MemoryScope.DEPARTMENT
    assert len(thought.vector) == 1536
    assert thought.scope_id == "dept_oncology"


def test_cached_thought_invalid_scope() -> None:
    """Test that creating a CachedThought with an invalid scope fails."""
    with pytest.raises(ValidationError):
        CachedThought(
            id=uuid4(),
            vector=[0.1] * 1536,
            entities=[],
            scope="INVALID_SCOPE",  # type: ignore
            scope_id="test",
            prompt_text="test",
            reasoning_trace="test",
            final_response="test",
            source_urns=[],
            created_at=datetime.now(),
            ttl_seconds=100,
            access_roles=[],
        )


def test_cached_thought_missing_fields() -> None:
    """Test that missing required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        CachedThought(
            id=uuid4(),
            # Missing vector
            entities=[],
            scope=MemoryScope.USER,
            scope_id="user_123",
            prompt_text="test",
            reasoning_trace="test",
            final_response="test",
            source_urns=[],
            created_at=datetime.now(),
            ttl_seconds=100,
            access_roles=[],  # type: ignore[call-arg]
        )


def test_memory_scope_enum() -> None:
    """Test MemoryScope enum values."""
    assert MemoryScope.USER == "USER"
    assert MemoryScope.PROJECT == "PROJECT"
    assert MemoryScope.DEPARTMENT == "DEPT"
    assert MemoryScope.CLIENT == "CLIENT"
