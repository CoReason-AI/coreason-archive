# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from datetime import datetime, timezone
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
        owner_id="user_oncologist",
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
            owner_id="user_test",
            source_urns=[],
            created_at=datetime.now(),
            ttl_seconds=100,
            access_roles=[],
        )


def test_cached_thought_missing_fields() -> None:
    """Test that missing required fields raises ValidationError."""
    with pytest.raises(ValidationError):
        CachedThought(  # type: ignore[call-arg]
            id=uuid4(),
            # Missing vector
            entities=[],
            scope=MemoryScope.USER,
            scope_id="user_123",
            prompt_text="test",
            reasoning_trace="test",
            final_response="test",
            owner_id="user_123",
            source_urns=[],
            created_at=datetime.now(),
            ttl_seconds=100,
            access_roles=[],
        )


def test_memory_scope_enum() -> None:
    """Test MemoryScope enum values."""
    assert MemoryScope.USER == "USER"
    assert MemoryScope.PROJECT == "PROJECT"
    assert MemoryScope.DEPARTMENT == "DEPT"
    assert MemoryScope.CLIENT == "CLIENT"


def test_cached_thought_boundary_values() -> None:
    """Test edge cases with empty lists and zero values."""
    thought = CachedThought(
        id=uuid4(),
        vector=[],  # Empty vector
        entities=[],  # Empty entities
        scope=MemoryScope.USER,
        scope_id="user_empty",
        prompt_text="",  # Empty string
        reasoning_trace="",
        final_response="",
        owner_id="user_empty",
        source_urns=[],  # Empty sources
        created_at=datetime.now(),
        ttl_seconds=0,  # Zero TTL
        access_roles=[],  # Public access (implied)
    )
    assert thought.vector == []
    assert thought.entities == []
    assert thought.ttl_seconds == 0
    assert thought.prompt_text == ""


def test_cached_thought_complex_content() -> None:
    """Test with complex content including Markdown and Unicode."""
    complex_trace = """
    # Reasoning Trace
    1. Analysis of `Concept:X` 🧪
    2. Calculation: $E = mc^2$
    3. Conclusion: ✅ Approved
    """
    thought = CachedThought(
        id=uuid4(),
        vector=[0.5, -0.5, 0.0],
        entities=["Concept:X", "Symbol:∑"],
        scope=MemoryScope.PROJECT,
        scope_id="proj_α",
        prompt_text="Analyze 🧪 scenario",
        reasoning_trace=complex_trace,
        final_response="Approved ✅",
        owner_id="user_admin",
        source_urns=["urn:doc:日本語"],
        created_at=datetime.now(),
        ttl_seconds=3600,
        access_roles=["role:admin"],
    )
    assert "🧪" in thought.reasoning_trace
    assert "✅" in thought.final_response
    assert "urn:doc:日本語" in thought.source_urns[0]


def test_cached_thought_serialization_roundtrip() -> None:
    """Test JSON serialization and deserialization roundtrip."""
    original_id = uuid4()
    # Use timezone-aware datetime for precise comparison
    created_at = datetime.now(timezone.utc)

    thought = CachedThought(
        id=original_id,
        vector=[0.1, 0.2, 0.3],
        entities=["Entity:1"],
        scope=MemoryScope.CLIENT,
        scope_id="client_corp",
        prompt_text="Test Prompt",
        reasoning_trace="Thinking...",
        final_response="Answer",
        owner_id="user_client",
        source_urns=["urn:1"],
        created_at=created_at,
        ttl_seconds=60,
        access_roles=["role:read"],
    )

    # Serialize to JSON
    json_data = thought.model_dump_json()

    # Deserialize back to object
    restored_thought = CachedThought.model_validate_json(json_data)

    assert restored_thought.id == original_id
    assert restored_thought.scope == MemoryScope.CLIENT
    assert restored_thought.vector == [0.1, 0.2, 0.3]
    # Pydantic usually handles datetime roundtrip well, ensuring equality
    assert restored_thought.created_at == created_at
