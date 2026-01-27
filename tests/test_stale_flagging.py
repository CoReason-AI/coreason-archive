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
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.vector_store import VectorStore


@pytest.fixture
def sample_thought() -> CachedThought:
    return CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=[],
        scope=MemoryScope.USER,
        scope_id="user_1",
        prompt_text="test prompt",
        reasoning_trace="test reasoning",
        final_response="test response",
        owner_id="user_1",
        source_urns=["urn:mcp:doc:123"],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=[],
        is_stale=False,
    )


def test_thought_defaults_not_stale() -> None:
    """Test that a new thought is not stale by default."""
    thought = CachedThought(
        id=uuid4(),
        vector=[0.1] * 1536,
        entities=[],
        scope=MemoryScope.USER,
        scope_id="user_1",
        prompt_text="p",
        reasoning_trace="r",
        final_response="f",
        owner_id="user_1",
        source_urns=[],
        created_at=datetime.now(timezone.utc),
        ttl_seconds=3600,
        access_roles=[],
    )
    assert thought.is_stale is False


def test_vector_store_mark_stale(sample_thought: CachedThought) -> None:
    """Test VectorStore.mark_stale_by_urn functionality."""
    store = VectorStore()

    # Add thought
    store.add(sample_thought)

    # Add another thought with different URN
    thought2 = sample_thought.model_copy(update={"id": uuid4(), "source_urns": ["urn:mcp:doc:456"]})
    store.add(thought2)

    # Add a third thought sharing the first URN
    thought3 = sample_thought.model_copy(update={"id": uuid4()})
    store.add(thought3)

    # Mark first URN as stale
    count = store.mark_stale_by_urn("urn:mcp:doc:123")

    assert count == 2
    assert store.thoughts[0].is_stale is True  # thought1
    assert store.thoughts[1].is_stale is False  # thought2
    assert store.thoughts[2].is_stale is True  # thought3


def test_mark_stale_idempotent(sample_thought: CachedThought) -> None:
    """Test that marking stale is idempotent."""
    store = VectorStore()
    store.add(sample_thought)

    # First mark
    count1 = store.mark_stale_by_urn("urn:mcp:doc:123")
    assert count1 == 1
    assert store.thoughts[0].is_stale is True

    # Second mark
    count2 = store.mark_stale_by_urn("urn:mcp:doc:123")
    assert count2 == 0  # Should be 0 since it's already stale
    assert store.thoughts[0].is_stale is True


def test_archive_invalidate_source() -> None:
    """Test CoreasonArchive.invalidate_source delegation."""
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_graph_store = MagicMock()
    mock_embedder = MagicMock()

    archive = CoreasonArchive(vector_store=mock_vector_store, graph_store=mock_graph_store, embedder=mock_embedder)

    mock_vector_store.mark_stale_by_urn.return_value = 5

    count = archive.invalidate_source("urn:mcp:doc:999")

    assert count == 5
    mock_vector_store.mark_stale_by_urn.assert_called_once_with("urn:mcp:doc:999")
