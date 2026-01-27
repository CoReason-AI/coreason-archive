from typing import List

import pytest
from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.models import MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1536


@pytest.mark.asyncio
async def test_sovereignty_violation() -> None:
    """Test that writing to another user's scope raises ValueError."""
    v_store = VectorStore()
    g_store = GraphStore()
    embedder = MockEmbedder()
    archive = CoreasonArchive(v_store, g_store, embedder)

    ctx = UserContext(user_id="user_123", email="test@example.com")

    # User 123 tries to write to User 456's scope
    with pytest.raises(ValueError, match="Sovereignty Violation"):
        await archive.add_thought(
            prompt="Attack",
            response="Malicious",
            scope=MemoryScope.USER,
            scope_id="user_456",
            user_context=ctx,
        )
