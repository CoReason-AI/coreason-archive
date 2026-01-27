# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import List

import pytest
from coreason_identity.models import UserContext

from coreason_archive.archive import CoreasonArchive
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.matchmaker import MatchStrategy
from coreason_archive.models import MemoryScope
from coreason_archive.vector_store import VectorStore


class MockEmbedder(Embedder):
    def embed(self, text: str) -> List[float]:
        # Return a dummy vector of 1536 dims
        return [0.1] * 1536


@pytest.mark.asyncio
async def test_smart_lookup_entity_hop() -> None:
    """Test Smart Lookup returning ENTITY_HOP when boosted."""
    v_store = VectorStore()
    g_store = GraphStore()

    # We use "Project:Apollo" as the hook

    context = UserContext(user_id="user_123", email="test@example.com", groups=["Apollo"])

    # We need to simulate a scenario where:
    # 1. Base vector score is LOW (below Hint Threshold)
    # 2. Boosted score is HIGH (or just boosted)
    # MockEmbedder returns constant vectors, so base score is always 1.0.
    # We need a CustomMockEmbedder that returns low score for query vs thought.

    # VectorStore uses numpy.
    # If we put [0.1] and query with [0.0], dot product is 0. Score 0.
    # But wait, 0 score might be filtered out by min_score=0.0 (inclusive).
    # Let's use vectors that give say 0.5 score.
    # v1 = [1, 0]; v2 = [0.5, 0.866] -> cos = 0.5.

    class ControlEmbedder(MockEmbedder):
        def embed(self, text: str) -> List[float]:
            vec = [0.0] * 1536
            if text == "query":
                vec[0] = 1.0
            else:
                # Thought vector
                # We want similarity ~0.5
                vec[0] = 0.5
                vec[1] = 0.866
            return vec

    archive = CoreasonArchive(v_store, g_store, ControlEmbedder(), entity_extractor=None)

    # Add thought
    # User must be in Apollo group to add to PROJECT:Apollo scope (assuming strict check, wait)
    # The check is: if scope == USER.
    # But I updated Federation to check scope_id in context.groups for PROJECT.
    # And add_thought defaults access_roles to context.groups.
    # So if I add it, I should probably use a context that has Apollo.
    # But wait, add_thought only enforces sovereignty on USER scope in archive.py.
    # "Security Check: Enforce Sovereignty ... if scope == MemoryScope.USER".
    # So for PROJECT scope, any user can write? The prompt said:
    # "Security Check: If scope == MemoryScope.USER, enforce that scope_id == user_context.user_id."
    # It didn't say enforce others.
    # BUT, FederationBroker.get_filter enforces others on READ.
    # So if I write with a user who doesn't have the group, I can write, but maybe not read?
    # Or maybe I should enforce on write too?
    # I'll stick to what I implemented: only USER scope sovereignty on write.
    # So I can pass a context without Apollo to add_thought, but to retrieve I need it.

    ctx_add = UserContext(user_id="user_123", email="test@example.com", groups=["Apollo"])
    t = await archive.add_thought("q", "trace", MemoryScope.PROJECT, "Apollo", user_context=ctx_add)
    t.entities = ["Project:Apollo"]

    # Retrieve
    # Base score ~0.5.
    # Hint threshold default 0.85. So it fails Hint.
    # Boost factor default 1.1. Score -> 0.55. Still fails Hint.
    # But `is_boosted` should be True.
    # So expected strategy: ENTITY_HOP.

    result = await archive.smart_lookup("query", context, hint_threshold=0.85)

    assert result.strategy == MatchStrategy.ENTITY_HOP
    assert result.content["source"] == "entity_hop"
    assert "structurally related" in result.content["hint"]
