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

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.matchmaker import MatchStrategy
from coreason_archive.models import MemoryScope
from coreason_archive.vector_store import VectorStore


class ConfigurableEmbedder(Embedder):
    """
    Embedder that returns specific vectors for specific texts.
    Default orthogonal vector [0...].
    """

    def __init__(self, mapping: dict[str, float]):
        """
        mapping: dict of text -> similarity score (0.0 to 1.0) relative to "query".
        We assume "query" maps to [1, 0, ...].
        Other texts map to [score, sqrt(1-score^2), ...]
        """
        self.mapping = mapping

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * 1536
        if text == "query":
            vec[0] = 1.0
            return vec

        # Check if any key in mapping is a substring of text
        score = 0.0
        for k, v in self.mapping.items():
            if k in text:
                score = v
                break

        # Approximate vector to yield dot product = score
        # [score, sqrt(1-score^2)] . [1, 0] = score
        vec[0] = score
        # Fill rest to normalize (roughly)
        import math

        if score < 1.0:
            vec[1] = math.sqrt(1.0 - score * score)
        return vec


@pytest.mark.asyncio
async def test_precedence_high_score_boosted() -> None:
    """
    Test that if a thought is boosted but also has a high enough score for SEMANTIC_HINT,
    SEMANTIC_HINT takes precedence over ENTITY_HOP.
    """
    # Setup: Thought score 0.9 (>= Hint 0.85). Boosted.
    embedder = ConfigurableEmbedder({"high_score_thought": 0.9})
    v_store = VectorStore()
    g_store = GraphStore()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    # Add thought linked to active project
    t = await archive.add_thought("high_score_thought", "content", MemoryScope.PROJECT, "Apollo", "u1")
    t.entities = ["Project:Apollo"]

    context = UserContext(user_id="u1", project_ids=["Apollo"])

    # Retrieve
    # Base score 0.9. Boost 1.1 -> 0.99.
    # > Hint (0.85). Also > Exact (0.99)? Maybe close.
    # Let's check precedence: Exact > Hint > Entity Hop.

    result = await archive.smart_lookup(
        "query",
        context,
        exact_threshold=0.999,  # Ensure it doesn't hit exact
        hint_threshold=0.85,
    )

    assert result.strategy == MatchStrategy.SEMANTIC_HINT
    # It should NOT be ENTITY_HOP even though it is boosted.


@pytest.mark.asyncio
async def test_low_score_entity_hop() -> None:
    """
    Test that a thought with very low base score but boosted is returned as ENTITY_HOP.
    """
    # Setup: Thought score 0.2. Boosted.
    embedder = ConfigurableEmbedder({"low_score_thought": 0.2})
    v_store = VectorStore()
    g_store = GraphStore()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    t = await archive.add_thought("low_score_thought", "content", MemoryScope.PROJECT, "Apollo", "u1")
    t.entities = ["Project:Apollo"]

    context = UserContext(user_id="u1", project_ids=["Apollo"])

    # Base 0.2. Boost 1.5 -> 0.3.
    # < Hint (0.85).
    # is_boosted = True.

    result = await archive.smart_lookup("query", context, graph_boost_factor=1.5)

    assert result.strategy == MatchStrategy.ENTITY_HOP
    assert result.score == pytest.approx(0.3, abs=0.01)


@pytest.mark.asyncio
async def test_exact_hit_precedence() -> None:
    """
    Test that EXACT_HIT takes precedence over everything.
    """
    embedder = ConfigurableEmbedder({"exact_thought": 1.0})
    v_store = VectorStore()
    g_store = GraphStore()
    archive = CoreasonArchive(v_store, g_store, embedder, entity_extractor=None)

    t = await archive.add_thought("exact_thought", "content", MemoryScope.PROJECT, "Apollo", "u1")
    t.entities = ["Project:Apollo"]

    context = UserContext(user_id="u1", project_ids=["Apollo"])

    # Base 1.0. Boost 1.1 -> 1.1.
    # > Exact (0.99).

    result = await archive.smart_lookup("query", context, exact_threshold=0.99)

    assert result.strategy == MatchStrategy.EXACT_HIT
