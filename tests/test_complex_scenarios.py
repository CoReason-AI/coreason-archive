from datetime import datetime, timedelta, timezone
from typing import List
from uuid import uuid4

import pytest

from coreason_archive.archive import CoreasonArchive
from coreason_archive.federation import UserContext
from coreason_archive.graph_store import GraphStore
from coreason_archive.interfaces import Embedder
from coreason_archive.matchmaker import MatchStrategy
from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.vector_store import VectorStore


class DictMockEmbedder(Embedder):
    """
    Mock embedder that maps text to specific vectors.
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def embed(self, text: str) -> List[float]:
        # Return mapped vector or zero vector if not found
        return self.mapping.get(text, [0.0] * 1536)


def create_thought(
    id_str: str,
    vector: list[float],
    scope: MemoryScope,
    scope_id: str,
    created_at: datetime,
    entities: list[str] | None = None,
) -> CachedThought:
    """Helper to manually create thoughts."""
    return CachedThought(
        id=uuid4(),
        vector=vector,
        entities=entities or [],
        scope=scope,
        scope_id=scope_id,
        prompt_text=f"prompt {id_str}",
        reasoning_trace=f"trace {id_str}",
        final_response=f"response {id_str}",
        source_urns=[],
        created_at=created_at,
        ttl_seconds=86400 * 365,  # Long TTL
        access_roles=[],
    )


@pytest.mark.asyncio
async def test_full_hybrid_loop() -> None:
    """
    Complex Scenario:
    User 'Alice' from 'Dept-Eng' working on 'Project-Apollo'.

    Candidates:
    1. A (Accessible, High Sim, Now): Dept-Eng thought.
    2. B (Accessible, Med Sim, Now, Boosted): Project-Apollo thought.
    3. C (Inaccessible, High Sim, Now): Dept-Sales thought.
    4. D (Accessible, High Sim, Old): Dept-Eng thought from 1 year ago.
    """

    # Vectors (simplified for orthogonality)
    # 1536 dims, but we only care about first few
    vec_query = [1.0] + [0.0] * 1535
    vec_high = [1.0] + [0.0] * 1535  # Sim 1.0
    vec_med = [0.707, 0.707] + [0.0] * 1534  # Sim ~0.7

    embedder = DictMockEmbedder(
        {
            "query": vec_query,
        }
    )

    store = VectorStore()
    graph = GraphStore()
    archive = CoreasonArchive(store, graph, embedder, entity_extractor=None)

    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)

    # 1. Thought A: Dept-Eng, High Sim, Now
    t_a = create_thought("A", vec_high, MemoryScope.DEPARTMENT, "Dept-Eng", now)
    store.add(t_a)

    # 2. Thought B: Project-Apollo, Med Sim, Now
    # Entities boost it.
    # scope_id must match context.project_ids ("Apollo")
    t_b = create_thought("B", vec_med, MemoryScope.PROJECT, "Apollo", now, entities=["Project:Apollo"])
    store.add(t_b)

    # 3. Thought C: Dept-Sales, High Sim, Now (Should be filtered)
    t_c = create_thought("C", vec_high, MemoryScope.DEPARTMENT, "Dept-Sales", now)
    store.add(t_c)

    # 4. Thought D: Dept-Eng, High Sim, Old (Should be decayed)
    t_d = create_thought("D", vec_high, MemoryScope.DEPARTMENT, "Dept-Eng", one_year_ago)
    store.add(t_d)

    # Context
    context = UserContext(
        user_id="Alice",
        dept_ids=["Dept-Eng"],
        # context project ID should be "Apollo" so the constructed entity is "Project:Apollo"
        project_ids=["Apollo"],
    )

    # Execute Retrieve
    # Graph Boost Factor = 1.5 to ensure B can compete
    results = await archive.retrieve("query", context, limit=10, graph_boost_factor=1.5)

    # Analysis
    ids = [t.id for t, s in results]
    scores = {t.id: s for t, s in results}

    # 1. C should be gone
    assert t_c.id not in ids

    # 2. A, B, D should be present
    assert t_a.id in ids
    assert t_b.id in ids
    assert t_d.id in ids

    # 3. Ranking logic
    # Score A: ~1.0 (High Sim) * 1.0 (Time) = 1.0
    # Score B: ~0.7 (Med Sim) * 1.0 (Time) * 1.5 (Boost) = 1.05
    # Score D: ~1.0 (High Sim) * Decay(1yr)
    # Decay(Dept, 1yr): lambda=4e-8. t=3.15e7. exp(-1.26) ~= 0.28.
    # Score D ~= 0.28

    # Expected Order: B > A > D
    assert results[0][0].id == t_b.id
    assert results[1][0].id == t_a.id
    assert results[2][0].id == t_d.id

    # Verify scores roughly
    assert scores[t_b.id] > 1.0  # > 1.0 due to boost
    assert scores[t_a.id] == pytest.approx(1.0, abs=1e-5)
    assert scores[t_d.id] < 0.5


@pytest.mark.asyncio
async def test_matchmaker_misconfiguration() -> None:
    """Test behavior when thresholds are logically inverted."""
    # Hint (0.9) > Exact (0.5)
    # If score is 0.7:
    # It is > Exact (0.5). Should it return Exact?
    # Yes, standard logic checks strict ">=".

    vec = [1.0] + [0.0] * 1535
    embedder = DictMockEmbedder({"q": vec})
    store = VectorStore()
    archive = CoreasonArchive(store, GraphStore(), embedder)

    # Thought with score 0.7
    # We use a vector that gives 0.7 similarity to [1, 0...]
    # [0.7, 0.7, ...]
    vec_07 = [0.7, 0.7] + [0.0] * 1534
    # Actually 0.7 / 1 * 1 (approx, norms are tricky manually)
    # Norm vec_07 = sqrt(0.49+0.49) = sqrt(0.98) ~ 0.99
    # Dot = 0.7. Score = 0.7 / 0.99 ~= 0.707.

    t = create_thought("X", vec_07, MemoryScope.USER, "u1", datetime.now(timezone.utc))
    store.add(t)

    context = UserContext(user_id="u1")

    # Exact Threshold 0.5, Hint 0.9
    # Score is ~0.7.
    # 0.7 > 0.5 -> Should match Exact first if `if score >= exact` is first check.
    result = await archive.smart_lookup("q", context, exact_threshold=0.5, hint_threshold=0.9)

    assert result.strategy == MatchStrategy.EXACT_HIT


@pytest.mark.asyncio
async def test_empty_context_edge_case() -> None:
    """User with absolutely no roles or memberships searching."""
    store = VectorStore()
    embedder = DictMockEmbedder({"q": [1.0] * 1536})
    archive = CoreasonArchive(store, GraphStore(), embedder)

    # 1. Public user thought (Wait, user scope always requires matching ID)
    t1 = create_thought("1", [1.0] * 1536, MemoryScope.USER, "other", datetime.now(timezone.utc))
    store.add(t1)

    # 2. Dept thought
    t2 = create_thought("2", [1.0] * 1536, MemoryScope.DEPARTMENT, "dept", datetime.now(timezone.utc))
    store.add(t2)

    context = UserContext(user_id="lonely_user")  # No depts, no projects

    results = await archive.retrieve("q", context)

    # Should see nothing
    assert len(results) == 0


@pytest.mark.asyncio
async def test_future_date_decay() -> None:
    """Test thought with future timestamp handles gracefully (no decay)."""
    store = VectorStore()
    embedder = DictMockEmbedder({"q": [1.0] * 1536})
    archive = CoreasonArchive(store, GraphStore(), embedder)

    future = datetime.now(timezone.utc) + timedelta(days=365)
    t = create_thought("F", [1.0] * 1536, MemoryScope.USER, "u1", future)
    store.add(t)

    context = UserContext(user_id="u1")
    results = await archive.retrieve("q", context)

    assert len(results) == 1
    # decay factor should be 1.0 (clamped)
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
