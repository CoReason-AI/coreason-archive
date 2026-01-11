from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pytest

from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.vector_store import VectorStore


def create_dummy_thought(vector: list[float], scope: MemoryScope = MemoryScope.USER) -> CachedThought:
    """Helper to create a dummy thought with a specific vector."""
    return CachedThought(
        id=uuid4(),
        vector=vector,
        entities=["Entity:Test"],
        scope=scope,
        scope_id="test_scope",
        prompt_text="test prompt",
        reasoning_trace="test trace",
        final_response="test response",
        source_urns=[],
        created_at=datetime.now(),
        ttl_seconds=3600,
        access_roles=[],
    )


def test_add_and_search_basic() -> None:
    """Test adding thoughts and retrieving them with exact match."""
    store = VectorStore()

    # Vector A: [1, 0, 0]
    thought_a = create_dummy_thought([1.0, 0.0, 0.0])
    # Vector B: [0, 1, 0] (Orthogonal)
    thought_b = create_dummy_thought([0.0, 1.0, 0.0])

    store.add(thought_a)
    store.add(thought_b)

    # Search for A
    results = store.search([1.0, 0.0, 0.0])
    assert len(results) >= 1
    top_thought, score = results[0]

    assert top_thought.id == thought_a.id
    assert pytest.approx(score, abs=1e-5) == 1.0


def test_cosine_similarity_logic() -> None:
    """Test various vector relationships."""
    store = VectorStore()

    # A: [1, 0]
    # B: [0, 1] -> 90 degrees, sim 0
    # C: [-1, 0] -> 180 degrees, sim -1
    # D: [1, 1] -> 45 degrees, sim ~0.707

    # Add only D
    thought_d = create_dummy_thought([1.0, 1.0])
    store.add(thought_d)

    # Search with [1, 0]
    results = store.search([1.0, 0.0])
    assert len(results) == 1
    _, score = results[0]

    # Cos(45) = 1/sqrt(2) approx 0.7071
    assert pytest.approx(score, abs=1e-4) == 0.7071


def test_min_score_filter() -> None:
    """Test filtering by minimum score."""
    store = VectorStore()

    thought_match = create_dummy_thought([0.99, 0.01])  # close to [1,0]
    thought_mismatch = create_dummy_thought([0.01, 0.99])  # far from [1,0]

    store.add(thought_match)
    store.add(thought_mismatch)

    # Search with high threshold
    results = store.search([1.0, 0.0], min_score=0.8)

    assert len(results) == 1
    assert results[0][0].id == thought_match.id


def test_limit_results() -> None:
    """Test limiting the number of results."""
    store = VectorStore()

    for _ in range(5):
        store.add(create_dummy_thought([1.0, 0.0]))

    results = store.search([1.0, 0.0], limit=3)
    assert len(results) == 3


def test_persistence() -> None:
    """Test saving and loading from disk."""
    with TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "vector_store.json"

        # Setup
        store = VectorStore()
        original_thought = create_dummy_thought([0.1, 0.2, 0.3])
        store.add(original_thought)
        store.save(filepath)

        # Reload
        new_store = VectorStore()
        new_store.load(filepath)

        assert len(new_store.thoughts) == 1
        loaded_thought = new_store.thoughts[0]

        assert loaded_thought.id == original_thought.id
        assert loaded_thought.vector == [0.1, 0.2, 0.3]

        # Verify vector cache was rebuilt
        assert len(new_store._vectors) == 1
        assert new_store._vectors[0] == [0.1, 0.2, 0.3]


def test_search_empty_store() -> None:
    """Test searching an empty store returns empty list."""
    store = VectorStore()
    results = store.search([1.0, 0.0])
    assert results == []


def test_zero_norm_query() -> None:
    """Test searching with a zero vector handles gracefully."""
    store = VectorStore()
    store.add(create_dummy_thought([1.0, 0.0]))

    results = store.search([0.0, 0.0])
    assert results == []


def test_load_missing_file() -> None:
    """Test loading a non-existent file works gracefully."""
    store = VectorStore()
    store.load(Path("non_existent.json"))
    assert len(store.thoughts) == 0


def test_complex_vectors() -> None:
    """Test with 1536-dim vectors (typical OpenAI size)."""
    store = VectorStore()

    # Create random vector
    vec = np.random.rand(1536).tolist()
    thought = create_dummy_thought(vec)
    store.add(thought)

    # Search with same vector
    results = store.search(vec)
    assert len(results) == 1
    assert pytest.approx(results[0][1], abs=1e-5) == 1.0
