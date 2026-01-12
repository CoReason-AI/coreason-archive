import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import mock_open, patch
from uuid import uuid4

import numpy as np
import pytest

from coreason_archive.models import CachedThought, MemoryScope
from coreason_archive.vector_store import VectorStore


def create_dummy_thought(
    vector: list[float], scope: MemoryScope = MemoryScope.USER, text: str = "test"
) -> CachedThought:
    """Helper to create a dummy thought with a specific vector."""
    return CachedThought(
        id=uuid4(),
        vector=vector,
        entities=["Entity:Test"],
        scope=scope,
        scope_id="test_scope",
        prompt_text=f"prompt {text}",
        reasoning_trace="test trace",
        final_response=f"response {text}",
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


# --- New Edge Cases & Complex Scenarios ---


def test_numerical_stability() -> None:
    """Test with very small vector magnitudes to ensure stability."""
    store = VectorStore()
    # Very small vector
    small_vec = [1e-10, 1e-10]
    thought = create_dummy_thought(small_vec)
    store.add(thought)

    # Search with same small vector
    # Dot product will be ~2e-20, Norm will be ~1.4e-10
    # Cosine sim should still be 1.0
    results = store.search(small_vec)

    assert len(results) == 1
    assert pytest.approx(results[0][1], abs=1e-5) == 1.0


def test_needle_in_haystack() -> None:
    """Test finding a specific vector among many random ones."""
    store = VectorStore()

    # Add 100 random noise vectors
    np.random.seed(42)
    for i in range(100):
        # random vector on unit circle
        v = np.random.randn(10).tolist()
        store.add(create_dummy_thought(v, text=f"noise_{i}"))

    # Add the "needle" (orthogonal to noise or just distinct)
    # Using a fixed distinctive pattern
    needle_vec = [10.0] * 10
    needle_thought = create_dummy_thought(needle_vec, text="needle")
    store.add(needle_thought)

    # Search for the needle
    results = store.search(needle_vec, limit=5)

    # Expect needle at top
    assert len(results) >= 1
    assert results[0][0].id == needle_thought.id
    assert pytest.approx(results[0][1], abs=1e-5) == 1.0


def test_corrupted_file_loading() -> None:
    """Test loading a corrupted JSON file raises appropriate error."""
    with TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "corrupt.json"

        # Write garbage
        with open(filepath, "w") as f:
            f.write("{ invalid json [")

        store = VectorStore()
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            store.load(filepath)


def test_unicode_persistence() -> None:
    """Verify that complex Unicode characters are preserved."""
    with TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "unicode.json"
        store = VectorStore()

        unicode_text = "Hello 🌍! This is a test: 日本語, العربية, 🐍."
        thought = create_dummy_thought([1.0, 0.0], text=unicode_text)
        thought.final_response = unicode_text  # Ensure it's in a field we check

        store.add(thought)
        store.save(filepath)

        new_store = VectorStore()
        new_store.load(filepath)

        assert len(new_store.thoughts) == 1
        loaded = new_store.thoughts[0]
        assert loaded.final_response == unicode_text
        assert "🌍" in loaded.final_response


def test_duplicate_handling() -> None:
    """Verify behavior when adding duplicate thoughts (same ID)."""
    store = VectorStore()

    thought = create_dummy_thought([1.0, 0.0])

    # Add same object twice
    store.add(thought)
    store.add(thought)

    # Current implementation appends both
    assert len(store.thoughts) == 2

    # Search should return both
    results = store.search([1.0, 0.0])
    assert len(results) == 2
    assert results[0][0].id == thought.id
    assert results[1][0].id == thought.id
    # Both have score 1.0
    assert pytest.approx(results[0][1], abs=1e-5) == 1.0


def test_delete_missing_thought() -> None:
    """Test that deleting a thought that doesn't exist returns False."""
    store = VectorStore()
    assert store.delete(uuid4()) is False


def test_get_by_scope_empty() -> None:
    """Test retrieving thoughts by scope when none match."""
    store = VectorStore()
    assert store.get_by_scope(MemoryScope.USER, "unknown") == []


def test_save_error(tmp_path: Path) -> None:
    """Test handling of save errors (e.g., permission denied)."""
    store = VectorStore()
    store.add(create_dummy_thought([1.0, 0.0]))

    filepath = tmp_path / "store.json"

    # Force IOError during open
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = IOError("Permission denied")
        with pytest.raises(IOError, match="Permission denied"):
            store.save(filepath)


def test_load_io_error(tmp_path: Path) -> None:
    """Test handling of load errors (e.g., permission denied)."""
    filepath = tmp_path / "store.json"
    # Ensure file exists so checks pass
    filepath.touch()

    store = VectorStore()

    # Force IOError during open
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = IOError("Permission denied")
        with pytest.raises(IOError, match="Permission denied"):
            store.load(filepath)


def test_mixed_dimensions_error() -> None:
    """Test that adding vectors of different dimensions raises ValueError."""
    store = VectorStore()
    store.add(create_dummy_thought([1.0, 0.0]))  # Dim 2

    # Try adding Dim 3
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        store.add(create_dummy_thought([1.0, 0.0, 0.0]))


def test_relocation_scenario() -> None:
    """
    Simulate a relocation scenario:
    1. Find all thoughts for a specific scope (e.g., old department).
    2. Delete them.
    3. Verify they are gone and others remain.
    """
    store = VectorStore()

    # User's personal thoughts (keep)
    t1 = create_dummy_thought([1.0], scope=MemoryScope.USER)
    t1.scope_id = "user_1"
    store.add(t1)

    # Old department thoughts (delete)
    t2 = create_dummy_thought([1.0], scope=MemoryScope.DEPARTMENT)
    t2.scope_id = "dept_old"
    store.add(t2)

    t3 = create_dummy_thought([1.0], scope=MemoryScope.DEPARTMENT)
    t3.scope_id = "dept_old"
    store.add(t3)

    # New department thoughts (keep/ignore)
    t4 = create_dummy_thought([1.0], scope=MemoryScope.DEPARTMENT)
    t4.scope_id = "dept_new"
    store.add(t4)

    # Step 1: Find old dept thoughts
    to_delete = store.get_by_scope(MemoryScope.DEPARTMENT, "dept_old")
    assert len(to_delete) == 2

    # Step 2: Delete them
    for t in to_delete:
        store.delete(t.id)

    # Step 3: Verify
    remaining = store.thoughts
    assert len(remaining) == 2
    ids = {t.id for t in remaining}
    assert t1.id in ids
    assert t4.id in ids
    assert t2.id not in ids
    assert t3.id not in ids

    # Verify vector consistency
    assert len(store._vectors) == 2


def test_search_limit_edge_cases() -> None:
    """Test search with limit 0 and limit > total."""
    store = VectorStore()
    store.add(create_dummy_thought([1.0]))
    store.add(create_dummy_thought([0.5]))

    # Limit 0
    results_zero = store.search([1.0], limit=0)
    assert len(results_zero) == 0

    # Limit > total
    results_all = store.search([1.0], limit=100)
    assert len(results_all) == 2
