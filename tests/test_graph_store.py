from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from coreason_archive.graph_store import GraphStore
from coreason_archive.models import GraphEdgeType


def test_add_entity_valid() -> None:
    """Test adding a valid entity."""
    store = GraphStore()
    store.add_entity("User:Alice")

    assert store.graph.has_node("User:Alice")
    assert store.graph.nodes["User:Alice"]["type"] == "User"
    assert store.graph.nodes["User:Alice"]["value"] == "Alice"


def test_add_entity_invalid_format() -> None:
    """Test adding an entity with invalid format raises ValueError."""
    store = GraphStore()

    with pytest.raises(ValueError, match="must follow 'Type:Value' format"):
        store.add_entity("InvalidEntity")

    with pytest.raises(ValueError, match="must have both Type and Value"):
        store.add_entity("TypeOnly:")

    with pytest.raises(ValueError, match="must have both Type and Value"):
        store.add_entity(":ValueOnly")


def test_add_relationship() -> None:
    """Test adding a relationship between entities."""
    store = GraphStore()
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)

    assert store.graph.has_node("User:Alice")
    assert store.graph.has_node("Project:Apollo")
    assert store.graph.has_edge("User:Alice", "Project:Apollo")
    assert store.graph.edges["User:Alice", "Project:Apollo"]["relation"] == "CREATED"


def test_add_relationship_auto_create_nodes() -> None:
    """Test that adding a relationship automatically creates missing nodes."""
    store = GraphStore()
    # Nodes "User:Bob" and "Project:Zeus" do not exist yet
    store.add_relationship("User:Bob", "Project:Zeus", GraphEdgeType.BELONGS_TO)

    assert store.graph.has_node("User:Bob")
    assert store.graph.has_node("Project:Zeus")
    assert store.graph.nodes["User:Bob"]["type"] == "User"
    assert store.graph.nodes["Project:Zeus"]["type"] == "Project"


def test_get_related_entities() -> None:
    """Test retrieving related entities."""
    store = GraphStore()
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)
    store.add_relationship("User:Alice", "Dept:R&D", GraphEdgeType.BELONGS_TO)

    # Get all relations
    related = store.get_related_entities("User:Alice")
    assert len(related) == 2
    assert ("Project:Apollo", "CREATED") in related
    assert ("Dept:R&D", "BELONGS_TO") in related

    # Filter by relation
    created_only = store.get_related_entities("User:Alice", GraphEdgeType.CREATED)
    assert len(created_only) == 1
    assert created_only[0] == ("Project:Apollo", "CREATED")

    # Non-existent node
    assert store.get_related_entities("User:Nobody") == []


def test_save_and_load() -> None:
    """Test saving and loading the graph from JSON."""
    with TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "graph.json"

        # Setup initial graph
        store = GraphStore()
        store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.RELATED_TO)
        store.save(filepath)

        # Verify file exists
        assert filepath.exists()

        # Load into new store
        new_store = GraphStore()
        new_store.load(filepath)

        assert new_store.graph.has_node("User:Alice")
        assert new_store.graph.has_node("Project:Apollo")
        assert new_store.graph.has_edge("User:Alice", "Project:Apollo")
        assert new_store.graph.edges["User:Alice", "Project:Apollo"]["relation"] == "RELATED_TO"


def test_load_non_existent_file() -> None:
    """Test loading a non-existent file does not crash and leaves graph empty."""
    store = GraphStore()
    filepath = Path("non_existent_file.json")
    store.load(filepath)
    # Should just log warning and continue
    assert len(store.graph.nodes) == 0


def test_edge_update() -> None:
    """Test updating an existing edge."""
    store = GraphStore()
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)
    # Update relationship
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.RELATED_TO)

    assert store.graph.edges["User:Alice", "Project:Apollo"]["relation"] == "RELATED_TO"
