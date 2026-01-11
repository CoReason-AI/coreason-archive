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
    # For MultiDiGraph, access edge data via key
    assert store.graph.edges["User:Alice", "Project:Apollo", "CREATED"]["relation"] == "CREATED"


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

    # Get all relations (default outgoing)
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
        assert new_store.graph.edges["User:Alice", "Project:Apollo", "RELATED_TO"]["relation"] == "RELATED_TO"


def test_load_non_existent_file() -> None:
    """Test loading a non-existent file does not crash and leaves graph empty."""
    store = GraphStore()
    filepath = Path("non_existent_file.json")
    store.load(filepath)
    # Should just log warning and continue
    assert len(store.graph.nodes) == 0


def test_edge_update() -> None:
    """Test updating an existing edge (overwriting with same key)."""
    store = GraphStore()
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)
    # Re-adding same relation type should just succeed (idempotent or update)
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)

    assert store.graph.number_of_edges() == 1
    assert store.graph.edges["User:Alice", "Project:Apollo", "CREATED"]["relation"] == "CREATED"


def test_multiple_relationships_same_nodes() -> None:
    """Test multiple relationship types between the same two entities."""
    store = GraphStore()
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.CREATED)
    store.add_relationship("User:Alice", "Project:Apollo", GraphEdgeType.BELONGS_TO)

    assert store.graph.number_of_edges() == 2
    assert store.graph.has_edge("User:Alice", "Project:Apollo", key="CREATED")
    assert store.graph.has_edge("User:Alice", "Project:Apollo", key="BELONGS_TO")

    related = store.get_related_entities("User:Alice")
    assert len(related) == 2
    assert ("Project:Apollo", "CREATED") in related
    assert ("Project:Apollo", "BELONGS_TO") in related


def test_complex_entity_parsing() -> None:
    """Test parsing of complex entity strings containing multiple colons."""
    store = GraphStore()
    complex_entity = "Config:Key:Value:With:Colons"
    store.add_entity(complex_entity)

    assert store.graph.nodes[complex_entity]["type"] == "Config"
    assert store.graph.nodes[complex_entity]["value"] == "Key:Value:With:Colons"


def test_bidirectional_search() -> None:
    """Test searching for incoming, outgoing, and both directions."""
    store = GraphStore()
    # A -> B
    store.add_relationship("Node:A", "Node:B", GraphEdgeType.RELATED_TO)

    # 1. Outgoing from A (Default)
    outgoing = store.get_related_entities("Node:A", direction="outgoing")
    assert len(outgoing) == 1
    assert outgoing[0] == ("Node:B", "RELATED_TO")

    # 2. Incoming to B
    incoming = store.get_related_entities("Node:B", direction="incoming")
    assert len(incoming) == 1
    assert incoming[0] == ("Node:A", "RELATED_TO")

    # 3. Both from A (should include B)
    both_a = store.get_related_entities("Node:A", direction="both")
    assert ("Node:B", "RELATED_TO") in both_a

    # 4. Both from B (should include A)
    both_b = store.get_related_entities("Node:B", direction="both")
    assert ("Node:A", "RELATED_TO") in both_b


def test_graph_cycles() -> None:
    """Test that cycles do not break persistence or traversal."""
    store = GraphStore()
    # A -> B -> A
    store.add_relationship("Node:A", "Node:B", GraphEdgeType.RELATED_TO)
    store.add_relationship("Node:B", "Node:A", GraphEdgeType.RELATED_TO)

    # Check traversal
    related_a = store.get_related_entities("Node:A", direction="outgoing")
    assert ("Node:B", "RELATED_TO") in related_a

    related_b = store.get_related_entities("Node:B", direction="outgoing")
    assert ("Node:A", "RELATED_TO") in related_b

    # Check persistence
    with TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "cycle.json"
        store.save(filepath)

        new_store = GraphStore()
        new_store.load(filepath)

        assert new_store.graph.has_edge("Node:A", "Node:B", key="RELATED_TO")
        assert new_store.graph.has_edge("Node:B", "Node:A", key="RELATED_TO")


def test_self_loop() -> None:
    """Test self-referencing relationship."""
    store = GraphStore()
    store.add_relationship("Node:Self", "Node:Self", GraphEdgeType.RELATED_TO)

    assert store.graph.has_edge("Node:Self", "Node:Self", key="RELATED_TO")

    # related = store.get_related_entities("Node:Self", direction="both")
    # assert len(store.get_related_entities("Node:Self", direction="outgoing")) == 1
    # assert len(store.get_related_entities("Node:Self", direction="both")) == 2
