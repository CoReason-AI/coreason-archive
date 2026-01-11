import json
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from coreason_archive.models import GraphEdgeType
from coreason_archive.utils.logger import logger


class GraphStore:
    """
    A lightweight Graph Store using NetworkX for in-memory graph operations
    and JSON for persistence.
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize the GraphStore.

        Args:
            graph: An optional existing NetworkX DiGraph. If None, creates a new one.
        """
        self.graph = graph if graph is not None else nx.DiGraph()

    def add_entity(self, entity_string: str) -> None:
        """
        Parses an entity string in the format "Type:Value" and adds it as a node.

        Args:
            entity_string: The entity string (e.g., "User:Alice").

        Raises:
            ValueError: If the entity string does not follow the "Type:Value" format.
        """
        if ":" not in entity_string:
            raise ValueError(f"Entity string '{entity_string}' must follow 'Type:Value' format.")

        entity_type, entity_value = entity_string.split(":", 1)

        if not entity_type or not entity_value:
            raise ValueError(f"Entity string '{entity_string}' must have both Type and Value.")

        if not self.graph.has_node(entity_string):
            self.graph.add_node(entity_string, type=entity_type, value=entity_value)
            logger.debug(f"Added node: {entity_string}")

    def add_relationship(self, source: str, target: str, relation: GraphEdgeType) -> None:
        """
        Adds a directed edge between source and target entities with a specific relationship type.
        Ensures nodes exist before adding the edge.

        Args:
            source: The source entity string.
            target: The target entity string.
            relation: The type of relationship (GraphEdgeType).
        """
        # Ensure nodes exist
        if not self.graph.has_node(source):
            self.add_entity(source)
        if not self.graph.has_node(target):
            self.add_entity(target)

        self.graph.add_edge(source, target, relation=relation.value)
        logger.debug(f"Added edge: {source} -> {target} [{relation.value}]")

    def get_related_entities(self, entity: str, relation: Optional[GraphEdgeType] = None) -> List[Tuple[str, str]]:
        """
        Retrieves entities related to the given entity.

        Args:
            entity: The source entity string.
            relation: Optional filter for a specific relationship type.

        Returns:
            A list of tuples (target_entity, relation_type).
        """
        if not self.graph.has_node(entity):
            return []

        related = []
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            edge_relation = edge_data.get("relation")

            if relation is None or edge_relation == relation.value:
                related.append((neighbor, edge_relation))

        return related

    def save(self, filepath: Path) -> None:
        """
        Saves the graph structure to a JSON file.

        Args:
            filepath: The path to the JSON file.
        """
        data = json_graph.node_link_data(self.graph)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Graph saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """
        Loads the graph structure from a JSON file.

        Args:
            filepath: The path to the JSON file.
        """
        if not filepath.exists():
            logger.warning(f"Graph file {filepath} not found. Starting with empty graph.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.info(f"Graph loaded from {filepath}")
