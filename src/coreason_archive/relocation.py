# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import Protocol, runtime_checkable

from coreason_archive.graph_store import GraphStore
from coreason_archive.models import GraphEdgeType, MemoryScope
from coreason_archive.utils.logger import logger
from coreason_archive.vector_store import VectorStore


@runtime_checkable
class RelocationManager(Protocol):
    """
    Interface for handling identity events like role changes and department transfers.
    """

    async def on_role_change(self, user_id: str, new_roles: list[str]) -> None:
        """
        Handle a change in user roles.
        Expected to revoke access to old scopes and migrate data if needed.
        """
        ...

    async def on_dept_transfer(self, user_id: str, old_dept_id: str, new_dept_id: str) -> None:
        """
        Handle a user transferring departments.
        Expected to lock old department memories and migrate user personal memories.
        """
        ...


class CoreasonRelocationManager(RelocationManager):
    """
    Implementation of the RelocationManager that orchestrates sanitization and migration.
    """

    def __init__(self, vector_store: VectorStore, graph_store: GraphStore) -> None:
        """
        Initialize the CoreasonRelocationManager.

        Args:
            vector_store: Access to the VectorStore for deleting thoughts.
            graph_store: Access to the GraphStore for traversing relationships.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store

    async def on_role_change(self, user_id: str, new_roles: list[str]) -> None:
        """
        Handle a change in user roles.
        Currently passive: access control is handled by FederationBroker.
        """
        logger.info(f"User {user_id} roles updated to {new_roles}. No active migration required.")

    async def on_dept_transfer(self, user_id: str, old_dept_id: str, new_dept_id: str) -> None:
        """
        Handle a user transferring departments.
        Performs "Sanitization":
        1. Finds all USER scope memories for the user.
        2. Checks if they are linked to any Entity belonging to the OLD department.
        3. Deletes any such memories.
        """
        logger.info(f"Processing transfer for {user_id} from {old_dept_id} to {new_dept_id}")

        # 1. Find all USER scope memories
        user_thoughts = self.vector_store.get_by_scope(MemoryScope.USER, user_id)

        # Expected entity format for department
        old_dept_entity = f"Department:{old_dept_id}"

        thoughts_to_delete = []

        for thought in user_thoughts:
            # 2. Check entities for links to old department
            is_contaminated = False
            for entity in thought.entities:
                # Check if this entity belongs to the old department
                # We look for outgoing edges from Entity -> BELONGS_TO -> Department:Old
                related = self.graph_store.get_related_entities(
                    entity, relation=GraphEdgeType.BELONGS_TO, direction="outgoing"
                )

                for neighbor, _ in related:
                    if neighbor == old_dept_entity:
                        is_contaminated = True
                        logger.warning(f"Thought {thought.id} contaminated by {entity} belonging to {old_dept_entity}")
                        break

                if is_contaminated:
                    break

            if is_contaminated:
                thoughts_to_delete.append(thought)

        # 3. Delete contaminated thoughts
        for thought in thoughts_to_delete:
            self.vector_store.delete(thought.id)
            logger.info(f"Sanitized (deleted) thought {thought.id} for user {user_id}")

        logger.info(f"Sanitization complete. Deleted {len(thoughts_to_delete)} thoughts.")


class StubRelocationManager(RelocationManager):
    """
    A simple stub implementation of the RelocationManager.
    """

    async def on_role_change(self, user_id: str, new_roles: list[str]) -> None:
        pass

    async def on_dept_transfer(self, user_id: str, old_dept_id: str, new_dept_id: str) -> None:
        pass
