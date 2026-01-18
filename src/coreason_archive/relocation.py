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

from coreason_archive.federation import FederationBroker
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
        Sanitizes user memories by validating them against the new role set.
        Perform two types of checks:
        1. RBAC Check: Ensure thought.access_roles matches new_roles.
        2. Graph Check: Ensure entities linked to the thought do not belong to
           restricted departments/contexts that the user no longer has access to.
        """
        logger.info(f"Processing role change for {user_id}. New roles: {new_roles}")

        # 1. Find all USER scope memories
        user_thoughts = self.vector_store.get_by_scope(MemoryScope.USER, user_id)

        thoughts_to_delete = []

        for thought in user_thoughts:
            is_compliant = True

            # 2. RBAC Check
            if not FederationBroker.check_access(new_roles, thought.access_roles):
                logger.warning(
                    f"Thought {thought.id} (required: {thought.access_roles}) not accessible with new roles {new_roles}"
                )
                is_compliant = False

            # 3. Graph Contamination Check (Active Sanitization)
            if is_compliant and thought.entities:
                for entity in thought.entities:
                    # Check if entity belongs to a Department
                    related = self.graph_store.get_related_entities(
                        entity, relation=GraphEdgeType.BELONGS_TO, direction="outgoing"
                    )
                    for neighbor, _ in related:
                        # Assuming neighbor format "Department:Name"
                        if neighbor.startswith("Department:"):
                            dept_name = neighbor.split(":", 1)[1]
                            # Check if user has access to this Department based on new_roles.
                            # We assume a naming convention or generic role check.
                            # Convention: "dept:<name>" or "admin" grants access.
                            required_role = f"dept:{dept_name}"
                            has_access = "admin" in new_roles or required_role in new_roles or dept_name in new_roles

                            if not has_access:
                                logger.warning(
                                    f"Thought {thought.id} contaminated by {entity} belonging to {neighbor}. "
                                    f"User lacks role {required_role}."
                                )
                                is_compliant = False
                                break
                    if not is_compliant:
                        break

            if not is_compliant:
                thoughts_to_delete.append(thought)

        # 4. Delete non-compliant thoughts
        for thought in thoughts_to_delete:
            self.vector_store.delete(thought.id)
            logger.info(f"Sanitized (deleted) thought {thought.id} due to role change for {user_id}")

        logger.info(f"Role change sanitization complete. Deleted {len(thoughts_to_delete)} thoughts.")

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
