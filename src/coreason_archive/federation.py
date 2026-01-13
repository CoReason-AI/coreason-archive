# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from typing import Callable, List

from pydantic import BaseModel, Field

from coreason_archive.models import CachedThought, MemoryScope


class UserContext(BaseModel):
    """
    Represents the security context of the user making a request.
    Includes their identity, group memberships, and assigned roles.
    """

    user_id: str = Field(..., description="Unique identifier of the user")
    dept_ids: List[str] = Field(default_factory=list, description="List of Department IDs the user belongs to")
    project_ids: List[str] = Field(default_factory=list, description="List of Project IDs the user belongs to")
    client_ids: List[str] = Field(default_factory=list, description="List of Client IDs the user belongs to")
    roles: List[str] = Field(default_factory=list, description="List of RBAC roles/claims assigned to the user")


class FederationBroker:
    """
    Enforces data sovereignty and RBAC policies.
    Constructs filters to ensure users only access memories within their allowed scope.
    """

    @staticmethod
    def get_filter(context: UserContext) -> Callable[[CachedThought], bool]:
        """
        Returns a filter function that accepts a CachedThought and returns True
        if the user (context) is allowed to access it.

        Args:
            context: The security context of the user.

        Returns:
            A callable predicate.
        """

        def filter_thought(thought: CachedThought) -> bool:
            # 1. Scope Check
            if thought.scope == MemoryScope.USER:
                if thought.scope_id != context.user_id:
                    return False

            elif thought.scope == MemoryScope.DEPARTMENT:
                if thought.scope_id not in context.dept_ids:
                    return False

            elif thought.scope == MemoryScope.PROJECT:
                if thought.scope_id not in context.project_ids:
                    return False

            elif thought.scope == MemoryScope.CLIENT:
                if thought.scope_id not in context.client_ids:
                    return False

            # 2. RBAC Check
            # If the thought requires specific roles, the user must have ALL of them?
            # Or ANY of them?
            # PRD: "access_roles: RBAC claims required to access this thought"
            # Usually 'required' implies ALL, or at least one match?
            # Standard RBAC often implies "User must have role X".
            # If access_roles is ["admin"], user must have "admin".
            # If access_roles is ["admin", "editor"], does user need both?
            # "Claims required" usually means the resource is protected by these claims.
            # A common interpretation is: The user must possess at least one of the allowed roles
            # OR the user must possess all required roles.
            # Given "access_roles: List[str]", let's assume it lists the roles that *can* access it.
            # i.e., User needs (Role A OR Role B).
            # If list is empty, no specific role required (Public within scope).
            if thought.access_roles:
                # intersection of thought.access_roles and context.roles must not be empty
                has_role = any(role in context.roles for role in thought.access_roles)
                if not has_role:
                    return False

            return True

        return filter_thought
