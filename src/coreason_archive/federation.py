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

from coreason_identity.models import UserContext

from coreason_archive.models import CachedThought, MemoryScope


class FederationBroker:
    """
    Enforces data sovereignty and RBAC policies.
    Constructs filters to ensure users only access memories within their allowed scope.
    """

    @staticmethod
    def check_access(user_roles: List[str], required_roles: List[str]) -> bool:
        """
        Checks if the user has the required roles to access a resource.
        Returns True if required_roles is empty OR if user has at least one matching role.

        Args:
            user_roles: The list of roles assigned to the user.
            required_roles: The list of roles required/allowed for the resource.

        Returns:
            True if access is granted, False otherwise.
        """
        if not required_roles:
            return True
        # Check for intersection
        return any(role in user_roles for role in required_roles)

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

            elif thought.scope in (MemoryScope.DEPARTMENT, MemoryScope.PROJECT, MemoryScope.CLIENT):
                # Check if the scope_id is present in the user's groups
                if thought.scope_id not in context.groups:
                    return False

            # 2. RBAC Check
            if not FederationBroker.check_access(context.groups, thought.access_roles):
                return False

            return True

        return filter_thought
