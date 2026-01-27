# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from datetime import datetime
from uuid import uuid4

from coreason_identity.models import UserContext

from coreason_archive.federation import FederationBroker
from coreason_archive.models import CachedThought, MemoryScope


def create_thought(
    scope: MemoryScope,
    scope_id: str,
    access_roles: list[str] | None = None,
    owner_id: str = "user_123",
) -> CachedThought:
    if access_roles is None:
        access_roles = []
    return CachedThought(
        id=uuid4(),
        vector=[0.0] * 1536,
        entities=[],
        scope=scope,
        scope_id=scope_id,
        prompt_text="test",
        reasoning_trace="test",
        final_response="test",
        owner_id=owner_id,
        source_urns=[],
        created_at=datetime.now(),
        ttl_seconds=3600,
        access_roles=access_roles,
    )


def test_user_scope_access() -> None:
    """Test access to USER scoped memories."""
    context = UserContext(user_id="user_123", email="test@example.com")
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Own memory -> Allowed
    thought_own = create_thought(MemoryScope.USER, "user_123")
    assert filter_func(thought_own) is True

    # Other's memory -> Denied
    thought_other = create_thought(MemoryScope.USER, "user_456")
    assert filter_func(thought_other) is False


def test_dept_scope_access() -> None:
    """Test access to DEPT scoped memories."""
    context = UserContext(user_id="user_123", email="test@example.com", groups=["dept_it", "dept_hr"])
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Member dept -> Allowed
    thought_it = create_thought(MemoryScope.DEPARTMENT, "dept_it")
    assert filter_func(thought_it) is True

    # Non-member dept -> Denied
    thought_sales = create_thought(MemoryScope.DEPARTMENT, "dept_sales")
    assert filter_func(thought_sales) is False


def test_project_scope_access() -> None:
    """Test access to PROJECT scoped memories."""
    context = UserContext(user_id="user_123", email="test@example.com", groups=["proj_alpha"])
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Member project -> Allowed
    thought_alpha = create_thought(MemoryScope.PROJECT, "proj_alpha")
    assert filter_func(thought_alpha) is True

    # Non-member project -> Denied
    thought_beta = create_thought(MemoryScope.PROJECT, "proj_beta")
    assert filter_func(thought_beta) is False


def test_client_scope_access() -> None:
    """Test access to CLIENT scoped memories."""
    context = UserContext(user_id="user_123", email="test@example.com", groups=["client_x"])
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Member client -> Allowed
    thought_x = create_thought(MemoryScope.CLIENT, "client_x")
    assert filter_func(thought_x) is True

    # Non-member client -> Denied
    thought_y = create_thought(MemoryScope.CLIENT, "client_y")
    assert filter_func(thought_y) is False


def test_rbac_access() -> None:
    """Test Role-Based Access Control logic."""
    # User has 'admin' role (in groups)
    context = UserContext(user_id="user_123", email="test@example.com", groups=["admin"])
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Thought requires 'admin' -> Allowed
    thought_admin = create_thought(MemoryScope.USER, "user_123", access_roles=["admin"])
    assert filter_func(thought_admin) is True

    # Thought requires 'editor' -> Denied
    thought_editor = create_thought(MemoryScope.USER, "user_123", access_roles=["editor"])
    assert filter_func(thought_editor) is False

    # Thought allows 'admin' OR 'editor' -> Allowed (since user has admin)
    thought_mixed = create_thought(MemoryScope.USER, "user_123", access_roles=["editor", "admin"])
    assert filter_func(thought_mixed) is True

    # Public thought (no roles required) -> Allowed
    thought_public = create_thought(MemoryScope.USER, "user_123", access_roles=[])
    assert filter_func(thought_public) is True


def test_rbac_and_scope_combined() -> None:
    """Test combination of Scope and RBAC."""
    # User in dept_it and has intern role (all in groups)
    context = UserContext(user_id="user_123", email="test@example.com", groups=["dept_it", "intern"])
    broker = FederationBroker()
    filter_func = broker.get_filter(context)

    # Right scope, Right role -> Allowed
    t1 = create_thought(MemoryScope.DEPARTMENT, "dept_it", access_roles=["intern"])
    assert filter_func(t1) is True

    # Right scope, Wrong role -> Denied
    t2 = create_thought(MemoryScope.DEPARTMENT, "dept_it", access_roles=["manager"])
    assert filter_func(t2) is False

    # Wrong scope, Right role -> Denied
    t3 = create_thought(MemoryScope.DEPARTMENT, "dept_sales", access_roles=["intern"])
    assert filter_func(t3) is False
