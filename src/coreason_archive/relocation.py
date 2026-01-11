from typing import Protocol, runtime_checkable


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


class StubRelocationManager(RelocationManager):
    """
    A simple stub implementation of the RelocationManager.
    """

    async def on_role_change(self, user_id: str, new_roles: list[str]) -> None:
        pass

    async def on_dept_transfer(self, user_id: str, old_dept_id: str, new_dept_id: str) -> None:
        pass
