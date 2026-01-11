import pytest

from coreason_archive.relocation import StubRelocationManager


@pytest.mark.asyncio
async def test_stub_relocation_manager() -> None:
    """Test that the stub methods can be called without error."""
    manager = StubRelocationManager()

    # Should do nothing and not raise
    await manager.on_role_change("user_1", ["admin"])
    await manager.on_dept_transfer("user_1", "dept_old", "dept_new")

    assert True
