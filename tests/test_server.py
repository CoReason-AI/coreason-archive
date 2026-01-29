# tests/test_server.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from coreason_archive.matchmaker import MatchStrategy, SearchResult
from coreason_archive.models import MemoryScope
from coreason_archive.server import app

# Create a mock archive that will be reused or we can create new ones per test
# But since we patch init_archive, we can control what it returns.


def test_health() -> None:
    mock_archive = MagicMock()
    mock_archive.vector_store.thoughts = [1, 2, 3]  # len 3
    mock_archive.graph_store.graph.nodes = [1, 2]  # len 2

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["vector_store_size"] == 3
                assert data["graph_nodes"] == 2


def test_add_thought() -> None:
    mock_archive = MagicMock()
    mock_archive.add_thought = AsyncMock()
    # Return a dummy object with id
    mock_thought = MagicMock()
    mock_thought.id = "1234-uuid"
    mock_archive.add_thought.return_value = mock_thought

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "prompt": "Hello",
                    "response": "World",
                    "user_id": "test_user",
                    "scope": "USER",
                }
                response = client.post("/thoughts", json=payload)
                assert response.status_code == 201
                assert response.json()["status"] == "success"
                assert response.json()["thought_id"] == "1234-uuid"

                # Verify call
                mock_archive.add_thought.assert_called_once()
                kwargs = mock_archive.add_thought.call_args.kwargs
                assert kwargs["prompt"] == "Hello"
                assert kwargs["scope"] == MemoryScope.USER


def test_add_thought_invalid_scope() -> None:
    mock_archive = MagicMock()
    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "prompt": "Hello",
                    "response": "World",
                    "user_id": "test_user",
                    "scope": "INVALID_SCOPE",
                }
                response = client.post("/thoughts", json=payload)
                assert response.status_code == 400
                assert "Invalid scope" in response.json()["detail"]


def test_add_thought_project_scope() -> None:
    mock_archive = MagicMock()
    mock_archive.add_thought = AsyncMock()
    mock_thought = MagicMock()
    mock_thought.id = "1234-uuid"
    mock_archive.add_thought.return_value = mock_thought

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "prompt": "Project Thought",
                    "response": "Data",
                    "user_id": "test_user",
                    "scope": "PROJECT",
                    "project_id": "proj_123",
                }
                response = client.post("/thoughts", json=payload)
                assert response.status_code == 201

                # Verify groups were populated
                kwargs = mock_archive.add_thought.call_args.kwargs
                user_context = kwargs["user_context"]
                assert "proj_123" in user_context.groups


def test_add_thought_exception() -> None:
    mock_archive = MagicMock()
    # Simulate generic exception
    mock_archive.add_thought = AsyncMock(side_effect=Exception("DB Error"))

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "prompt": "Hello",
                    "response": "World",
                    "user_id": "test_user",
                    "scope": "USER",
                }
                response = client.post("/thoughts", json=payload)
                assert response.status_code == 500
                assert response.json()["detail"] == "Internal Server Error"


def test_add_thought_value_error() -> None:
    mock_archive = MagicMock()
    # Simulate ValueError
    mock_archive.add_thought = AsyncMock(side_effect=ValueError("Bad Input"))

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "prompt": "Hello",
                    "response": "World",
                    "user_id": "test_user",
                    "scope": "USER",
                }
                response = client.post("/thoughts", json=payload)
                assert response.status_code == 400
                assert "Bad Input" in response.json()["detail"]


def test_search() -> None:
    mock_archive = MagicMock()
    mock_archive.smart_lookup = AsyncMock()

    result_obj = SearchResult(
        strategy=MatchStrategy.EXACT_HIT,
        thought=None,
        score=1.0,
        content={"message": "Found it"},
    )
    mock_archive.smart_lookup.return_value = result_obj

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                context_payload = {
                    "user_id": "test_user",
                    "email": "test@example.com",
                    "groups": ["group1"],
                }
                payload = {
                    "query": "Hello",
                    "context": context_payload,
                }
                response = client.post("/search", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert data["strategy"] == "EXACT_HIT"
                assert data["content"]["message"] == "Found it"

                # Verify context was passed correctly
                mock_archive.smart_lookup.assert_called_once()
                args = mock_archive.smart_lookup.call_args
                # args[0] is query, args[1] is context object
                assert args[0][0] == "Hello"
                assert args[0][1].user_id == "test_user"


def test_search_exception() -> None:
    mock_archive = MagicMock()
    mock_archive.smart_lookup = AsyncMock(side_effect=Exception("Search failed"))

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                payload = {
                    "query": "Hello",
                    "context": {"user_id": "u", "email": "e@example.com", "groups": []},
                }
                response = client.post("/search", json=payload)
                assert response.status_code == 500


def test_lifespan_save() -> None:
    mock_archive = MagicMock()
    with patch("coreason_archive.server.init_archive", return_value=mock_archive) as mock_init:
        with patch("coreason_archive.server.save_archive") as mock_save:
            with TestClient(app):
                # Triggers startup
                pass
            # Triggers shutdown

            mock_init.assert_called_once()
            mock_save.assert_called_once_with(mock_archive)


def test_get_archive_uninitialized() -> None:
    from coreason_archive.server import get_archive

    mock_request = MagicMock()
    # Mock state without archive
    # Ensure no archive attribute
    del mock_request.app.state.archive

    # Accessing missing attribute might raise AttributeError on MagicMock if not configured?
    # MagicMock usually creates attributes on fly. We need to ensure it DOES NOT have it.
    # We can use a real object or spec=object
    class MockState:
        pass

    mock_request.app.state = MockState()

    with pytest.raises(Exception) as exc:  # It raises HTTPException
        get_archive(mock_request)
    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
