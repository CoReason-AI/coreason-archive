# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from coreason_archive.server import app
from coreason_archive.models import MemoryScope
from coreason_archive.matchmaker import SearchResult, MatchStrategy

# Create a mock archive that will be reused or we can create new ones per test
# But since we patch init_archive, we can control what it returns.

def test_health():
    mock_archive = MagicMock()
    mock_archive.vector_store.thoughts = [1, 2, 3] # len 3
    mock_archive.graph_store.graph.nodes = [1, 2] # len 2

    with patch("coreason_archive.server.init_archive", return_value=mock_archive) as mock_init:
        with patch("coreason_archive.server.save_archive") as mock_save:
            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["vector_store_size"] == 3
                assert data["graph_nodes"] == 2

def test_add_thought():
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
                    "scope": "USER"
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

def test_search():
    mock_archive = MagicMock()
    mock_archive.smart_lookup = AsyncMock()

    result_obj = SearchResult(
        strategy=MatchStrategy.EXACT_HIT,
        thought=None,
        score=1.0,
        content={"message": "Found it"}
    )
    mock_archive.smart_lookup.return_value = result_obj

    with patch("coreason_archive.server.init_archive", return_value=mock_archive):
        with patch("coreason_archive.server.save_archive"):
            with TestClient(app) as client:
                context_payload = {
                    "user_id": "test_user",
                    "email": "test@example.com",
                    "groups": ["group1"]
                }
                payload = {
                    "query": "Hello",
                    "context": context_payload
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

def test_lifespan_save():
    mock_archive = MagicMock()
    with patch("coreason_archive.server.init_archive", return_value=mock_archive) as mock_init:
        with patch("coreason_archive.server.save_archive") as mock_save:
            with TestClient(app) as client:
                # Triggers startup
                pass
            # Triggers shutdown

            mock_init.assert_called_once()
            mock_save.assert_called_once_with(mock_archive)
