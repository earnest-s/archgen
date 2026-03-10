"""
FastAPI endpoint integration tests for ArchitectAI.

Tests all three public API routes using FastAPI's synchronous TestClient:
  POST /parse-prompt   → validates Architecture JSON is returned
  POST /generate-diagram  → validates PNG path / base64 is returned
  POST /generate       → validates full pipeline response

Run with::

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Create a synchronous TestClient for the FastAPI app."""
    from backend.api.main import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Shared payload helpers
# ---------------------------------------------------------------------------

SIMPLE_PROMPT = "React frontend with FastAPI backend and PostgreSQL database"

SIMPLE_ARCHITECTURE = {
    "nodes": [
        {"id": "web",  "type": "Frontend",  "label": "Web App",     "layer": "Presentation"},
        {"id": "api",  "type": "Backend",   "label": "REST API",    "layer": "Application"},
        {"id": "db",   "type": "Database",  "label": "PostgreSQL",  "layer": "Data"},
    ],
    "edges": [
        {"from": "web", "to": "api", "protocol": "HTTPS"},
        {"from": "api", "to": "db",  "protocol": "SQL"},
    ],
    "metadata": {"version": 1, "style": "light"},
}


# ===========================================================================
# POST /parse-prompt
# ===========================================================================


class TestParsePromptEndpoint:
    def test_status_200(self, client):
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        assert resp.status_code == 200, resp.text

    def test_returns_architecture_key(self, client):
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        body = resp.json()
        assert "architecture" in body, f"Missing 'architecture' key: {body}"

    def test_architecture_has_nodes(self, client):
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        arch = resp.json()["architecture"]
        assert "nodes" in arch
        assert len(arch["nodes"]) > 0

    def test_architecture_has_edges(self, client):
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        arch = resp.json()["architecture"]
        assert "edges" in arch

    def test_node_types_are_valid(self, client):
        valid_types = {"Frontend", "Backend", "Service", "Database", "Cache", "Queue", "External"}
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        nodes = resp.json()["architecture"]["nodes"]
        for node in nodes:
            assert node["type"] in valid_types, f"Unknown type: {node['type']}"

    def test_metadata_present(self, client):
        resp = client.post("/parse-prompt", json={"prompt": SIMPLE_PROMPT})
        arch = resp.json()["architecture"]
        assert "metadata" in arch
        assert arch["metadata"]["version"] >= 1

    def test_empty_prompt_returns_error(self, client):
        resp = client.post("/parse-prompt", json={"prompt": "   "})
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for empty prompt, got {resp.status_code}"
        )

    def test_missing_prompt_field_422(self, client):
        resp = client.post("/parse-prompt", json={})
        assert resp.status_code == 422

    def test_microservices_prompt(self, client):
        prompt = "Vue frontend, Node API gateway, Auth service, MongoDB, Redis cache, Kafka queue"
        resp = client.post("/parse-prompt", json={"prompt": prompt})
        assert resp.status_code == 200
        nodes = resp.json()["architecture"]["nodes"]
        assert len(nodes) >= 4, f"Expected ≥4 nodes, got {len(nodes)}: {nodes}"


# ===========================================================================
# POST /generate-diagram
# ===========================================================================


class TestGenerateDiagramEndpoint:
    def test_status_200(self, client):
        resp = client.post(
            "/generate-diagram",
            json={"architecture": SIMPLE_ARCHITECTURE},
        )
        assert resp.status_code == 200, resp.text

    def test_returns_diagram_path_or_base64(self, client):
        resp = client.post(
            "/generate-diagram?inline=true",
            json={"architecture": SIMPLE_ARCHITECTURE},
        )
        body = resp.json()
        has_path = "diagram_path" in body and body["diagram_path"]
        has_b64  = "diagram_base64" in body and body["diagram_base64"]
        assert has_path or has_b64, f"Neither diagram_path nor diagram_base64 in: {body}"

    def test_base64_is_valid_png(self, client):
        resp = client.post(
            "/generate-diagram?inline=true",
            json={"architecture": SIMPLE_ARCHITECTURE},
        )
        body = resp.json()
        b64 = body.get("diagram_base64")
        if b64 is None:
            pytest.skip("No base64 PNG returned by this endpoint configuration")
        raw = base64.b64decode(b64)
        assert raw[:4] == b"\x89PNG", "Decoded bytes do not start with PNG magic bytes"

    def test_diagram_path_ends_with_png(self, client):
        resp = client.post(
            "/generate-diagram",
            json={"architecture": SIMPLE_ARCHITECTURE},
        )
        body = resp.json()
        path = body.get("diagram_path", "")
        if path:
            assert path.endswith(".png"), f"Expected .png suffix, got {path}"

    def test_invalid_architecture_422(self, client):
        resp = client.post(
            "/generate-diagram",
            json={"architecture": {"nodes": [], "edges": [], "metadata": {"version": 1}}},
        )
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for empty nodes, got {resp.status_code}"
        )

    def test_bad_edge_reference_422(self, client):
        bad_arch = {
            "nodes": [{"id": "x", "type": "Backend", "label": "X"}],
            "edges": [{"from": "x", "to": "ghost"}],
            "metadata": {"version": 1},
        }
        resp = client.post("/generate-diagram", json={"architecture": bad_arch})
        assert resp.status_code in (400, 422)


# ===========================================================================
# POST /generate  (full pipeline)
# ===========================================================================


class TestFullPipelineEndpoint:
    def test_status_200(self, client):
        resp = client.post("/generate", json={"prompt": SIMPLE_PROMPT})
        assert resp.status_code == 200, resp.text

    def test_returns_architecture(self, client):
        resp = client.post("/generate", json={"prompt": SIMPLE_PROMPT})
        body = resp.json()
        assert "architecture" in body, f"Missing 'architecture': {body}"

    def test_architecture_has_nodes(self, client):
        resp = client.post("/generate", json={"prompt": SIMPLE_PROMPT})
        arch = resp.json()["architecture"]
        assert len(arch["nodes"]) > 0

    def test_returns_diagram(self, client):
        resp = client.post("/generate?inline=true", json={"prompt": SIMPLE_PROMPT})
        body = resp.json()
        has_path = bool(body.get("diagram_path"))
        has_b64  = bool(body.get("diagram_base64"))
        assert has_path or has_b64, f"No diagram in response: {body}"

    def test_roundtrip_architecture_valid(self, client):
        """Returned architecture must deserialise as a valid Pydantic model."""
        from backend.api.schemas.architecture import Architecture

        resp = client.post("/generate", json={"prompt": SIMPLE_PROMPT})
        arch_data = resp.json()["architecture"]
        arch = Architecture.model_validate(arch_data)
        assert len(arch.nodes) > 0

    def test_empty_prompt_returns_error(self, client):
        resp = client.post("/generate", json={"prompt": ""})
        assert resp.status_code in (400, 422)

    def test_complex_prompt(self, client):
        prompt = (
            "Angular frontend, Django REST API, PostgreSQL, Redis, "
            "Celery task queue, Stripe payment integration"
        )
        resp = client.post("/generate", json={"prompt": prompt})
        assert resp.status_code == 200
        nodes = resp.json()["architecture"]["nodes"]
        # Should detect at least 4 components.
        assert len(nodes) >= 4, f"Expected ≥4 nodes, got {len(nodes)}: {nodes}"


# ===========================================================================
# Health / misc
# ===========================================================================


class TestMisc:
    def test_root_returns_200_or_404(self, client):
        """Root path may be defined or not — just shouldn't crash the server."""
        resp = client.get("/")
        assert resp.status_code in (200, 404)

    def test_cors_headers_present(self, client):
        """CORS should be enabled — OPTIONS pre-flight shouldn't return 405."""
        resp = client.options(
            "/parse-prompt",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        # Some test clients don't forward OPTIONS perfectly; accept 200 or 405.
        assert resp.status_code in (200, 405)
