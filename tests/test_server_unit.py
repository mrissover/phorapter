"""Server unit tests over the in-memory store and the fake embedder (no Docker).

These run in the default lane. They exercise the full REST lifecycle through a
FastAPI ``TestClient`` backed by :class:`~phoropter.stores.memory.InMemoryStore`
and :class:`~phoropter.embed.FakeEmbedder`, plus the MCP surface via the
in-memory FastMCP client, the error envelope shapes, and optional bearer auth.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phoropter.config import (
    DefaultsSettings,
    EmbedderSettings,
    McpSettings,
    ServerSettings,
    Settings,
    StoreSettings,
)
from phoropter.embed import FakeEmbedder
from phoropter.server.rest import create_app
from phoropter.service.core import ServiceCore
from phoropter.stores.memory import InMemoryStore

DIM = 32
GRID = (64, 128, 256)


def _settings(**overrides: object) -> Settings:
    base = {
        "store": StoreSettings(kind="memory"),
        "embedder": EmbedderSettings(provider="fake", model=f"deterministic-{DIM}"),
        "defaults": DefaultsSettings(grid_sizes=GRID, top_k_per_size=10),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _make_client(settings: Settings) -> TestClient:
    core = ServiceCore(store=InMemoryStore(), embedder=FakeEmbedder(DIM), settings=settings)
    app = create_app(settings, core=core, run_startup=False)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _make_client(_settings())


_DOC = "The quick brown fox jumps over the lazy dog. " * 20


# ── health / info ────────────────────────────────────────────────────────────


def test_liveness(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_ok(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["store"] is True
    assert body["embedder"] is True


def test_info(client: TestClient) -> None:
    r = client.get("/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "phoropter"
    assert "memory" in body["stores_available"]
    assert "fake" in body["embedders_available"]


def test_request_id_echoed(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.headers.get("X-Request-Id")
    supplied = client.get("/healthz", headers={"X-Request-Id": "abc-123"})
    assert supplied.headers["X-Request-Id"] == "abc-123"


# ── corpus lifecycle ─────────────────────────────────────────────────────────


def test_corpus_create_list_inspect_drop(client: TestClient) -> None:
    assert client.get("/v1/corpora").json() == {"corpora": []}

    r = client.post("/v1/corpora", json={"name": "docs"})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "docs"
    assert body["dimension"] == DIM
    assert body["grid"]["sizes"] == list(GRID)
    assert body["tokenizer"] == "tiktoken:o200k_base"

    assert client.get("/v1/corpora").json() == {"corpora": ["docs"]}

    inspect = client.get("/v1/corpora/docs")
    assert inspect.status_code == 200
    assert inspect.json()["document_count"] == 0
    assert inspect.json()["degraded"] == []

    assert client.delete("/v1/corpora/docs").status_code == 204
    assert client.get("/v1/corpora").json() == {"corpora": []}


def test_duplicate_corpus_conflict(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    r = client.post("/v1/corpora", json={"name": "docs"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "CORPUS_EXISTS"


def test_unknown_corpus_404(client: TestClient) -> None:
    r = client.get("/v1/corpora/ghost")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CORPUS_NOT_FOUND"


def test_create_with_bad_grid_422(client: TestClient) -> None:
    r = client.post("/v1/corpora", json={"name": "bad", "grid_sizes": [100, 150]})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_GRID"


def test_create_with_bad_tokenizer_422(client: TestClient) -> None:
    r = client.post("/v1/corpora", json={"name": "bad", "tokenizer": "nope:xyz"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNKNOWN_TOKENIZER"


# ── documents ────────────────────────────────────────────────────────────────


def test_document_put_get_list_delete(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})

    r = client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
    assert r.status_code == 200
    rec = r.json()
    assert rec["document_id"] == "d1"
    assert rec["slice_count"] > 0
    assert len(rec["content_marker"]) == 64

    got = client.get("/v1/corpora/docs/documents/d1")
    assert got.status_code == 200
    assert got.json()["slice_count"] == rec["slice_count"]

    listing = client.get("/v1/corpora/docs/documents")
    assert listing.status_code == 200
    assert [x["document_id"] for x in listing.json()["records"]] == ["d1"]

    assert client.delete("/v1/corpora/docs/documents/d1").status_code == 204
    assert client.get("/v1/corpora/docs/documents/d1").status_code == 404


def test_document_replace_shrinks_slice_count(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    big = client.put("/v1/corpora/docs/documents/d1", json={"text": "x" * 512})
    small = client.put("/v1/corpora/docs/documents/d1", json={"text": "x" * 64})
    assert small.json()["slice_count"] < big.json()["slice_count"]
    # The corpus point count must reflect the tombstoning, not accumulate.
    stats = client.get("/v1/corpora/docs").json()
    total = sum(s["count"] for s in stats["points_by_size"])
    assert total == small.json()["slice_count"]


def test_document_metadata_roundtrip(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    r = client.put(
        "/v1/corpora/docs/documents/d1",
        json={"text": "hello", "metadata": [{"key": "source", "value": "unit"}]},
    )
    assert r.status_code == 200
    got = client.get("/v1/corpora/docs/documents/d1").json()
    assert got["metadata"] == [{"key": "source", "value": "unit"}]


def test_document_too_large_413(client: TestClient) -> None:
    settings = _settings()
    settings.limits.max_document_codepoints = 10
    small_client = _make_client(settings)
    small_client.post("/v1/corpora", json={"name": "docs"})
    r = small_client.put("/v1/corpora/docs/documents/d1", json={"text": "x" * 100})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "DOCUMENT_TOO_LARGE"


def test_document_put_unknown_corpus_404(client: TestClient) -> None:
    r = client.put("/v1/corpora/ghost/documents/d1", json={"text": "hi"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CORPUS_NOT_FOUND"


def test_batch_documents(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    r = client.post(
        "/v1/corpora/docs/documents/batch",
        json={
            "documents": [
                {"document_id": "a", "text": "alpha text here"},
                {"document_id": "b", "text": "beta text here"},
            ]
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["document_id"] for x in results] == ["a", "b"]
    assert all(x["ok"] for x in results)


def test_batch_over_limit_413(client: TestClient) -> None:
    settings = _settings()
    settings.limits.max_batch_documents = 1
    c = _make_client(settings)
    c.post("/v1/corpora", json={"name": "docs"})
    r = c.post(
        "/v1/corpora/docs/documents/batch",
        json={"documents": [{"document_id": "a", "text": "x"}, {"document_id": "b", "text": "y"}]},
    )
    assert r.status_code == 413


# ── query ────────────────────────────────────────────────────────────────────


def test_query_lifecycle_with_trace(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})

    r = client.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 800})
    assert r.status_code == 200
    body = r.json()
    assert body["corpus"] == "docs"
    assert body["strategy"] == "greedy_upward"
    assert body["partial"] is False
    assert body["budget"]["limit"] == 800
    assert body["budget"]["counter"] == "tiktoken:o200k_base"
    assert body["results"]  # at least one slice
    assert body["budget"]["used"] <= 800

    first = body["results"][0]
    assert set(first["coords"]) == {
        "document_id",
        "size",
        "codepoint_offset",
        "codepoint_length",
        "codepoint_end",
        "own_marker",
    }
    assert first["provenance"] in {"retrieved", "traded_up"}
    assert first["selection"]["action"] in {"kept", "traded_up"}
    assert "retrieval" in first and "evidence" in first

    trace = body["trace"]
    assert set(trace) >= {
        "fusion",
        "forest",
        "dedup",
        "initial_pack",
        "trade_ups",
        "rejections",
        "final",
    }
    assert trace["final"]["tokens_used"] == body["budget"]["used"]


def test_query_dedup_only_without_budget(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
    r = client.post("/v1/corpora/docs/query", json={"query": "dog"})
    assert r.status_code == 200
    body = r.json()
    assert body["budget"]["limit"] is None
    # No trade-up happens without a budget: every result is kept.
    assert all(res["selection"]["action"] == "kept" for res in body["results"])


def test_query_budget_too_small_returns_empty_200(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
    r = client.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert any("too small" in w for w in body["warnings"])


def test_query_budget_capped(client: TestClient) -> None:
    settings = _settings()
    settings.limits.max_token_budget = 50
    c = _make_client(settings)
    c.post("/v1/corpora", json={"name": "docs"})
    c.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
    r = c.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 999999})
    assert r.status_code == 200
    body = r.json()
    assert body["budget"]["limit"] == 50
    assert any("capped" in w for w in body["warnings"])


def test_query_unknown_corpus_404(client: TestClient) -> None:
    r = client.post("/v1/corpora/ghost/query", json={"query": "x"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "CORPUS_NOT_FOUND"


def test_query_bad_tokenizer_422(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
    r = client.post("/v1/corpora/docs/query", json={"query": "x", "tokenizer": "nope:xyz"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNKNOWN_TOKENIZER"


def test_query_missing_field_422_envelope(client: TestClient) -> None:
    client.post("/v1/corpora", json={"name": "docs"})
    r = client.post("/v1/corpora/docs/query", json={})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "request_id" in body["error"]


# ── error envelope shape ─────────────────────────────────────────────────────


def test_error_envelope_shape(client: TestClient) -> None:
    r = client.get("/v1/corpora/ghost")
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details", "request_id"}
    assert body["error"]["request_id"] == r.headers["X-Request-Id"]


# ── auth ─────────────────────────────────────────────────────────────────────


def test_auth_off_by_default(client: TestClient) -> None:
    # No api_key configured: every endpoint is open.
    assert client.get("/v1/corpora").status_code == 200


def test_auth_enforced_when_key_set() -> None:
    settings = _settings(server=ServerSettings(api_key="secret"))
    c = _make_client(settings)
    # Probes stay public.
    assert c.get("/healthz").status_code == 200
    assert c.get("/v1/health").status_code == 200
    # Everything else needs the token.
    assert c.get("/v1/corpora").status_code == 401
    assert c.get("/v1/corpora").json()["error"]["code"] == "UNAUTHORIZED"
    ok = c.get("/v1/corpora", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    bad = c.get("/v1/corpora", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401


# ── MCP surface ──────────────────────────────────────────────────────────────


async def test_mcp_query_and_list_roundtrip() -> None:
    from fastmcp import Client

    from phoropter.server.mcp import build_mcp

    settings = _settings()
    core = ServiceCore(store=InMemoryStore(), embedder=FakeEmbedder(DIM), settings=settings)
    await core.corpora.create("docs")
    await core.documents.put("docs", "d1", _DOC)

    mcp = build_mcp(core)
    async with Client(mcp) as mc:
        names = {t.name for t in await mc.list_tools()}
        assert {"phoropter_query", "phoropter_list_corpora"} <= names
        # Write tools are OFF by default.
        assert "phoropter_add_document" not in names
        assert "phoropter_delete_document" not in names

        listed = await mc.call_tool("phoropter_list_corpora", {})
        assert listed.data == {"corpora": ["docs"]}

        result = await mc.call_tool(
            "phoropter_query", {"corpus": "docs", "query": "fox", "token_budget": 500}
        )
        data = result.data
        assert data["corpus"] == "docs"
        assert isinstance(data["results"], list)
        assert data["text"]  # a reader-facing text block is attached


async def test_mcp_write_tools_registered_when_enabled() -> None:
    from fastmcp import Client

    from phoropter.server.mcp import build_mcp

    settings = _settings(mcp=McpSettings(enable_document_tools=True))
    core = ServiceCore(store=InMemoryStore(), embedder=FakeEmbedder(DIM), settings=settings)
    await core.corpora.create("docs")

    mcp = build_mcp(core)
    async with Client(mcp) as mc:
        names = {t.name for t in await mc.list_tools()}
        assert "phoropter_add_document" in names
        assert "phoropter_delete_document" in names
        await mc.call_tool(
            "phoropter_add_document",
            {"corpus": "docs", "document_id": "d1", "text": "hello world"},
        )
        rec = await core.documents.get("docs", "d1")
        assert rec.document_id == "d1"
