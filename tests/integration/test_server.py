"""Server integration tests against a real Qdrant container and the fake embedder.

These require Docker; they are marked ``integration`` and excluded from the
default lane. The Qdrant container is started once per module (the container spin
is the slow part), and each test uses a distinct corpus/prefix so they do not
interfere. The embedder is always the deterministic :class:`FakeEmbedder` — no
Ollama in CI.

Coverage: corpus lifecycle and duplicate conflict, document PUT/replace with
exact tombstone counts, delete, a budgeted query returning a full trace, a
partial fan-out (one size collection dropped mid-flight → 200 ``partial:true``),
a total fan-out failure (all size collections dropped → 503), an
``EMBEDDER_MISMATCH`` when a differently-dimensioned embedder feeds a pinned
corpus, and the MCP query/list round-trip with write tools gated off.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient

from phoropter.config import (
    DefaultsSettings,
    EmbedderSettings,
    McpSettings,
    Settings,
    StoreSettings,
)
from phoropter.embed import FakeEmbedder
from phoropter.server.rest import create_app
from phoropter.service.core import ServiceCore

pytestmark = pytest.mark.integration

DIM = 32
GRID = (64, 128, 256)
_DOC = "The quick brown fox jumps over the lazy dog. " * 20


@pytest.fixture(scope="module")
def qdrant_url() -> Iterator[str]:
    pytest.importorskip("testcontainers.core.container")
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = DockerContainer("qdrant/qdrant:v1.12.4").with_exposed_ports(6333)
    try:
        container.start()
        wait_for_logs(container, "Qdrant HTTP listening", timeout=60)
    except Exception as e:  # docker not available or never became ready
        with contextlib.suppress(Exception):
            container.stop()
        pytest.skip(f"Docker/Qdrant container unavailable: {e}")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        yield f"http://{host}:{port}"
    finally:
        container.stop()


def _settings(qdrant_url: str, *, prefix: str, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "store": StoreSettings(kind="qdrant", url=qdrant_url, prefix=prefix),
        "embedder": EmbedderSettings(provider="fake", model=f"deterministic-{DIM}"),
        "defaults": DefaultsSettings(grid_sizes=GRID, top_k_per_size=10),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@contextlib.asynccontextmanager
async def _core(settings: Settings, *, dimension: int = DIM) -> AsyncIterator[ServiceCore]:
    from phoropter.config import build_store

    store = build_store(settings)
    core = ServiceCore(store=store, embedder=FakeEmbedder(dimension), settings=settings)
    await core.startup()
    try:
        yield core
    finally:
        await core.aclose()


def _client(settings: Settings, core: ServiceCore) -> TestClient:
    app = create_app(settings, core=core, run_startup=False)
    return TestClient(app)


def _prefix() -> str:
    return f"pt_{uuid.uuid4().hex[:8]}"


# ── lifecycle and documents ──────────────────────────────────────────────────


async def test_corpus_and_document_lifecycle(qdrant_url: str) -> None:
    settings = _settings(qdrant_url, prefix=_prefix())
    async with _core(settings) as core:
        client = _client(settings, core)

        assert client.post("/v1/corpora", json={"name": "docs"}).status_code == 201
        assert client.post("/v1/corpora", json={"name": "docs"}).status_code == 409
        assert client.get("/v1/corpora").json()["corpora"] == ["docs"]

        put = client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
        assert put.status_code == 200
        slice_count = put.json()["slice_count"]

        stats = client.get("/v1/corpora/docs").json()
        assert sum(s["count"] for s in stats["points_by_size"]) == slice_count
        assert stats["degraded"] == []

        client.delete("/v1/corpora/docs/documents/d1")
        assert client.get("/v1/corpora/docs/documents/d1").status_code == 404
        assert client.delete("/v1/corpora/docs").status_code == 204


async def test_replace_tombstones_exact_orphans(qdrant_url: str) -> None:
    settings = _settings(qdrant_url, prefix=_prefix())
    async with _core(settings) as core:
        client = _client(settings, core)
        client.post("/v1/corpora", json={"name": "docs"})

        big = client.put("/v1/corpora/docs/documents/d1", json={"text": "x" * 1024}).json()
        small = client.put("/v1/corpora/docs/documents/d1", json={"text": "x" * 100}).json()
        assert small["slice_count"] < big["slice_count"]

        total = sum(s["count"] for s in client.get("/v1/corpora/docs").json()["points_by_size"])
        assert total == small["slice_count"]


# ── query ────────────────────────────────────────────────────────────────────


async def test_budgeted_query_returns_trace(qdrant_url: str) -> None:
    settings = _settings(qdrant_url, prefix=_prefix())
    async with _core(settings) as core:
        client = _client(settings, core)
        client.post("/v1/corpora", json={"name": "docs"})
        client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})

        r = client.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 800})
        assert r.status_code == 200
        body = r.json()
        assert body["partial"] is False
        assert body["results"]
        assert body["budget"]["used"] <= 800
        assert body["trace"]["final"]["tokens_used"] == body["budget"]["used"]
        # Every fan-out size reported success.
        assert all(f["ok"] for f in body["fanout"])


async def test_partial_fanout_when_one_size_dropped(qdrant_url: str) -> None:
    prefix = _prefix()
    settings = _settings(qdrant_url, prefix=prefix)
    async with _core(settings) as core:
        client = _client(settings, core)
        client.post("/v1/corpora", json={"name": "docs"})
        client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})

        # Drop one size's collection out from under the running server.
        store = core.store
        await store._client.delete_collection(f"{prefix}__docs__s64")  # type: ignore[attr-defined]

        r = client.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 800})
        assert r.status_code == 200
        body = r.json()
        assert body["partial"] is True
        failed = [f for f in body["fanout"] if not f["ok"]]
        assert any(f["size"] == 64 for f in failed)
        assert body["results"]  # the surviving sizes still answer


async def test_total_fanout_failure_returns_503(qdrant_url: str) -> None:
    prefix = _prefix()
    settings = _settings(qdrant_url, prefix=prefix)
    async with _core(settings) as core:
        client = _client(settings, core)
        client.post("/v1/corpora", json={"name": "docs"})
        client.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})

        store = core.store
        for size in GRID:
            await store._client.delete_collection(f"{prefix}__docs__s{size}")  # type: ignore[attr-defined]

        r = client.post("/v1/corpora/docs/query", json={"query": "fox", "token_budget": 800})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "STORE_UNAVAILABLE"


# ── embedder mismatch ────────────────────────────────────────────────────────


async def test_embedder_mismatch_on_wrong_dimension(qdrant_url: str) -> None:
    prefix = _prefix()
    # Create the corpus with a DIM-dimensioned embedder.
    settings = _settings(qdrant_url, prefix=prefix)
    async with _core(settings) as core:
        client = _client(settings, core)
        client.post("/v1/corpora", json={"name": "docs"})

    # Reopen with a differently-dimensioned embedder feeding the pinned corpus.
    async with _core(settings, dimension=DIM * 2) as core2:
        client2 = _client(settings, core2)
        r = client2.put("/v1/corpora/docs/documents/d1", json={"text": _DOC})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "EMBEDDER_MISMATCH"


# ── MCP ──────────────────────────────────────────────────────────────────────


async def test_mcp_roundtrip_and_write_tools_gated(qdrant_url: str) -> None:
    from fastmcp import Client

    from phoropter.server.mcp import build_mcp

    settings = _settings(qdrant_url, prefix=_prefix(), mcp=McpSettings(enable_document_tools=False))
    async with _core(settings) as core:
        await core.corpora.create("docs")
        await core.documents.put("docs", "d1", _DOC)

        mcp = build_mcp(core)
        async with Client(mcp) as mc:
            names = {t.name for t in await mc.list_tools()}
            assert {"phoropter_query", "phoropter_list_corpora"} <= names
            assert "phoropter_add_document" not in names  # gated off

            listed = await mc.call_tool("phoropter_list_corpora", {})
            assert listed.data == {"corpora": ["docs"]}

            result = await mc.call_tool(
                "phoropter_query", {"corpus": "docs", "query": "fox", "token_budget": 500}
            )
            assert result.data["corpus"] == "docs"
            assert result.data["text"]
