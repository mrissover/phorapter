"""Vector store adapter conformance suite.

Every adapter must satisfy this contract. It is parametrized over the store
implementations: the in-memory store always, and the Qdrant adapter only when
the ``integration`` marker is selected and Docker is available. A third-party
adapter claims compliance by importing and parametrizing this suite over its own
factory.

The suite drives the SPI through realistic lifecycles — create, upsert, search,
fetch, the replace/tombstone diff on a shrinking document, delete — and asserts
the normative guarantees: payload-complete hits, deterministic ordering, frozen
config, exact orphan sets.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from phoropter import DEFAULT_GRID, GridSpec, multi_view_slice
from phoropter.errors import (
    CorpusExistsError,
    CorpusMismatchError,
    CorpusNotFoundError,
    DocumentNotFoundError,
)
from phoropter.model import Slice
from phoropter.stores import (
    CorpusConfig,
    DocumentRecord,
    SlicePoint,
    VectorStoreAdapter,
)
from phoropter.stores.memory import InMemoryStore

DIM = 8


def _vector_for(slice_: Slice) -> tuple[float, ...]:
    """A deterministic unit-ish vector seeded by the slice marker (test fixture only)."""
    seed = int(slice_.own_marker[:8], 16)
    return tuple(float((seed >> (i * 3)) & 0x7) + 1.0 for i in range(DIM))


def _points_for(text: str, document_id: str, grid: GridSpec) -> tuple[SlicePoint, ...]:
    doc = multi_view_slice(document_id, text, grid)
    return tuple(SlicePoint(slice=s, vector=_vector_for(s)) for s in doc.slices)


def _record_for(text: str, document_id: str, grid: GridSpec) -> DocumentRecord:
    doc = multi_view_slice(document_id, text, grid)
    return DocumentRecord(
        document_id=document_id,
        codepoint_length=doc.codepoint_length,
        byte_length=doc.byte_length,
        slice_count=len(doc.slices),
        content_marker=doc.slices[-1].own_marker if doc.slices else "",
    )


def _config(name: str = "docs", grid: GridSpec = DEFAULT_GRID) -> CorpusConfig:
    return CorpusConfig(
        name=name,
        grid=grid,
        embedder_fingerprint="fake:deterministic-8",
        dimension=DIM,
        token_counter_id="tiktoken:o200k_base",
    )


# ── store factories: (id, async-context-manager factory) ────────────────────


@contextlib.asynccontextmanager
async def _memory_store() -> AsyncIterator[VectorStoreAdapter]:
    store = InMemoryStore()
    await store.bootstrap()
    yield store


@contextlib.asynccontextmanager
async def _qdrant_store() -> AsyncIterator[VectorStoreAdapter]:
    pytest.importorskip("testcontainers.core.container")
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    from phoropter.stores.qdrant import QdrantStore

    container = DockerContainer("qdrant/qdrant:v1.12.4").with_exposed_ports(6333)
    try:
        container.start()
        # The port mapping is not queryable until Qdrant reports it is serving.
        wait_for_logs(container, "Qdrant HTTP listening", timeout=60)
    except Exception as e:  # docker not available or container never became ready
        with contextlib.suppress(Exception):
            container.stop()
        pytest.skip(f"Docker/Qdrant container unavailable: {e}")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        store = QdrantStore(url=f"http://{host}:{port}", prefix="phoropter_test")
        await store.bootstrap()
        yield store
        with contextlib.suppress(Exception):
            await store.aclose()
    finally:
        container.stop()


StoreFactory = Callable[[], "contextlib.AbstractAsyncContextManager[VectorStoreAdapter]"]

STORE_PARAMS = [
    pytest.param(_memory_store, id="memory"),
    pytest.param(_qdrant_store, id="qdrant", marks=pytest.mark.integration),
]


@pytest.fixture(params=STORE_PARAMS)
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStoreAdapter]:
    factory: StoreFactory = request.param
    async with factory() as adapter:
        yield adapter


class TestCorpusLifecycle:
    async def test_create_get_list_drop(self, store: VectorStoreAdapter) -> None:
        assert await store.list_corpora() == ()
        await store.create_corpus(_config("docs"))
        assert await store.list_corpora() == ("docs",)
        got = await store.get_corpus_meta("docs")
        assert got.dimension == DIM
        assert got.grid.sizes == DEFAULT_GRID.sizes
        await store.drop_corpus("docs")
        assert await store.list_corpora() == ()

    async def test_duplicate_create_rejected(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        with pytest.raises(CorpusExistsError):
            await store.create_corpus(_config("docs"))

    async def test_unknown_corpus_raises(self, store: VectorStoreAdapter) -> None:
        with pytest.raises(CorpusNotFoundError):
            await store.get_corpus_meta("nope")

    async def test_verify_healthy_corpus(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        assert await store.verify_corpus("docs") == []


class TestUpsertAndSearch:
    async def test_search_returns_payload_complete_hits(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        points = _points_for("x" * 300, "doc-a", DEFAULT_GRID)
        await store.upsert_slices("docs", points, grid_fingerprint=DEFAULT_GRID.fingerprint())

        probe = next(p for p in points if p.slice.size == 64)
        hits = await store.search_size("docs", 64, probe.vector, k=5)
        assert hits, "expected at least one hit"
        top = hits[0]
        assert top.slice.own_marker  # marker present
        assert (
            top.slice.descendant_markers == probe.slice.descendant_markers or top.slice.size == 64
        )
        assert top.slice.text
        assert top.rank_in_size == 0
        assert top.score is not None

    async def test_search_is_deterministically_ordered(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        points = _points_for("hello world " * 40, "doc-a", DEFAULT_GRID)
        await store.upsert_slices("docs", points, grid_fingerprint=DEFAULT_GRID.fingerprint())
        probe = next(p for p in points if p.slice.size == 64)
        first = await store.search_size("docs", 64, probe.vector, k=10)
        second = await store.search_size("docs", 64, probe.vector, k=10)
        assert [h.slice.ref for h in first] == [h.slice.ref for h in second]
        scores = [h.score for h in first]
        assert scores == sorted(scores, reverse=True)
        assert [h.rank_in_size for h in first] == list(range(len(first)))

    async def test_upsert_rejects_grid_mismatch(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        other = GridSpec((32, 64))
        points = _points_for("x" * 100, "doc-a", other)
        with pytest.raises(CorpusMismatchError):
            await store.upsert_slices("docs", points, grid_fingerprint=other.fingerprint())

    async def test_upsert_rejects_wrong_dimension(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        doc = multi_view_slice("doc-a", "x" * 64, DEFAULT_GRID)
        bad = tuple(SlicePoint(slice=s, vector=(1.0, 2.0)) for s in doc.slices)  # dim 2 != 8
        with pytest.raises(CorpusMismatchError):
            await store.upsert_slices("docs", bad, grid_fingerprint=DEFAULT_GRID.fingerprint())

    async def test_fetch_by_ids(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        points = _points_for("x" * 300, "doc-a", DEFAULT_GRID)
        await store.upsert_slices("docs", points, grid_fingerprint=DEFAULT_GRID.fingerprint())
        want = [p for p in points if p.slice.size == 128]
        ids = [p.slice.ref.uuid() for p in want]
        fetched = await store.fetch_by_ids("docs", 128, ids)
        assert {p.slice.ref for p in fetched} == {p.slice.ref for p in want}

    async def test_fetch_unknown_ids_omitted(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        from phoropter.model import SliceRef

        missing = SliceRef("ghost", 64, 0).uuid()
        assert await store.fetch_by_ids("docs", 64, [missing]) == []

    async def test_reupsert_is_idempotent(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        points = _points_for("x" * 200, "doc-a", DEFAULT_GRID)
        fp = DEFAULT_GRID.fingerprint()
        await store.upsert_slices("docs", points, grid_fingerprint=fp)
        await store.upsert_slices("docs", points, grid_fingerprint=fp)
        stats = await store.corpus_stats("docs")
        doc = multi_view_slice("doc-a", "x" * 200, DEFAULT_GRID)
        assert stats.total_points == len(doc.slices)


class TestDocumentsAndReplace:
    async def test_document_registry_roundtrip(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        record = _record_for("x" * 300, "doc-a", DEFAULT_GRID)
        await store.put_document_meta("docs", record)
        got = await store.get_document_meta("docs", "doc-a")
        assert got.document_id == "doc-a"
        assert got.slice_count == record.slice_count

    async def test_unknown_document_raises(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        with pytest.raises(DocumentNotFoundError):
            await store.get_document_meta("docs", "ghost")

    async def test_list_documents_paginates(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        for i in range(5):
            await store.put_document_meta("docs", _record_for("x" * 80, f"doc-{i}", DEFAULT_GRID))
        page1 = await store.list_documents("docs", limit=2)
        assert len(page1.records) == 2
        assert page1.next_cursor is not None
        page2 = await store.list_documents("docs", limit=2, cursor=page1.next_cursor)
        assert len(page2.records) == 2
        seen = {r.document_id for r in page1.records} | {r.document_id for r in page2.records}
        assert len(seen) == 4

    async def test_replace_shrink_tombstones_exact_orphans(self, store: VectorStoreAdapter) -> None:
        # The replace workhorse: after re-slicing a shrunken document, the orphan
        # set is exactly (old ids minus new ids), per size.
        await store.create_corpus(_config("docs"))
        fp = DEFAULT_GRID.fingerprint()
        big = _points_for("x" * 1024, "doc-a", DEFAULT_GRID)
        await store.upsert_slices("docs", big, grid_fingerprint=fp)
        await store.put_document_meta("docs", _record_for("x" * 1024, "doc-a", DEFAULT_GRID))

        old_ids = await store.list_point_ids("docs", "doc-a")

        small = _points_for("x" * 100, "doc-a", DEFAULT_GRID)
        new_ids_by_size: dict[int, set] = {}
        for p in small:
            new_ids_by_size.setdefault(p.slice.size, set()).add(p.slice.ref.uuid())
        await store.upsert_slices("docs", small, grid_fingerprint=fp)

        orphans = {
            size: old_ids.get(size, set()) - new_ids_by_size.get(size, set())
            for size in DEFAULT_GRID.sizes
        }
        await store.delete_points("docs", {s: ids for s, ids in orphans.items() if ids})

        remaining = await store.list_point_ids("docs", "doc-a")
        for size in DEFAULT_GRID.sizes:
            assert remaining.get(size, set()) == new_ids_by_size.get(size, set())

    async def test_delete_document_removes_points_and_registry(
        self, store: VectorStoreAdapter
    ) -> None:
        await store.create_corpus(_config("docs"))
        fp = DEFAULT_GRID.fingerprint()
        await store.upsert_slices(
            "docs", _points_for("x" * 200, "doc-a", DEFAULT_GRID), grid_fingerprint=fp
        )
        await store.put_document_meta("docs", _record_for("x" * 200, "doc-a", DEFAULT_GRID))
        await store.delete_document("docs", "doc-a")
        with pytest.raises(DocumentNotFoundError):
            await store.get_document_meta("docs", "doc-a")
        ids = await store.list_point_ids("docs", "doc-a")
        assert all(len(v) == 0 for v in ids.values())

    async def test_delete_unknown_document_raises(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        with pytest.raises(DocumentNotFoundError):
            await store.delete_document("docs", "ghost")


class TestStats:
    async def test_corpus_stats(self, store: VectorStoreAdapter) -> None:
        await store.create_corpus(_config("docs"))
        fp = DEFAULT_GRID.fingerprint()
        await store.upsert_slices(
            "docs", _points_for("x" * 300, "doc-a", DEFAULT_GRID), grid_fingerprint=fp
        )
        await store.put_document_meta("docs", _record_for("x" * 300, "doc-a", DEFAULT_GRID))
        stats = await store.corpus_stats("docs")
        assert stats.document_count == 1
        doc = multi_view_slice("doc-a", "x" * 300, DEFAULT_GRID)
        assert stats.total_points == len(doc.slices)
        assert dict(stats.points_by_size)[64] == len(doc.slices_of_size(64))


def test_config_is_a_frozen_value_type() -> None:
    # Config is immutable; a "change" is a new value, never a mutation.
    cfg = _config("docs")
    with pytest.raises(FrozenInstanceError):
        cfg.dimension = 99  # type: ignore[misc]
    assert replace(cfg, dimension=16).dimension == 16
