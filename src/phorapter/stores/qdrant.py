"""Qdrant store adapter — the default production backend (requires the ``qdrant`` extra).

Storage realization of the logical (corpus, size) coordinates:

- one collection per (corpus, size): ``{prefix}__{corpus}__s{size}`` — top-k
  retrieval is always per size, each size gets its own HNSW geometry, and a
  single unhealthy collection degrades exactly one size (the unit of the
  partial-failure policy);
- ``{prefix}__meta`` — one point per corpus holding its frozen
  :class:`~phorapter.stores.CorpusConfig`;
- ``{prefix}__docs`` — the document registry (one point per document).

Slice payloads follow the schema_version 1 contract (see docs/adapters.md):
descendant markers travel packed via
:func:`~phorapter.stores.pack_markers`, and document-level metadata lives on
the registry record, never per slice. Every slice collection carries a keyword
payload index on ``document_id``.
"""

from __future__ import annotations

import math
import uuid as _uuid
from typing import TYPE_CHECKING, Any

from phorapter.errors import (
    CorpusExistsError,
    CorpusMismatchError,
    CorpusNotFoundError,
    DocumentNotFoundError,
    StoreError,
)
from phorapter.ids import PHORAPTER_NAMESPACE
from phorapter.model import HitProvenance, RetrievedHit, Slice
from phorapter.stores import (
    PAYLOAD_SCHEMA_VERSION,
    CorpusConfig,
    CorpusStats,
    DocumentPage,
    DocumentRecord,
    SlicePoint,
    VectorStoreAdapter,
    pack_markers,
    unpack_markers,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

    from qdrant_client import AsyncQdrantClient

__all__ = ["QdrantStore"]

_INSTALL_HINT = 'QdrantStore requires the qdrant extra: pip install "phorapter[qdrant]"'


def _escape(field: str) -> str:
    return field.replace("\\", "\\\\").replace("|", "\\|")


def _registry_uuid(kind: str, *parts: str) -> _uuid.UUID:
    """Deterministic id for a meta/registry point (same namespace, distinct tag space)."""
    name = "|".join(["phorapter-registry", kind, *(_escape(p) for p in parts)])
    return _uuid.uuid5(PHORAPTER_NAMESPACE, name)


def _slice_payload(s: Slice) -> dict[str, Any]:
    """The schema_version 1 slice payload. Field set is normative; see docs/adapters.md."""
    return {
        "document_id": s.document_id,
        "size": s.size,
        "codepoint_offset": s.codepoint_offset,
        "codepoint_length": s.codepoint_length,
        "byte_offset": s.byte_offset,
        "byte_length": s.byte_length,
        "document_codepoint_length": s.document_codepoint_length,
        "text": s.text,
        "own_marker": s.own_marker,
        "descendant_markers": pack_markers(s.descendant_markers),
        "token_count": s.token_count,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
    }


def _slice_from_payload(payload: Mapping[str, Any]) -> Slice:
    return Slice(
        document_id=payload["document_id"],
        size=payload["size"],
        codepoint_offset=payload["codepoint_offset"],
        codepoint_length=payload["codepoint_length"],
        byte_offset=payload["byte_offset"],
        byte_length=payload["byte_length"],
        document_codepoint_length=payload["document_codepoint_length"],
        text=payload["text"],
        own_marker=payload["own_marker"],
        descendant_markers=unpack_markers(payload["descendant_markers"]),
        token_count=payload["token_count"],
    )


def _record_payload(corpus: str, record: DocumentRecord) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "document_id": record.document_id,
        "codepoint_length": record.codepoint_length,
        "byte_length": record.byte_length,
        "slice_count": record.slice_count,
        "content_marker": record.content_marker,
        # A list of pairs, not an object: key order is part of the record.
        "metadata": [[key, value] for key, value in record.metadata],
        "schema_version": PAYLOAD_SCHEMA_VERSION,
    }


def _record_from_payload(payload: Mapping[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        document_id=payload["document_id"],
        codepoint_length=payload["codepoint_length"],
        byte_length=payload["byte_length"],
        slice_count=payload["slice_count"],
        content_marker=payload["content_marker"],
        metadata=tuple((key, value) for key, value in payload["metadata"]),
    )


class QdrantStore(VectorStoreAdapter):
    """:class:`~phorapter.stores.VectorStoreAdapter` over ``qdrant-client``'s async client.

    ``prefix`` namespaces every collection this store touches, so multiple
    phorapter deployments (or a test run) can share one Qdrant instance
    without collisions. ``bootstrap()`` must run once per deployment before
    corpora are created.
    """

    _SCROLL_PAGE = 512

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        prefix: str = "phorapter",
        timeout_s: float = 10.0,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient, models
            from qdrant_client.http.exceptions import (
                ApiException,
                ResponseHandlingException,
                UnexpectedResponse,
            )
        except ImportError as e:
            raise ImportError(_INSTALL_HINT) from e
        self._models = models
        self._backend_errors: tuple[type[Exception], ...] = (
            ApiException,
            ResponseHandlingException,
            UnexpectedResponse,
            OSError,
            TimeoutError,
        )
        self._prefix = prefix
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            url=url, api_key=api_key, timeout=max(1, math.ceil(timeout_s))
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ── naming ─────────────────────────────────────────────────────────────

    def _slice_collection(self, corpus: str, size: int) -> str:
        return f"{self._prefix}__{corpus}__s{size}"

    @property
    def _meta_collection(self) -> str:
        return f"{self._prefix}__meta"

    @property
    def _docs_collection(self) -> str:
        return f"{self._prefix}__docs"

    async def _run(self, description: str, awaitable: Awaitable[Any]) -> Any:
        try:
            return await awaitable
        except self._backend_errors as e:
            raise StoreError(f"qdrant {description} failed: {e}") from e

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
        except self._backend_errors:
            return False
        return True

    async def bootstrap(self) -> None:
        m = self._models
        for name, indexed_fields in (
            (self._meta_collection, ()),
            (self._docs_collection, ("corpus", "document_id")),
        ):
            if not await self._run("collection check", self._client.collection_exists(name)):
                # Registry collections carry payloads only; the 1-dim dot-product
                # vector is a placeholder every Qdrant version accepts.
                await self._run(
                    "bootstrap",
                    self._client.create_collection(
                        collection_name=name,
                        vectors_config=m.VectorParams(size=1, distance=m.Distance.DOT),
                    ),
                )
                for field in indexed_fields:
                    await self._run(
                        "bootstrap index",
                        self._client.create_payload_index(
                            collection_name=name,
                            field_name=field,
                            field_schema=m.PayloadSchemaType.KEYWORD,
                        ),
                    )

    async def create_corpus(self, config: CorpusConfig) -> None:
        m = self._models
        await self.bootstrap()
        meta_id = str(_registry_uuid("corpus", config.name))
        existing = await self._run(
            "meta read",
            self._client.retrieve(
                collection_name=self._meta_collection, ids=[meta_id], with_payload=False
            ),
        )
        if existing:
            raise CorpusExistsError(f"corpus {config.name!r} already exists")
        for size in config.grid.sizes:
            name = self._slice_collection(config.name, size)
            if not await self._run("collection check", self._client.collection_exists(name)):
                await self._run(
                    "collection create",
                    self._client.create_collection(
                        collection_name=name,
                        vectors_config=m.VectorParams(
                            size=config.dimension, distance=m.Distance.COSINE
                        ),
                    ),
                )
                await self._run(
                    "index create",
                    self._client.create_payload_index(
                        collection_name=name,
                        field_name="document_id",
                        field_schema=m.PayloadSchemaType.KEYWORD,
                    ),
                )
        await self._run(
            "meta write",
            self._client.upsert(
                collection_name=self._meta_collection,
                points=[
                    m.PointStruct(
                        id=meta_id,
                        vector=[1.0],
                        payload={
                            "name": config.name,
                            "grid_sizes": list(config.grid.sizes),
                            "embedder_fingerprint": config.embedder_fingerprint,
                            "dimension": config.dimension,
                            "token_counter_id": config.token_counter_id,
                            "schema_version": PAYLOAD_SCHEMA_VERSION,
                        },
                    )
                ],
                wait=True,
            ),
        )

    async def drop_corpus(self, corpus: str) -> None:
        m = self._models
        config = await self.get_corpus_meta(corpus)
        for size in config.grid.sizes:
            name = self._slice_collection(corpus, size)
            if await self._run("collection check", self._client.collection_exists(name)):
                await self._run("collection drop", self._client.delete_collection(name))
        await self._run(
            "docs purge",
            self._client.delete(
                collection_name=self._docs_collection,
                points_selector=m.FilterSelector(filter=self._corpus_filter(corpus)),
                wait=True,
            ),
        )
        await self._run(
            "meta delete",
            self._client.delete(
                collection_name=self._meta_collection,
                points_selector=m.PointIdsList(points=[str(_registry_uuid("corpus", corpus))]),
                wait=True,
            ),
        )

    async def get_corpus_meta(self, corpus: str) -> CorpusConfig:
        if not await self._run(
            "collection check", self._client.collection_exists(self._meta_collection)
        ):
            raise CorpusNotFoundError(f"corpus {corpus!r} does not exist (store not bootstrapped)")
        records = await self._run(
            "meta read",
            self._client.retrieve(
                collection_name=self._meta_collection,
                ids=[str(_registry_uuid("corpus", corpus))],
                with_payload=True,
            ),
        )
        if not records:
            raise CorpusNotFoundError(f"corpus {corpus!r} does not exist")
        payload = records[0].payload or {}
        from phorapter.grid import GridSpec

        return CorpusConfig(
            name=payload["name"],
            grid=GridSpec(tuple(payload["grid_sizes"])),
            embedder_fingerprint=payload["embedder_fingerprint"],
            dimension=payload["dimension"],
            token_counter_id=payload["token_counter_id"],
        )

    async def list_corpora(self) -> tuple[str, ...]:
        if not await self._run(
            "collection check", self._client.collection_exists(self._meta_collection)
        ):
            return ()
        names: list[str] = []
        offset: Any = None
        while True:
            records, offset = await self._run(
                "meta scroll",
                self._client.scroll(
                    collection_name=self._meta_collection,
                    limit=self._SCROLL_PAGE,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            names.extend((record.payload or {})["name"] for record in records)
            if offset is None:
                break
        return tuple(sorted(names))

    async def verify_corpus(self, corpus: str) -> list[str]:
        config = await self.get_corpus_meta(corpus)
        m = self._models
        reasons: list[str] = []
        for size in config.grid.sizes:
            name = self._slice_collection(corpus, size)
            if not await self._run("collection check", self._client.collection_exists(name)):
                reasons.append(f"collection {name!r} (size {size}) is missing")
                continue
            info = await self._run("collection info", self._client.get_collection(name))
            params = info.config.params.vectors
            if not isinstance(params, m.VectorParams):
                reasons.append(f"collection {name!r} has an unexpected vector configuration")
                continue
            if params.size != config.dimension:
                reasons.append(
                    f"collection {name!r} has dimension {params.size}, "
                    f"corpus is pinned to {config.dimension}"
                )
            if params.distance != m.Distance.COSINE:
                reasons.append(f"collection {name!r} uses distance {params.distance}, not cosine")
        return reasons

    # ── writes ─────────────────────────────────────────────────────────────

    async def upsert_slices(
        self, corpus: str, points: Sequence[SlicePoint], *, grid_fingerprint: str
    ) -> None:
        m = self._models
        config = await self.get_corpus_meta(corpus)
        if grid_fingerprint != config.grid.fingerprint():
            raise CorpusMismatchError(
                f"grid fingerprint {grid_fingerprint!r} does not match corpus "
                f"{corpus!r} (created with sizes {config.grid.sizes})"
            )
        by_size: dict[int, list[Any]] = {}
        for point in points:
            if point.slice.size not in config.grid.sizes:
                raise CorpusMismatchError(
                    f"slice size {point.slice.size} is not a grid size of corpus {corpus!r}"
                )
            if point.vector is None:
                raise ValueError(f"point {point.slice.ref} has no vector; upserts require one")
            if len(point.vector) != config.dimension:
                raise CorpusMismatchError(
                    f"vector dimension {len(point.vector)} does not match corpus "
                    f"{corpus!r} dimension {config.dimension} "
                    f"(embedder pin {config.embedder_fingerprint!r})"
                )
            by_size.setdefault(point.slice.size, []).append(
                m.PointStruct(
                    id=str(point.slice.ref.uuid()),
                    vector=list(point.vector),
                    payload=_slice_payload(point.slice),
                )
            )
        for size, structs in sorted(by_size.items()):
            await self._run(
                f"upsert size {size}",
                self._client.upsert(
                    collection_name=self._slice_collection(corpus, size),
                    points=structs,
                    wait=True,
                ),
            )

    async def put_document_meta(self, corpus: str, record: DocumentRecord) -> None:
        m = self._models
        await self.get_corpus_meta(corpus)
        await self._run(
            "docs write",
            self._client.upsert(
                collection_name=self._docs_collection,
                points=[
                    m.PointStruct(
                        id=str(_registry_uuid("document", corpus, record.document_id)),
                        vector=[1.0],
                        payload=_record_payload(corpus, record),
                    )
                ],
                wait=True,
            ),
        )

    async def get_document_meta(self, corpus: str, document_id: str) -> DocumentRecord:
        await self.get_corpus_meta(corpus)
        records = await self._run(
            "docs read",
            self._client.retrieve(
                collection_name=self._docs_collection,
                ids=[str(_registry_uuid("document", corpus, document_id))],
                with_payload=True,
            ),
        )
        if not records:
            raise DocumentNotFoundError(
                f"document {document_id!r} is not registered in corpus {corpus!r}"
            )
        return _record_from_payload(records[0].payload or {})

    async def list_documents(
        self, corpus: str, *, limit: int = 100, cursor: str | None = None
    ) -> DocumentPage:
        await self.get_corpus_meta(corpus)
        records, next_offset = await self._run(
            "docs scroll",
            self._client.scroll(
                collection_name=self._docs_collection,
                scroll_filter=self._corpus_filter(corpus),
                limit=limit,
                offset=cursor,
                with_payload=True,
                with_vectors=False,
            ),
        )
        return DocumentPage(
            records=tuple(_record_from_payload(record.payload or {}) for record in records),
            next_cursor=None if next_offset is None else str(next_offset),
        )

    async def list_point_ids(self, corpus: str, document_id: str) -> dict[int, set[_uuid.UUID]]:
        config = await self.get_corpus_meta(corpus)
        result: dict[int, set[_uuid.UUID]] = {}
        for size in config.grid.sizes:
            ids: set[_uuid.UUID] = set()
            offset: Any = None
            while True:
                records, offset = await self._run(
                    f"scroll size {size}",
                    self._client.scroll(
                        collection_name=self._slice_collection(corpus, size),
                        scroll_filter=self._document_filter(document_id),
                        limit=self._SCROLL_PAGE,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    ),
                )
                ids.update(_uuid.UUID(str(record.id)) for record in records)
                if offset is None:
                    break
            result[size] = ids
        return result

    async def delete_points(
        self, corpus: str, ids_by_size: Mapping[int, AbstractSet[_uuid.UUID]]
    ) -> None:
        m = self._models
        await self.get_corpus_meta(corpus)
        for size, ids in sorted(ids_by_size.items()):
            if not ids:
                continue
            await self._run(
                f"delete size {size}",
                self._client.delete(
                    collection_name=self._slice_collection(corpus, size),
                    points_selector=m.PointIdsList(points=sorted(str(i) for i in ids)),
                    wait=True,
                ),
            )

    async def delete_document(self, corpus: str, document_id: str) -> None:
        m = self._models
        config = await self.get_corpus_meta(corpus)
        await self.get_document_meta(corpus, document_id)  # raises if unregistered
        for size in config.grid.sizes:
            await self._run(
                f"delete size {size}",
                self._client.delete(
                    collection_name=self._slice_collection(corpus, size),
                    points_selector=m.FilterSelector(filter=self._document_filter(document_id)),
                    wait=True,
                ),
            )
        await self._run(
            "docs delete",
            self._client.delete(
                collection_name=self._docs_collection,
                points_selector=m.PointIdsList(
                    points=[str(_registry_uuid("document", corpus, document_id))]
                ),
                wait=True,
            ),
        )

    # ── reads ──────────────────────────────────────────────────────────────

    async def search_size(
        self,
        corpus: str,
        size: int,
        vector: Sequence[float],
        k: int,
        *,
        timeout_s: float | None = None,
    ) -> list[RetrievedHit]:
        try:
            response = await self._client.query_points(
                collection_name=self._slice_collection(corpus, size),
                query=list(vector),
                limit=k,
                with_payload=True,
                with_vectors=False,
                timeout=None if timeout_s is None else max(1, math.ceil(timeout_s)),
            )
        except self._backend_errors as e:
            await self.get_corpus_meta(corpus)  # translate to CorpusNotFoundError if unknown
            raise StoreError(f"qdrant search at size {size} failed: {e}") from e
        # Qdrant orders by score; re-sort with the id tiebreak for cross-adapter determinism.
        scored = sorted(
            ((point.score, _uuid.UUID(str(point.id)), point) for point in response.points),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            RetrievedHit(
                slice=_slice_from_payload(point.payload or {}),
                corpus=corpus,
                score=score,
                rank_in_size=rank,
                provenance=HitProvenance.RETRIEVED,
            )
            for rank, (score, _, point) in enumerate(scored)
        ]

    async def fetch_by_ids(
        self, corpus: str, size: int, ids: Sequence[_uuid.UUID]
    ) -> list[SlicePoint]:
        if not ids:
            return []
        try:
            records = await self._client.retrieve(
                collection_name=self._slice_collection(corpus, size),
                ids=[str(i) for i in ids],
                with_payload=True,
                with_vectors=False,
            )
        except self._backend_errors as e:
            await self.get_corpus_meta(corpus)  # translate to CorpusNotFoundError if unknown
            raise StoreError(f"qdrant fetch at size {size} failed: {e}") from e
        return [
            SlicePoint(slice=_slice_from_payload(record.payload or {}), vector=None)
            for record in records
        ]

    async def corpus_stats(self, corpus: str) -> CorpusStats:
        config = await self.get_corpus_meta(corpus)
        points_by_size: list[tuple[int, int]] = []
        for size in config.grid.sizes:
            name = self._slice_collection(corpus, size)
            if await self._run("collection check", self._client.collection_exists(name)):
                result = await self._run(
                    f"count size {size}", self._client.count(collection_name=name, exact=True)
                )
                points_by_size.append((size, result.count))
            else:
                points_by_size.append((size, 0))
        doc_count = await self._run(
            "docs count",
            self._client.count(
                collection_name=self._docs_collection,
                count_filter=self._corpus_filter(corpus),
                exact=True,
            ),
        )
        return CorpusStats(
            corpus=corpus,
            document_count=doc_count.count,
            points_by_size=tuple(points_by_size),
        )

    # ── filters ────────────────────────────────────────────────────────────

    def _corpus_filter(self, corpus: str) -> Any:
        m = self._models
        return m.Filter(must=[m.FieldCondition(key="corpus", match=m.MatchValue(value=corpus))])

    def _document_filter(self, document_id: str) -> Any:
        m = self._models
        return m.Filter(
            must=[m.FieldCondition(key="document_id", match=m.MatchValue(value=document_id))]
        )
