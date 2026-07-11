"""In-memory vector store: the reference SPI implementation and library-mode store.

Zero third-party dependencies (this module is core), brute-force cosine
search, and exact SPI semantics — it is the behavioral yardstick the
conformance suite holds every other adapter to. Search is exhaustive rather
than approximate, so its ordering (score descending, point id ascending) is
exact by construction.

Beyond the async SPI, :class:`InMemoryStore` exposes a synchronous
:meth:`~InMemoryStore.get_slices` — the engine's ``SliceSource`` protocol — so
in-process library users can run budget-fitting trade-up directly against it
without an event loop.

Not safe for concurrent mutation from multiple threads; it targets tests and
single-process library use.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from phoropter.errors import (
    CorpusExistsError,
    CorpusMismatchError,
    CorpusNotFoundError,
    DocumentNotFoundError,
)
from phoropter.model import HitProvenance, RetrievedHit, Slice, SliceRef
from phoropter.stores import (
    CorpusConfig,
    CorpusStats,
    DocumentPage,
    DocumentRecord,
    SlicePoint,
    VectorStoreAdapter,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence
    from collections.abc import Set as AbstractSet

__all__ = ["InMemoryStore"]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryStore(VectorStoreAdapter):
    """Full :class:`~phoropter.stores.VectorStoreAdapter` over plain dicts."""

    def __init__(self) -> None:
        self._corpora: dict[str, CorpusConfig] = {}
        # corpus -> size -> point id -> point (vector always present here).
        self._points: dict[str, dict[int, dict[uuid.UUID, SlicePoint]]] = {}
        self._documents: dict[str, dict[str, DocumentRecord]] = {}

    def _config(self, corpus: str) -> CorpusConfig:
        try:
            return self._corpora[corpus]
        except KeyError:
            raise CorpusNotFoundError(f"corpus {corpus!r} does not exist") from None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        return True

    async def bootstrap(self) -> None:
        return None

    async def create_corpus(self, config: CorpusConfig) -> None:
        if config.name in self._corpora:
            raise CorpusExistsError(f"corpus {config.name!r} already exists")
        self._corpora[config.name] = config
        self._points[config.name] = {size: {} for size in config.grid.sizes}
        self._documents[config.name] = {}

    async def drop_corpus(self, corpus: str) -> None:
        self._config(corpus)
        del self._corpora[corpus]
        del self._points[corpus]
        del self._documents[corpus]

    async def get_corpus_meta(self, corpus: str) -> CorpusConfig:
        return self._config(corpus)

    async def list_corpora(self) -> tuple[str, ...]:
        return tuple(sorted(self._corpora))

    async def verify_corpus(self, corpus: str) -> list[str]:
        config = self._config(corpus)
        reasons = []
        for size in config.grid.sizes:
            if size not in self._points[corpus]:  # pragma: no cover - unreachable in-process
                reasons.append(f"size {size} storage missing")
        return reasons

    # ── writes ─────────────────────────────────────────────────────────────

    async def upsert_slices(
        self, corpus: str, points: Sequence[SlicePoint], *, grid_fingerprint: str
    ) -> None:
        config = self._config(corpus)
        if grid_fingerprint != config.grid.fingerprint():
            raise CorpusMismatchError(
                f"grid fingerprint {grid_fingerprint!r} does not match corpus "
                f"{corpus!r} (created with sizes {config.grid.sizes})"
            )
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
        for point in points:
            self._points[corpus][point.slice.size][point.slice.ref.uuid()] = point

    async def put_document_meta(self, corpus: str, record: DocumentRecord) -> None:
        self._config(corpus)
        self._documents[corpus][record.document_id] = record

    async def get_document_meta(self, corpus: str, document_id: str) -> DocumentRecord:
        self._config(corpus)
        try:
            return self._documents[corpus][document_id]
        except KeyError:
            raise DocumentNotFoundError(
                f"document {document_id!r} is not registered in corpus {corpus!r}"
            ) from None

    async def list_documents(
        self, corpus: str, *, limit: int = 100, cursor: str | None = None
    ) -> DocumentPage:
        self._config(corpus)
        ordered = sorted(self._documents[corpus])
        if cursor is not None:
            ordered = [doc_id for doc_id in ordered if doc_id > cursor]
        page = ordered[:limit]
        next_cursor = page[-1] if len(ordered) > limit and page else None
        return DocumentPage(
            records=tuple(self._documents[corpus][doc_id] for doc_id in page),
            next_cursor=next_cursor,
        )

    async def list_point_ids(self, corpus: str, document_id: str) -> dict[int, set[uuid.UUID]]:
        self._config(corpus)
        return {
            size: {
                point_id
                for point_id, point in by_id.items()
                if point.slice.document_id == document_id
            }
            for size, by_id in self._points[corpus].items()
        }

    async def delete_points(
        self, corpus: str, ids_by_size: Mapping[int, AbstractSet[uuid.UUID]]
    ) -> None:
        self._config(corpus)
        for size, ids in ids_by_size.items():
            by_id = self._points[corpus].get(size, {})
            for point_id in ids:
                by_id.pop(point_id, None)

    async def delete_document(self, corpus: str, document_id: str) -> None:
        await self.get_document_meta(corpus, document_id)  # raises if unregistered
        for by_id in self._points[corpus].values():
            orphans = [
                point_id
                for point_id, point in by_id.items()
                if point.slice.document_id == document_id
            ]
            for point_id in orphans:
                del by_id[point_id]
        del self._documents[corpus][document_id]

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
        config = self._config(corpus)
        if size not in config.grid.sizes:
            raise CorpusMismatchError(f"{size} is not a grid size of corpus {corpus!r}")
        scored = sorted(
            (
                (_cosine(vector, point.vector or ()), point_id, point)
                for point_id, point in self._points[corpus][size].items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            RetrievedHit(
                slice=point.slice,
                corpus=corpus,
                score=score,
                rank_in_size=rank,
                provenance=HitProvenance.RETRIEVED,
            )
            for rank, (score, _, point) in enumerate(scored[:k])
        ]

    async def fetch_by_ids(
        self, corpus: str, size: int, ids: Sequence[uuid.UUID]
    ) -> list[SlicePoint]:
        config = self._config(corpus)
        if size not in config.grid.sizes:
            raise CorpusMismatchError(f"{size} is not a grid size of corpus {corpus!r}")
        by_id = self._points[corpus][size]
        return [by_id[point_id] for point_id in ids if point_id in by_id]

    async def corpus_stats(self, corpus: str) -> CorpusStats:
        config = self._config(corpus)
        return CorpusStats(
            corpus=corpus,
            document_count=len(self._documents[corpus]),
            points_by_size=tuple(
                (size, len(self._points[corpus][size])) for size in config.grid.sizes
            ),
        )

    # ── engine bridge (SliceSource protocol) ───────────────────────────────

    def get_slices(self, refs: Iterable[SliceRef], *, corpus: str) -> dict[SliceRef, Slice]:
        """Synchronously resolve slice refs to stored slices; unknown refs are omitted.

        This is the engine's ``SliceSource`` protocol: the selection strategy
        runs pure and synchronous, and in-process users hand it this method so
        trade-up can materialize un-retrieved ancestors without an event loop.
        A ref that resolves to nothing simply stays absent from the result —
        the engine records a fetch miss and moves on.
        """
        config = self._config(corpus)
        found: dict[SliceRef, Slice] = {}
        for ref in refs:
            if ref.size not in config.grid.sizes:
                continue
            point = self._points[corpus][ref.size].get(ref.uuid())
            if point is not None:
                found[ref] = point.slice
        return found
