"""Document lifecycle service: add/replace (full re-slice), delete, list.

Add and replace are the same operation: the document is sliced at every grid
size, every slice is embedded, and the points are upserted idempotently
(deterministic ids make a repeat ingest an in-place upsert). After upserting the
new generation, the orphan set from a shrunk replacement — the exact per-size
difference between the old and new ids — is tombstoned. The order is upsert then
tombstone, so a query racing a replace is protected by the marker staleness guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phoropter.errors import PhoropterError
from phoropter.markers import marker_for_text
from phoropter.slicer import multi_view_slice
from phoropter.stores import DocumentRecord, SlicePoint
from phoropter.tokens import get_counter

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from phoropter.service.core import ServiceCore
    from phoropter.stores import DocumentPage

__all__ = ["DocumentService", "DocumentTooLargeError"]


class DocumentTooLargeError(PhoropterError):
    """A document exceeds the configured code-point limit.

    A trust-boundary limit (``limits.max_document_codepoints``), so it lives in
    the service layer rather than the core error hierarchy. Its ``DOCUMENT_TOO_LARGE``
    code maps to HTTP 413.
    """

    code = "DOCUMENT_TOO_LARGE"


class DocumentService:
    """Add/replace, delete, and list documents in a corpus."""

    def __init__(self, core: ServiceCore) -> None:
        self._core = core

    async def put(
        self,
        corpus: str,
        document_id: str,
        text: str,
        *,
        metadata: Sequence[tuple[str, str]] = (),
    ) -> DocumentRecord:
        """Add or replace one document; returns its registry record.

        Raises :class:`~phoropter.errors.DocumentTooLargeError` (HTTP 413) when
        the text exceeds ``limits.max_document_codepoints``, and
        :class:`~phoropter.errors.SlicingError` (HTTP 422) when the text cannot
        be sliced. Idempotent under deterministic ids.
        """
        store = self._core.store
        config = await store.get_corpus_meta(corpus)  # CORPUS_NOT_FOUND if absent
        limit = self._core.settings.limits.max_document_codepoints
        if len(text) > limit:
            raise DocumentTooLargeError(
                f"document {document_id!r} has {len(text)} code points, "
                f"exceeding the limit of {limit}"
            )

        counter = get_counter(config.token_counter_id)
        sliced = multi_view_slice(document_id, text, config.grid, token_counter=counter)

        # Old generation ids, captured before the upsert, drive tombstoning.
        old_ids = await store.list_point_ids(corpus, document_id)

        if sliced.slices:
            vectors = await self._core.embedder.embed([s.text for s in sliced.slices])
            points = [
                SlicePoint(slice=s, vector=tuple(vec))
                for s, vec in zip(sliced.slices, vectors, strict=True)
            ]
            await store.upsert_slices(corpus, points, grid_fingerprint=config.grid.fingerprint())

        new_ids_by_size: dict[int, set[uuid.UUID]] = {size: set() for size in config.grid.sizes}
        for s in sliced.slices:
            new_ids_by_size[s.size].add(s.ref.uuid())
        orphans = {
            size: old_ids.get(size, set()) - new_ids_by_size.get(size, set())
            for size in config.grid.sizes
        }
        orphans = {size: ids for size, ids in orphans.items() if ids}
        if orphans:
            await store.delete_points(corpus, orphans)

        record = DocumentRecord(
            document_id=document_id,
            codepoint_length=sliced.codepoint_length,
            byte_length=sliced.byte_length,
            slice_count=len(sliced.slices),
            content_marker=marker_for_text(text) if text else "",
            metadata=tuple(metadata),
        )
        await store.put_document_meta(corpus, record)
        return record

    async def get(self, corpus: str, document_id: str) -> DocumentRecord:
        """The registry record, or DOCUMENT_NOT_FOUND."""
        store = self._core.store
        await store.get_corpus_meta(corpus)  # CORPUS_NOT_FOUND before DOCUMENT_NOT_FOUND
        return await store.get_document_meta(corpus, document_id)

    async def delete(self, corpus: str, document_id: str) -> None:
        """Delete a document's points and registry entry. DOCUMENT_NOT_FOUND if absent."""
        store = self._core.store
        await store.get_corpus_meta(corpus)
        await store.delete_document(corpus, document_id)

    async def list_page(
        self, corpus: str, *, limit: int = 100, cursor: str | None = None
    ) -> DocumentPage:
        """One page of document records. CORPUS_NOT_FOUND for an unknown corpus."""
        return await self._core.store.list_documents(corpus, limit=limit, cursor=cursor)
