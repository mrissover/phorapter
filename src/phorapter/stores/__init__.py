"""Vector store SPI: :class:`VectorStoreAdapter`, its value types, and the marker payload codec.

The SPI speaks **logical** coordinates — a corpus name and a grid size. How an
adapter realizes them in storage (one collection per (corpus, size), one big
table with discriminator columns, ...) is the adapter's business; nothing above
this interface may depend on the physical layout.

Adapters are registered under the ``phorapter.stores`` entry-point group and
claim compliance by passing the conformance suite
(``tests/spi/test_store_contract.py``). See ``docs/adapters.md``.

This module is part of the core: it imports only the standard library and other
core phorapter modules (enforced by import-linter).
"""

from __future__ import annotations

import base64
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from phorapter.errors import StoreError
from phorapter.grid import GridSpec
from phorapter.model import RetrievedHit, Slice, SliceRef

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from collections.abc import Set as AbstractSet

__all__ = [
    "PAYLOAD_SCHEMA_VERSION",
    "CorpusConfig",
    "CorpusStats",
    "DocumentPage",
    "DocumentRecord",
    "SlicePoint",
    "VectorStoreAdapter",
    "available_store_names",
    "load_store_class",
    "pack_markers",
    "unpack_markers",
]

PAYLOAD_SCHEMA_VERSION = 1
"""Version of the slice payload contract (see docs/adapters.md). Bumped only on breaking change."""

_MARKER_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST_BYTES = 32


def pack_markers(markers: tuple[str, ...]) -> str:
    """Pack a descendant-marker tuple into one base64 string for a payload field.

    Each marker is a full SHA-256 digest as 64 **lowercase** hex characters; the
    packed form is the base64 of the concatenated 32-byte digests. The encoding
    is lossless and order-preserving — descendant order (child size ascending,
    then offset ascending) is part of the payload contract. Digests are packed
    in full: truncation would trade the exactness guarantee for a collision
    caveat, and the space saved is immaterial.
    """
    raw = bytearray()
    for marker in markers:
        if not _MARKER_RE.fullmatch(marker):
            raise ValueError(f"not a marker (64 lowercase hex characters): {marker!r}")
        raw += bytes.fromhex(marker)
    return base64.b64encode(bytes(raw)).decode("ascii")


def unpack_markers(packed: str) -> tuple[str, ...]:
    """Invert :func:`pack_markers`, recovering the marker tuple in original order."""
    try:
        raw = base64.b64decode(packed.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as e:
        raise ValueError(f"not a base64 marker pack: {packed!r}") from e
    if len(raw) % _DIGEST_BYTES != 0:
        raise ValueError(
            f"marker pack length {len(raw)} is not a multiple of {_DIGEST_BYTES} bytes"
        )
    return tuple(raw[i : i + _DIGEST_BYTES].hex() for i in range(0, len(raw), _DIGEST_BYTES))


@dataclass(frozen=True, slots=True)
class SlicePoint:
    """A slice paired with its embedding vector — the unit of upsert.

    ``vector`` must be present for upserts. Points returned by
    :meth:`VectorStoreAdapter.fetch_by_ids` may carry ``vector=None``: fetches
    exist to materialize slice *content* (trade-up ancestors), and adapters are
    free to skip vector retrieval there.
    """

    slice: Slice
    vector: tuple[float, ...] | None

    @property
    def ref(self) -> SliceRef:
        return self.slice.ref


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """Everything that governs slicing, indexing, and querying one corpus.

    Written into the store's meta storage exactly once, at corpus creation, and
    immutable thereafter — servers stay stateless by reading it back. It is a
    pure value type: no timestamps, no free-form metadata, nothing
    environment-dependent, so two processes that agree on a config agree on
    every derived artifact.

    ``embedder_fingerprint`` is the ``"provider:model"`` pin — a corpus never
    silently accepts vectors from a different embedder. ``dimension`` is the
    embedding dimension probed at creation. ``token_counter_id`` pins the
    tokenizer whose per-slice counts are stored at ingest.
    """

    name: str
    grid: GridSpec
    embedder_fingerprint: str
    dimension: int
    token_counter_id: str


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """The registry entry for one ingested document.

    Document-level facts live here, once — never duplicated onto every slice
    payload. ``content_marker`` is the marker of the full document text, which
    makes replace-with-identical-content detectable without fetching anything.
    ``metadata`` holds caller-supplied key/value pairs, kept as an ordered
    tuple of pairs so records remain hashable value types.
    """

    document_id: str
    codepoint_length: int
    byte_length: int
    slice_count: int
    content_marker: str
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of a cursor-paginated document listing.

    ``next_cursor`` is an opaque adapter-defined token; ``None`` means the
    listing is complete. Against an unchanging corpus, walking pages until
    ``None`` yields every document exactly once.
    """

    records: tuple[DocumentRecord, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Point and document counts for one corpus.

    ``points_by_size`` holds ``(size, count)`` pairs in ascending size order,
    one per grid size — sizes with zero points included, so the shape always
    mirrors the grid.
    """

    corpus: str
    document_count: int
    points_by_size: tuple[tuple[int, int], ...]

    @property
    def total_points(self) -> int:
        return sum(count for _, count in self.points_by_size)


class VectorStoreAdapter(ABC):
    """The vector store SPI.

    All methods are async; adapters own their connection lifecycle. Contract
    highlights (normative — the conformance suite enforces them):

    - **Deterministic search ordering.** :meth:`search_size` returns hits
      sorted by score descending, ties broken by point id ascending, with
      0-based ``rank_in_size`` assigned in that order.
    - **Payload-complete hits.** Every hit and every fetched point carries a
      fully reconstructed :class:`~phorapter.model.Slice` — all coordinates,
      text, markers, and token count. Callers never re-derive slice content
      from a second source.
    - **Config is frozen.** :meth:`create_corpus` persists the
      :class:`CorpusConfig` once; :meth:`upsert_slices` refuses data whose grid
      fingerprint or vector dimension disagrees with it
      (:class:`~phorapter.errors.CorpusMismatchError`).
    - **Errors.** Unknown corpus → :class:`~phorapter.errors.CorpusNotFoundError`;
      duplicate creation → :class:`~phorapter.errors.CorpusExistsError`; unknown
      document → :class:`~phorapter.errors.DocumentNotFoundError`; backend
      unreachable or failing → :class:`~phorapter.errors.StoreError`.
    """

    @abstractmethod
    async def ping(self) -> bool:
        """True if the backend is reachable; connectivity failures return False, not raise."""

    @abstractmethod
    async def bootstrap(self) -> None:
        """Idempotently create shared infrastructure (meta/document storage), not corpora."""

    @abstractmethod
    async def create_corpus(self, config: CorpusConfig) -> None:
        """Create a corpus and persist its config, exactly once.

        Raises :class:`~phorapter.errors.CorpusExistsError` if the name is taken.
        """

    @abstractmethod
    async def drop_corpus(self, corpus: str) -> None:
        """Remove a corpus: its points, its document registry entries, and its config."""

    @abstractmethod
    async def get_corpus_meta(self, corpus: str) -> CorpusConfig:
        """The config persisted at creation."""

    @abstractmethod
    async def list_corpora(self) -> tuple[str, ...]:
        """Names of all corpora, sorted."""

    @abstractmethod
    async def verify_corpus(self, corpus: str) -> list[str]:
        """Degradation reasons for a corpus; the empty list means healthy.

        A degraded corpus (a missing size collection, dimension drift
        introduced out-of-band, ...) still *exists* — callers decide whether to
        serve partial results or refuse. An unknown corpus raises
        :class:`~phorapter.errors.CorpusNotFoundError` instead.
        """

    @abstractmethod
    async def upsert_slices(
        self, corpus: str, points: Sequence[SlicePoint], *, grid_fingerprint: str
    ) -> None:
        """Write points idempotently (deterministic ids make re-ingest an in-place upsert).

        ``grid_fingerprint`` must equal the corpus grid's fingerprint — data
        sliced on a different grid is refused
        (:class:`~phorapter.errors.CorpusMismatchError`), as are vectors of the
        wrong dimension and points whose size is not a grid size.
        """

    @abstractmethod
    async def put_document_meta(self, corpus: str, record: DocumentRecord) -> None:
        """Create or replace the registry entry for one document."""

    @abstractmethod
    async def get_document_meta(self, corpus: str, document_id: str) -> DocumentRecord:
        """The registry entry, or :class:`~phorapter.errors.DocumentNotFoundError`."""

    @abstractmethod
    async def list_documents(
        self, corpus: str, *, limit: int = 100, cursor: str | None = None
    ) -> DocumentPage:
        """One page of document records, in a stable adapter-defined order."""

    @abstractmethod
    async def list_point_ids(self, corpus: str, document_id: str) -> dict[int, set[uuid.UUID]]:
        """Every stored point id of one document, grouped by size.

        This is the replace/tombstone workhorse: the orphan set after a shrink
        is the exact per-size difference between this and the new generation's
        ids.
        """

    @abstractmethod
    async def delete_points(
        self, corpus: str, ids_by_size: Mapping[int, AbstractSet[uuid.UUID]]
    ) -> None:
        """Delete specific points. Absent ids are ignored (deletes are idempotent)."""

    @abstractmethod
    async def delete_document(self, corpus: str, document_id: str) -> None:
        """Delete a document's points and its registry entry.

        Raises :class:`~phorapter.errors.DocumentNotFoundError` if the document
        is not registered.
        """

    @abstractmethod
    async def search_size(
        self,
        corpus: str,
        size: int,
        vector: Sequence[float],
        k: int,
        *,
        timeout_s: float | None = None,
    ) -> list[RetrievedHit]:
        """Top-``k`` nearest points at one grid size, payload-complete, deterministically ordered."""

    @abstractmethod
    async def fetch_by_ids(
        self, corpus: str, size: int, ids: Sequence[uuid.UUID]
    ) -> list[SlicePoint]:
        """Materialize points by id (trade-up ancestor prefetch); unknown ids are omitted."""

    @abstractmethod
    async def corpus_stats(self, corpus: str) -> CorpusStats:
        """Document and per-size point counts."""


def available_store_names() -> tuple[str, ...]:
    """Adapter names registered under the ``phorapter.stores`` entry-point group, sorted."""
    from importlib.metadata import entry_points

    return tuple(sorted({ep.name for ep in entry_points(group="phorapter.stores")}))


def load_store_class(name: str) -> type[VectorStoreAdapter]:
    """Resolve a store adapter class from the ``phorapter.stores`` entry-point group."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="phorapter.stores"):
        if ep.name == name:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, VectorStoreAdapter)):
                raise StoreError(
                    f"entry point {name!r} in group 'phorapter.stores' is not a "
                    f"VectorStoreAdapter subclass: {cls!r}"
                )
            return cls
    known = ", ".join(available_store_names()) or "(none)"
    raise StoreError(f"no store adapter registered under {name!r}; available: {known}")
