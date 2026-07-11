"""Deterministic slice identity.

Every slice has a deterministic UUIDv5 derived from its coordinates
``(document_id, size, codepoint_offset)`` under the project namespace. Determinism
is load-bearing for the document lifecycle: re-ingesting a document re-derives the
same IDs, so an updated document upserts in place, and the orphaned IDs of a
shrunken document are an exact set difference — no scanning.

The name string joins fields with ``|`` after escaping (``\\`` → ``\\\\``,
``|`` → ``\\|``) on free-text fields, which makes the mapping injective for
arbitrary document IDs. The trailing numeric fields cannot contain the separator.

``corpus`` is optional: the default store layout isolates corpora in separate
collections, so IDs need only be unique within a corpus. Adapters that co-mingle
corpora in one collection MUST pass ``corpus`` to keep IDs globally unique.
"""

from __future__ import annotations

import uuid

__all__ = ["PHOROPTER_NAMESPACE", "slice_name", "slice_uuid"]

PHOROPTER_NAMESPACE = uuid.UUID("fabb402a-1e0f-55a8-83cd-2d0a124a3fca")
"""The project's UUIDv5 namespace.

Derived once as ``uuid5(NAMESPACE_DNS, "phoropter.impluvium.software")`` and
hard-coded; a pinned test guards both the value and its derivation.
"""


def _escape(field: str) -> str:
    return field.replace("\\", "\\\\").replace("|", "\\|")


def slice_name(
    document_id: str,
    size: int,
    codepoint_offset: int,
    *,
    corpus: str | None = None,
) -> str:
    """The canonical, injective name string for a slice's coordinates."""
    parts = [] if corpus is None else [_escape(corpus)]
    parts += [_escape(document_id), str(size), str(codepoint_offset)]
    return "|".join(parts)


def slice_uuid(
    document_id: str,
    size: int,
    codepoint_offset: int,
    *,
    corpus: str | None = None,
) -> uuid.UUID:
    """Deterministic UUIDv5 for a slice's coordinates under the project namespace."""
    return uuid.uuid5(
        PHOROPTER_NAMESPACE, slice_name(document_id, size, codepoint_offset, corpus=corpus)
    )
