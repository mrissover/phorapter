"""Core value types.

All types here are frozen, slotted dataclasses: they are created in bulk
(hundreds of thousands of slices for a modest corpus), so they carry no
validation machinery — validation belongs at trust boundaries (the server DTO
layer), not in inner loops.

**Identity vs content.** A slice's *identity* is its coordinates —
:class:`SliceRef` ``(document_id, size, codepoint_offset)``. Its *content* is
attested by markers. Two slices at different coordinates may carry identical
text and therefore identical markers; they are still different slices.
"""

from __future__ import annotations

import enum
import uuid as _uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from phoropter.ids import slice_uuid

if TYPE_CHECKING:
    from phoropter.grid import GridSpec

__all__ = [
    "CandidateHit",
    "HitProvenance",
    "RetrievedHit",
    "Slice",
    "SliceRef",
    "SlicedDocument",
]


@dataclass(frozen=True, slots=True)
class SliceRef:
    """The identity of a slice: its coordinates on the grid."""

    document_id: str
    size: int
    codepoint_offset: int

    def uuid(self, *, corpus: str | None = None) -> _uuid.UUID:
        """Deterministic store point ID for this slice."""
        return slice_uuid(self.document_id, self.size, self.codepoint_offset, corpus=corpus)

    def parent(self, grid: GridSpec) -> SliceRef | None:
        """The ref of the next-larger grid slice containing this one, or ``None`` at the top."""
        above = grid.levels_above(self.size)
        if not above:
            return None
        parent_size = above[0]
        return SliceRef(
            self.document_id,
            parent_size,
            grid.parent_offset(self.codepoint_offset, parent_size),
        )

    def ancestors(self, grid: GridSpec) -> tuple[SliceRef, ...]:
        """Refs of every larger grid slice containing this one, size ascending."""
        return tuple(
            SliceRef(self.document_id, s, grid.parent_offset(self.codepoint_offset, s))
            for s in grid.levels_above(self.size)
        )


@dataclass(frozen=True, slots=True)
class Slice:
    """One code-point span of one document at one grid size, with dual coordinates and markers.

    ``codepoint_length`` may be shorter than ``size`` for the final slice of a
    document. ``descendant_markers`` holds the markers of every strictly-smaller
    grid slice fully contained in this slice's span, in canonical order (child
    size ascending, then offset ascending); duplicates are preserved when
    distinct descendants carry identical text.
    """

    document_id: str
    size: int
    codepoint_offset: int
    codepoint_length: int
    byte_offset: int
    byte_length: int
    document_codepoint_length: int
    text: str
    own_marker: str
    descendant_markers: tuple[str, ...]
    token_count: int | None = None

    @property
    def ref(self) -> SliceRef:
        return SliceRef(self.document_id, self.size, self.codepoint_offset)

    @property
    def codepoint_end(self) -> int:
        """Exclusive end of this slice's span in code points."""
        return self.codepoint_offset + self.codepoint_length

    @property
    def is_short(self) -> bool:
        """True for a clipped final slice (span shorter than the grid size)."""
        return self.codepoint_length < self.size


@dataclass(frozen=True, slots=True)
class SlicedDocument:
    """The full multi-view slicing of one document.

    ``slices`` is ordered canonically: size ascending, then offset ascending.
    """

    document_id: str
    text: str
    codepoint_length: int
    byte_length: int
    grid: GridSpec
    token_counter_id: str | None
    slices: tuple[Slice, ...]

    def slices_of_size(self, size: int) -> tuple[Slice, ...]:
        """All slices at one grid size, offset ascending."""
        return tuple(s for s in self.slices if s.size == size)

    def slice_at(self, size: int, codepoint_offset: int) -> Slice | None:
        """The slice at exact coordinates, or ``None``."""
        for s in self.slices:
            if s.size == size and s.codepoint_offset == codepoint_offset:
                return s
        return None

    def spine(self) -> tuple[Slice, ...]:
        """The left-anchored spine: the offset-0 slice of every size, size ascending.

        Along the spine, each slice's UTF-8 bytes are a prefix of the next
        larger slice's bytes — the *prefix property*, the gating correctness
        invariant of the whole system. The prefix is proper whenever the
        document extends past the smaller size; for documents shorter than a
        grid size, consecutive spine slices cover the same span and are
        byte-identical (with identical markers).
        """
        return tuple(s for s in self.slices if s.codepoint_offset == 0)


class HitProvenance(enum.Enum):
    """How a slice entered the working set at query time."""

    RETRIEVED = "retrieved"
    TRADED_UP = "traded_up"


@dataclass(frozen=True, slots=True)
class RetrievedHit:
    """One slice as returned by (or introduced during) retrieval.

    ``score`` is the raw store score. It is **ordinal within one
    (corpus, size, query) triple only** — scores from different sizes are not
    comparable and are never fused. A slice introduced by trade-up has no score
    (``None``) rather than a fabricated one.
    """

    slice: Slice
    corpus: str
    score: float | None
    rank_in_size: int | None
    provenance: HitProvenance = field(default=HitProvenance.RETRIEVED)

    @property
    def ref(self) -> SliceRef:
        return self.slice.ref


@dataclass(frozen=True, slots=True)
class CandidateHit:
    """A retrieved hit with its position after cross-size rank fusion."""

    hit: RetrievedHit
    fused_rank: int
