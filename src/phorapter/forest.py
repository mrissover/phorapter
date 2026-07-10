"""The containment forest over a set of retrieved hits.

Containment here is decided **positionally**, from grid coordinates — never by
comparing text. Because every document's grid slices form a laminar family
(nested or disjoint), each hit has at most one enclosing slice per larger size,
and that slice's offset is closed-form: ``parent_offset = (offset // S) * S``. So
building the forest is a walk up the grid, not a search.

Markers are the **integrity guard**, not the decision. A positional parent that
is genuinely present in the retrieved set must also list the child's marker among
its descendants; if it does not, the two came from different generations of a
document (a read racing a replace), and the pair is recorded as a
:class:`ContainmentAnomaly` rather than linked. Two byte-identical slices at
different offsets never link, because the false parent is never even looked up —
identity is coordinates, not content.

The result is a forest: each node has at most one *minimal* parent (its smallest
enclosing retrieved slice), and chains run strictly upward in size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from phorapter.errors import PhorapterError
from phorapter.model import RetrievedHit, SliceRef

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phorapter.grid import GridSpec

__all__ = ["ContainmentAnomaly", "ContainmentForest", "Edge", "contains"]


class ForestError(PhorapterError):
    """A hit set cannot form a forest (e.g. an off-grid slice)."""

    code = "VALIDATION_ERROR"


@dataclass(frozen=True, slots=True)
class Edge:
    """A containment edge: ``parent`` encloses ``child`` (``parent.size > child.size``)."""

    parent: RetrievedHit
    child: RetrievedHit


@dataclass(frozen=True, slots=True)
class ContainmentAnomaly:
    """A positional parent whose descendant markers do not vouch for the child.

    Recorded, never fatal: it signals a cross-generation read (a document replace
    in flight), and it stops that parent from being used for substitution.
    """

    parent_ref: SliceRef
    child_ref: SliceRef
    reason: str


def _positionally_nests(parent: RetrievedHit, child: RetrievedHit) -> bool:
    return (
        parent.slice.document_id == child.slice.document_id
        and parent.slice.size > child.slice.size
        and parent.slice.codepoint_offset <= child.slice.codepoint_offset
        and child.slice.codepoint_end <= parent.slice.codepoint_end
    )


def contains(parent: RetrievedHit, child: RetrievedHit) -> bool:
    """True if ``parent`` encloses ``child``: positional nesting plus marker vouching.

    This is the standalone predicate; the forest applies the same test while
    walking the grid.
    """
    return _positionally_nests(parent, child) and (
        child.slice.own_marker in parent.slice.descendant_markers
    )


class ContainmentForest:
    """A forest of minimal-parent containment edges over retrieved hits.

    Construct with :meth:`build`. Complexity is ``O(H * |sizes|)``: each hit walks
    up the grid once, and each parent's descendant markers are hashed into a set
    once.
    """

    def __init__(
        self,
        hits: Sequence[RetrievedHit],
        grid: GridSpec,
        *,
        minimal_parent: dict[int, int | None],
        children: dict[int, list[int]],
        anomalies: tuple[ContainmentAnomaly, ...],
        closure: bool,
    ) -> None:
        self._hits = tuple(hits)
        self._grid = grid
        self._minimal_parent = minimal_parent
        self._children = children
        self._anomalies = anomalies
        self._closure = closure
        self._index_by_ref = {h.ref: i for i, h in enumerate(self._hits)}

    @classmethod
    def build(
        cls,
        hits: Sequence[RetrievedHit],
        grid: GridSpec,
        *,
        closure: bool = False,
    ) -> ContainmentForest:
        """Build the forest over ``hits``.

        Duplicate refs are collapsed to one hit. ``closure`` only affects
        :meth:`edges` (whether transitive ancestor edges are emitted); the
        minimal-parent structure is always computed. An off-grid hit (offset not
        a multiple of its size) raises :class:`ForestError` — the grid guarantee
        it violates is what makes the family laminar.
        """
        deduped: list[RetrievedHit] = []
        seen: set[SliceRef] = set()
        for hit in hits:
            if hit.slice.codepoint_offset % hit.slice.size != 0:
                raise ForestError(
                    f"off-grid hit {hit.ref}: offset {hit.slice.codepoint_offset} "
                    f"is not a multiple of size {hit.slice.size}"
                )
            if hit.ref not in seen:
                seen.add(hit.ref)
                deduped.append(hit)

        # Per-document positional index: (size, offset) -> hit index.
        by_coords: dict[tuple[str, int, int], int] = {
            (h.slice.document_id, h.slice.size, h.slice.codepoint_offset): i
            for i, h in enumerate(deduped)
        }
        descendant_sets = [frozenset(h.slice.descendant_markers) for h in deduped]

        minimal_parent: dict[int, int | None] = {}
        children: dict[int, list[int]] = {i: [] for i in range(len(deduped))}
        anomalies: list[ContainmentAnomaly] = []

        for i, child in enumerate(deduped):
            parent_idx: int | None = None
            for parent_size in grid.levels_above(child.slice.size):
                offset = grid.parent_offset(child.slice.codepoint_offset, parent_size)
                key = (child.slice.document_id, parent_size, offset)
                candidate = by_coords.get(key)
                if candidate is None:
                    continue
                if child.slice.own_marker in descendant_sets[candidate]:
                    parent_idx = candidate
                    break
                anomalies.append(
                    ContainmentAnomaly(
                        parent_ref=deduped[candidate].ref,
                        child_ref=child.ref,
                        reason="child marker absent from positional parent's descendants",
                    )
                )
            minimal_parent[i] = parent_idx
            if parent_idx is not None:
                children[parent_idx].append(i)

        return cls(
            deduped,
            grid,
            minimal_parent=minimal_parent,
            children=children,
            anomalies=tuple(anomalies),
            closure=closure,
        )

    @property
    def hits(self) -> tuple[RetrievedHit, ...]:
        return self._hits

    @property
    def anomalies(self) -> tuple[ContainmentAnomaly, ...]:
        return self._anomalies

    def _chain_up(self, index: int) -> list[int]:
        chain: list[int] = []
        current = self._minimal_parent[index]
        while current is not None:
            chain.append(current)
            current = self._minimal_parent[current]
        return chain

    def minimal_parent(self, ref: SliceRef) -> SliceRef | None:
        """The ref of the smallest enclosing retrieved slice, or ``None``."""
        index = self._index_by_ref.get(ref)
        if index is None:
            return None
        parent = self._minimal_parent[index]
        return None if parent is None else self._hits[parent].ref

    def retrieved_ancestors(self, ref: SliceRef) -> tuple[SliceRef, ...]:
        """The minimal-parent chain above ``ref``, size ascending (nearest first)."""
        index = self._index_by_ref.get(ref)
        if index is None:
            return ()
        return tuple(self._hits[i].ref for i in self._chain_up(index))

    def leaves(self) -> tuple[RetrievedHit, ...]:
        """Hits with no retrieved slice contained in them."""
        return tuple(h for i, h in enumerate(self._hits) if not self._children[i])

    def roots(self) -> tuple[RetrievedHit, ...]:
        """Hits with no enclosing retrieved slice."""
        return tuple(h for i, h in enumerate(self._hits) if self._minimal_parent[i] is None)

    def edges(self) -> tuple[Edge, ...]:
        """Containment edges.

        Minimal-parent edges by default; with ``closure=True`` at build time,
        every ancestor-descendant pair (the transitive closure).
        """
        out: list[Edge] = []
        for i, child in enumerate(self._hits):
            ancestors = self._chain_up(i)
            if not self._closure:
                ancestors = ancestors[:1]
            out.extend(Edge(parent=self._hits[a], child=child) for a in ancestors)
        return tuple(out)

    def participation_rate(self) -> float:
        """Fraction of hits in at least one minimal edge (as parent or child)."""
        if not self._hits:
            return 0.0
        participating = sum(
            1
            for i in range(len(self._hits))
            if self._minimal_parent[i] is not None or self._children[i]
        )
        return participating / len(self._hits)

    def max_depth(self) -> int:
        """Nodes in the longest minimal-parent chain (a standalone hit counts as 1)."""
        if not self._hits:
            return 0
        return max(1 + len(self._chain_up(i)) for i in range(len(self._hits)))
