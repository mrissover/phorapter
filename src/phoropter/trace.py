"""Substitution trace types: the explainability contract of the selection engine.

A :class:`SubstitutionTrace` records everything the engine did to a query's
candidates: which sizes contributed, what the containment forest looked like,
which retrieved ancestors were folded into evidence, what was packed, every
trade-up with its subsumption accounting, every rejection with its reason, and
the final budget arithmetic.

**Lifecycle invariant (normative).** Every slice the engine touched appears in
exactly one lifecycle path:

- a retrieved ancestor folded into a class's evidence appears in a
  :class:`DedupEntry`'s ``dropped_ids`` (it may later resurface as a trade-up
  *target*, in which case the :class:`TradeUpEntry`'s ``to_provenance`` is
  ``"retrieved"``);
- a class top appears in :class:`InitialPackTrace` — either in ``selected_ids``
  or as a :class:`SkippedUnaffordable` entry — and, if packed, its subsequent
  history is a chain of :class:`TradeUpEntry` links (``from_id`` → ``to_id``),
  possibly ending in another entry's ``subsumed_ids``.

The invariant is machine-checkable: :meth:`SubstitutionTrace.replay_top_ids`
replays the pack and every trade and raises if any id is used out of turn; the
ids it returns must be exactly the ids of the final selection.

All ids are the string form of the slice's deterministic UUID
(:meth:`phoropter.model.SliceRef.uuid` without a corpus qualifier); traces are
scoped to a single corpus, so the unqualified id is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phoropter.model import SliceRef

__all__ = [
    "DedupEntry",
    "FinalTrace",
    "ForestTrace",
    "FusionTrace",
    "InitialPackTrace",
    "RejectionEntry",
    "RejectionReason",
    "SkippedUnaffordable",
    "SubstitutionTrace",
    "TargetProvenance",
    "TradeUpEntry",
]

TargetProvenance = Literal["retrieved", "fetched"]
"""Where a trade-up target's slice came from: the retrieved hits themselves, or
the :class:`~phoropter.selection.SliceSource` (the prefetched ancestor closure)."""

RejectionReason = Literal["over_budget", "stale_parent", "fetch_miss", "saturated"]
"""Why a trade-up attempt was rejected. None of these ever fails the request:

- ``over_budget`` — the netted token delta does not fit the remaining budget;
  the class stays alive and may retry the same level in a later round.
- ``stale_parent`` — the resolved parent's descendant markers do not contain the
  class top's marker (a cross-generation read, e.g. racing a document replace);
  the class keeps its top and skips past that level.
- ``fetch_miss`` — the target is absent from the slice source; the class keeps
  its top and skips past that level.
- ``saturated`` — the class has no further target (top of the grid, degenerate
  short-document levels only, ``max_slice_size`` reached, or the per-class level
  bound hit); recorded once, when the class stops trading.
"""


@dataclass(frozen=True, slots=True)
class FusionTrace:
    """What rank fusion saw: which sizes contributed hits and how many candidates emerged."""

    sizes: tuple[int, ...]
    """Grid sizes that contributed at least one candidate, ascending."""
    candidate_count: int
    """Number of fused candidates handed to the strategy (before ref-dedup)."""


@dataclass(frozen=True, slots=True)
class ForestTrace:
    """Shape of the containment forest built over the deduplicated hits."""

    hit_count: int
    """Distinct hits (by :class:`~phoropter.model.SliceRef`) in the forest."""
    edge_count: int
    """Minimal-parent containment edges."""
    anomaly_count: int
    """Positional-containment pairs whose marker check failed (stale generations)."""
    participation_rate: float
    """Fraction of hits participating in at least one minimal edge."""
    max_depth: int
    """Nodes in the longest minimal-parent chain (a standalone hit counts as 1)."""


@dataclass(frozen=True, slots=True)
class DedupEntry:
    """One dedup class: the kept leaf and the retrieved ancestors folded into its evidence.

    A retrieved ancestor spanning several leaves appears in each of their
    entries; its lifecycle path is still single (retrieved → deduped), and it
    may later resurface as a trade-up target with genuine provenance.
    """

    kept_ref: SliceRef
    kept_id: str
    dropped_refs: tuple[SliceRef, ...]
    """Retrieved ancestors demoted to evidence, size ascending."""
    dropped_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkippedUnaffordable:
    """A class whose top did not fit the remaining budget during the initial pack."""

    ref: SliceRef
    id: str
    tokens: int
    """What the top would have cost (token count plus join overhead)."""
    budget_left: int
    """Remaining budget at the moment the class was skipped."""


@dataclass(frozen=True, slots=True)
class InitialPackTrace:
    """Result of the first-fit initial pack over class tops in priority order."""

    selected_ids: tuple[str, ...]
    """Packed class tops, in pack order (effective rank, then document, then offset)."""
    skipped_unaffordable: tuple[SkippedUnaffordable, ...]
    """Classes that did not fit; the walk continued past each of them."""
    smallest_unaffordable_tokens: int | None
    """Cheapest skipped class's cost, or ``None`` when nothing was skipped."""


@dataclass(frozen=True, slots=True)
class TradeUpEntry:
    """One accepted upward substitution, with its subsumption netting."""

    round: int
    """Trade-up round (1-based)."""
    from_ref: SliceRef
    """The class top that was replaced."""
    from_id: str
    to_ref: SliceRef
    """The parent slice the class traded up to."""
    to_id: str
    to_provenance: TargetProvenance
    subsumed_ids: tuple[str, ...]
    """Tops of other live classes the parent spans (marker-verified); their
    classes were absorbed, and their token costs were netted out of the delta."""
    delta_tokens: int
    """``cost(parent) - cost(top) - sum(cost(subsumed tops))``; may be negative."""
    budget_left: int
    """Remaining budget after applying the delta."""


@dataclass(frozen=True, slots=True)
class RejectionEntry:
    """One rejected trade-up attempt. Rejections never fail the request."""

    round: int
    """Trade-up round (1-based)."""
    target: SliceRef
    """The attempted parent, or the class's current top for ``saturated``."""
    reason: RejectionReason
    delta_tokens: int | None = None
    """The netted delta, where one was computed (``over_budget`` only)."""


@dataclass(frozen=True, slots=True)
class FinalTrace:
    """Final budget arithmetic."""

    tokens_used: int
    budget: int | None
    utilization: float | None
    """``tokens_used / budget``; ``None`` when the budget is absent or zero."""
    budget_exhausted: bool
    """True when the budget constrained the outcome: at least one class was
    skipped as unaffordable, or at least one trade-up was rejected over budget."""


@dataclass(frozen=True, slots=True)
class SubstitutionTrace:
    """The full lifecycle record of one selection run."""

    fusion: FusionTrace
    forest: ForestTrace
    dedup: tuple[DedupEntry, ...]
    initial_pack: InitialPackTrace
    trade_ups: tuple[TradeUpEntry, ...]
    rejections: tuple[RejectionEntry, ...]
    final: FinalTrace
    warnings: tuple[str, ...] = ()

    def replay_top_ids(self) -> frozenset[str]:
        """Replay the initial pack and every trade-up; return the surviving top ids.

        This is the lifecycle invariant made executable: every ``from_id`` and
        every subsumed id must be a live top when its entry consumes it, and no
        trade may introduce a top that is already live. The returned set must
        equal the ids of the final selection's slices. Raises ``ValueError`` on
        any inconsistency.
        """
        tops = set(self.initial_pack.selected_ids)
        if len(tops) != len(self.initial_pack.selected_ids):
            raise ValueError("initial pack selected the same slice twice")
        for entry in self.trade_ups:
            if entry.from_id not in tops:
                raise ValueError(f"trade-up from {entry.from_id} which is not a live top")
            tops.remove(entry.from_id)
            for sid in entry.subsumed_ids:
                if sid not in tops:
                    raise ValueError(f"subsumed {sid} which is not a live top")
                tops.remove(sid)
            if entry.to_id in tops:
                raise ValueError(f"trade-up to {entry.to_id} which is already a live top")
            tops.add(entry.to_id)
        return frozenset(tops)
