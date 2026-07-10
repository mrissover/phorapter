"""Selection strategies: turning a fused candidate list into budget-fitted context.

The engine is **synchronous and pure**. Everything it might need — including the
ancestor slices a trade-up may pull in — is handed to it up front through a
:class:`SliceSource`, so a strategy never performs I/O and, given the same
inputs, produces byte-identical output and trace.

:class:`GreedyUpwardStrategy` (the default) performs the two moves that are safe
by construction:

1. **Dedup (keep the leaf).** Where several retrieved slices nest, keep the
   smallest (the leaf) and fold its retrieved ancestors into *evidence*. Nothing
   relevant is lost — the ancestors' content is a superset of the leaf's — and
   the ranking signal of the ancestors is preserved via the leaf's effective
   rank.
2. **Upward trade-up.** While budget allows, replace a kept slice with its
   enclosing parent. This only ever adds surrounding context; it never drops the
   relevant span, because the parent contains the child's bytes. When one parent
   encloses several kept slices, it *subsumes* them all in one move, and the
   token cost is netted across them.

Downward moves — choosing which part of a large slice to keep when it will not
fit — are out of scope; they require a semantic signal structure alone cannot
supply. New strategies (for instance, global budget packing over the laminar
candidate tree) implement the same :class:`SelectionStrategy` interface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from phorapter.fusion import TierInterleave
from phorapter.grid import DEFAULT_GRID
from phorapter.model import CandidateHit, HitProvenance, RetrievedHit, Slice, SliceRef
from phorapter.tokens import DEFAULT_COUNTER_ID, get_counter
from phorapter.trace import (
    DedupEntry,
    FinalTrace,
    ForestTrace,
    FusionTrace,
    InitialPackTrace,
    RejectionEntry,
    SkippedUnaffordable,
    SubstitutionTrace,
    TradeUpEntry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from phorapter.forest import ContainmentForest
    from phorapter.grid import GridSpec
    from phorapter.tokens import TokenCounter

__all__ = [
    "DedupeOnly",
    "EvidenceItem",
    "GreedyUpwardStrategy",
    "SelectedSlice",
    "Selection",
    "SelectionOptions",
    "SelectionRequest",
    "SelectionStrategy",
    "SliceSource",
    "budget_fit",
]

ExpansionMode = Literal["fill", "retrieved_only", "off"]


@runtime_checkable
class SliceSource(Protocol):
    """Resolves slice refs to slices, synchronously. Unknown refs are omitted.

    The service prefetches the ancestor closure of the packed classes and hands
    it over as one of these; in-process users hand over an in-memory store. Either
    way the strategy stays pure.
    """

    def get_slices(self, refs: Iterable[SliceRef], *, corpus: str) -> dict[SliceRef, Slice]: ...


class _MappingSource:
    """A :class:`SliceSource` backed by a fixed ``{ref: slice}`` mapping."""

    def __init__(self, slices: dict[SliceRef, Slice]) -> None:
        self._slices = slices

    def get_slices(self, refs: Iterable[SliceRef], *, corpus: str) -> dict[SliceRef, Slice]:
        return {ref: self._slices[ref] for ref in refs if ref in self._slices}


@dataclass(frozen=True, slots=True)
class SelectionOptions:
    """Knobs for a selection run.

    ``expansion`` controls trade-up: ``fill`` (default) trades up to any enclosing
    slice the source can supply; ``retrieved_only`` trades up only to slices that
    were themselves retrieved; ``off`` disables trade-up (dedup + pack only).
    ``max_slice_size`` caps how large a traded-up slice may grow.
    ``join_overhead_tokens`` charges a fixed per-slice overhead against the budget
    (for separators/headers when the slices are later concatenated).
    """

    expansion: ExpansionMode = "fill"
    max_slice_size: int | None = None
    join_overhead_tokens: int = 0


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A retrieved hit that supports a selected slice, with its genuine scores.

    Scores are honest and size-scoped: ``raw_score`` is the store score within
    this hit's own size, ``fused_rank`` its position after cross-size fusion.
    """

    ref: SliceRef
    id: str
    size: int
    codepoint_offset: int
    raw_score: float | None
    rank_in_size: int | None
    fused_rank: int


@dataclass(frozen=True, slots=True)
class SelectedSlice:
    """One slice in the final selection, with its provenance and supporting evidence."""

    slice: Slice
    provenance: HitProvenance
    effective_rank: int
    action: Literal["kept", "traded_up"]
    replaced_ids: tuple[str, ...]
    levels: int
    evidence: tuple[EvidenceItem, ...]
    contiguous_with_next: bool = False

    @property
    def id(self) -> str:
        return str(self.slice.ref.uuid())


@dataclass(frozen=True, slots=True)
class Selection:
    """The result of a selection run: chosen slices plus the full decision trace."""

    slices: tuple[SelectedSlice, ...]
    tokens_used: int
    budget: int | None
    trace: SubstitutionTrace


@dataclass(frozen=True, slots=True)
class SelectionRequest:
    """Everything a strategy needs, prefetched. Immutable; strategies never mutate it."""

    candidates: tuple[CandidateHit, ...]
    forest: ContainmentForest
    budget: int | None
    grid: GridSpec
    counter: TokenCounter
    source: SliceSource | None
    options: SelectionOptions = field(default_factory=SelectionOptions)
    corpus: str = "default"
    trust_stored_counts: bool = True
    """Whether per-slice stored token counts are valid under ``counter`` (they are
    unless the request overrode the corpus's pinned tokenizer)."""


@runtime_checkable
class SelectionStrategy(Protocol):
    """Selects and right-sizes slices under a budget."""

    name: str

    def select(self, request: SelectionRequest) -> Selection: ...


# ── internal working state ───────────────────────────────────────────────────


@dataclass(slots=True)
class _Class:
    """A dedup class during selection: one kept slice plus its evidence.

    ``top`` starts as the leaf and grows upward through trade-up. ``alive`` goes
    False when another class subsumes this one.
    """

    top: Slice
    provenance: HitProvenance
    evidence: dict[SliceRef, EvidenceItem]
    eff_rank: int
    order_key: tuple[str, int]  # (document_id, leaf offset) — stable identity for ordering
    alive: bool = True
    levels: int = 0
    replaced: set[str] = field(default_factory=set)
    saturated: bool = False
    skip_sizes: set[int] = field(default_factory=set)
    parked_at: int | None = None
    """Budget level at which this class last hit ``over_budget``. It is retried
    only if the budget later frees back above this level (a negative-delta trade
    elsewhere), so a hopeless target is recorded once, not every round."""

    @property
    def top_id(self) -> str:
        return str(self.top.ref.uuid())

    def priority(self) -> tuple[int, str, int]:
        return (self.eff_rank, self.order_key[0], self.order_key[1])


def _tokens(
    slice_: Slice, counter: TokenCounter, trust_stored: bool, cache: dict[SliceRef, int]
) -> int:
    ref = slice_.ref
    cached = cache.get(ref)
    if cached is not None:
        return cached
    if trust_stored and slice_.token_count is not None:
        value = slice_.token_count
    else:
        value = counter.count(slice_.text)
    cache[ref] = value
    return value


class GreedyUpwardStrategy:
    """Dedup by keeping the leaf, then greedily trade up under the budget."""

    name = "greedy_upward"

    def select(self, request: SelectionRequest) -> Selection:
        forest = request.forest
        options = request.options
        counter = request.counter
        tok_cache: dict[SliceRef, int] = {}

        def tokens(slice_: Slice) -> int:
            return _tokens(slice_, counter, request.trust_stored_counts, tok_cache)

        def cost(slice_: Slice) -> int:
            return tokens(slice_) + options.join_overhead_tokens

        # Index the fused candidates.
        fused_rank: dict[SliceRef, int] = {c.hit.ref: c.fused_rank for c in request.candidates}
        retrieved: dict[SliceRef, RetrievedHit] = {c.hit.ref: c.hit for c in request.candidates}

        fusion_trace = FusionTrace(
            sizes=tuple(sorted({c.hit.slice.size for c in request.candidates})),
            candidate_count=len(request.candidates),
        )
        forest_trace = ForestTrace(
            hit_count=len(forest.hits),
            edge_count=len(forest.edges()),
            anomaly_count=len(forest.anomalies),
            participation_rate=forest.participation_rate(),
            max_depth=forest.max_depth(),
        )

        def evidence_for(ref: SliceRef) -> EvidenceItem | None:
            hit = retrieved.get(ref)
            if hit is None:
                return None
            return EvidenceItem(
                ref=ref,
                id=str(ref.uuid()),
                size=hit.slice.size,
                codepoint_offset=hit.slice.codepoint_offset,
                raw_score=hit.score,
                rank_in_size=hit.rank_in_size,
                fused_rank=fused_rank.get(ref, len(request.candidates)),
            )

        # ── (b) dedup: one class per forest leaf ────────────────────────────
        classes: list[_Class] = []
        dedup_entries: list[DedupEntry] = []
        for leaf in forest.leaves():
            ancestors = forest.retrieved_ancestors(leaf.ref)
            evidence: dict[SliceRef, EvidenceItem] = {}
            for ref in (leaf.ref, *ancestors):
                item = evidence_for(ref)
                if item is not None:
                    evidence[ref] = item
            eff_rank = min(
                (item.fused_rank for item in evidence.values()), default=len(request.candidates)
            )
            classes.append(
                _Class(
                    top=leaf.slice,
                    provenance=HitProvenance.RETRIEVED,
                    evidence=evidence,
                    eff_rank=eff_rank,
                    order_key=(leaf.slice.document_id, leaf.slice.codepoint_offset),
                )
            )
            if ancestors:
                dedup_entries.append(
                    DedupEntry(
                        kept_ref=leaf.ref,
                        kept_id=str(leaf.ref.uuid()),
                        dropped_refs=ancestors,
                        dropped_ids=tuple(str(r.uuid()) for r in ancestors),
                    )
                )

        classes.sort(key=lambda c: c.priority())

        # ── budget=None: dedup only, no packing or trade-up ─────────────────
        if request.budget is None:
            tokens_used = sum(tokens(c.top) for c in classes)
            initial_pack = InitialPackTrace(
                selected_ids=tuple(c.top_id for c in classes),
                skipped_unaffordable=(),
                smallest_unaffordable_tokens=None,
            )
            trace = SubstitutionTrace(
                fusion=fusion_trace,
                forest=forest_trace,
                dedup=tuple(dedup_entries),
                initial_pack=initial_pack,
                trade_ups=(),
                rejections=(),
                final=FinalTrace(
                    tokens_used=tokens_used, budget=None, utilization=None, budget_exhausted=False
                ),
            )
            return self._finalize(classes, tokens_used, None, trace)

        # ── (c) initial pack: first-fit in priority order ──────────────────
        budget = request.budget
        budget_left = budget
        packed: list[_Class] = []
        skipped: list[SkippedUnaffordable] = []
        for cls in classes:
            c = cost(cls.top)
            if c <= budget_left:
                packed.append(cls)
                budget_left -= c
            else:
                cls.alive = False
                skipped.append(
                    SkippedUnaffordable(
                        ref=cls.top.ref, id=cls.top_id, tokens=c, budget_left=budget_left
                    )
                )
        smallest_unaffordable = min((s.tokens for s in skipped), default=None)
        initial_pack = InitialPackTrace(
            selected_ids=tuple(c.top_id for c in packed),
            skipped_unaffordable=tuple(skipped),
            smallest_unaffordable_tokens=smallest_unaffordable,
        )

        trade_ups: list[TradeUpEntry] = []
        rejections: list[RejectionEntry] = []

        # ── (d) trade-up: round-robin, one content level per class per round ─
        if options.expansion != "off" and packed:
            budget_left = self._trade_up(
                packed=packed,
                request=request,
                retrieved=retrieved,
                tokens=tokens,
                cost=cost,
                evidence_for=evidence_for,
                budget_left=budget_left,
                trade_ups=trade_ups,
                rejections=rejections,
            )

        alive = [c for c in packed if c.alive]
        tokens_used = sum(tokens(c.top) for c in alive)
        budget_exhausted = bool(skipped) or any(r.reason == "over_budget" for r in rejections)
        final = FinalTrace(
            tokens_used=tokens_used,
            budget=budget,
            utilization=(tokens_used / budget) if budget else None,
            budget_exhausted=budget_exhausted,
        )
        trace = SubstitutionTrace(
            fusion=fusion_trace,
            forest=forest_trace,
            dedup=tuple(dedup_entries),
            initial_pack=initial_pack,
            trade_ups=tuple(trade_ups),
            rejections=tuple(rejections),
            final=final,
        )
        return self._finalize(alive, tokens_used, budget, trace)

    def _trade_up(
        self,
        *,
        packed: list[_Class],
        request: SelectionRequest,
        retrieved: dict[SliceRef, RetrievedHit],
        tokens: Callable[[Slice], int],
        cost: Callable[[Slice], int],
        evidence_for: Callable[[SliceRef], EvidenceItem | None],
        budget_left: int,
        trade_ups: list[TradeUpEntry],
        rejections: list[RejectionEntry],
    ) -> int:
        grid = request.grid
        options = request.options
        retrieved_only = options.expansion == "retrieved_only"

        # Effective source: retrieved slices always resolvable; the prefetched
        # closure supplies un-retrieved ancestors in ``fill`` mode.
        retrieved_slices = {ref: hit.slice for ref, hit in retrieved.items()}
        base_source = None if retrieved_only else request.source

        def resolve(ref: SliceRef) -> tuple[Slice | None, str]:
            if ref in retrieved_slices:
                return retrieved_slices[ref], "retrieved"
            if base_source is not None:
                found = base_source.get_slices([ref], corpus=request.corpus)
                if ref in found:
                    return found[ref], "fetched"
            return None, "fetched"

        def next_target_size(cls: _Class) -> int | None:
            if cls.levels >= len(grid.sizes) - 1:
                return None
            for size in grid.levels_above(cls.top.size):
                if size in cls.skip_sizes:
                    continue
                if options.max_slice_size is not None and size > options.max_slice_size:
                    return None
                offset = grid.parent_offset(cls.top.codepoint_offset, size)
                # Skip degenerate levels whose clipped span equals the current one.
                end = min(offset + size, cls.top.document_codepoint_length)
                if end - offset <= cls.top.codepoint_length and offset == cls.top.codepoint_offset:
                    continue
                return size
            return None

        max_rounds = len(packed) * len(grid.sizes) + 2
        round_no = 0
        while round_no < max_rounds:
            round_no += 1
            accepted = 0
            progressed = False
            for cls in sorted(
                (c for c in packed if c.alive and not c.saturated), key=lambda c: c.priority()
            ):
                if not cls.alive or cls.saturated:
                    continue
                if cls.parked_at is not None:
                    if budget_left <= cls.parked_at:
                        continue  # still parked; no budget has freed up
                    cls.parked_at = None  # budget grew — worth another look
                size = next_target_size(cls)
                if size is None:
                    cls.saturated = True
                    progressed = True
                    rejections.append(
                        RejectionEntry(round=round_no, target=cls.top.ref, reason="saturated")
                    )
                    continue
                parent_ref = SliceRef(
                    cls.top.document_id, size, grid.parent_offset(cls.top.codepoint_offset, size)
                )
                parent, provenance = resolve(parent_ref)
                if parent is None:
                    cls.skip_sizes.add(size)
                    progressed = True
                    rejections.append(
                        RejectionEntry(round=round_no, target=parent_ref, reason="fetch_miss")
                    )
                    continue
                if cls.top.own_marker not in parent.descendant_markers:
                    cls.skip_sizes.add(size)
                    progressed = True
                    rejections.append(
                        RejectionEntry(round=round_no, target=parent_ref, reason="stale_parent")
                    )
                    continue

                # Subsumption: other alive classes whose top the parent encloses
                # (or that already sit on the parent), marker-verified.
                subsumed = [
                    other
                    for other in packed
                    if other is not cls
                    and other.alive
                    and (
                        other.top.ref == parent.ref
                        or (
                            other.top.document_id == parent.document_id
                            and parent.size > other.top.size
                            and other.top.own_marker in parent.descendant_markers
                        )
                    )
                ]
                delta = cost(parent) - cost(cls.top) - sum(cost(o.top) for o in subsumed)
                if delta > budget_left:
                    cls.parked_at = budget_left
                    progressed = True
                    rejections.append(
                        RejectionEntry(
                            round=round_no,
                            target=parent_ref,
                            reason="over_budget",
                            delta_tokens=delta,
                        )
                    )
                    continue

                # Accept.
                budget_left -= delta
                from_ref = cls.top.ref
                from_id = cls.top_id
                subsumed_ids = tuple(o.top_id for o in subsumed)
                cls.replaced.add(from_id)
                for other in subsumed:
                    other.alive = False
                    cls.replaced.add(other.top_id)
                    cls.replaced.update(other.replaced)
                    for ref, item in other.evidence.items():
                        cls.evidence.setdefault(ref, item)
                    cls.eff_rank = min(cls.eff_rank, other.eff_rank)
                # Fold in the parent's own retrieved evidence, if any.
                parent_item = evidence_for(parent.ref)
                if parent_item is not None:
                    cls.evidence.setdefault(parent.ref, parent_item)
                    cls.eff_rank = min(cls.eff_rank, parent_item.fused_rank)
                cls.top = parent
                cls.levels += 1
                cls.provenance = (
                    HitProvenance.RETRIEVED
                    if provenance == "retrieved"
                    else HitProvenance.TRADED_UP
                )
                trade_ups.append(
                    TradeUpEntry(
                        round=round_no,
                        from_ref=from_ref,
                        from_id=from_id,
                        to_ref=parent.ref,
                        to_id=cls.top_id,
                        to_provenance=provenance,  # type: ignore[arg-type]
                        subsumed_ids=subsumed_ids,
                        delta_tokens=delta,
                        budget_left=budget_left,
                    )
                )
                accepted += 1
                progressed = True
            if accepted == 0 and not progressed:
                break
            if all(c.saturated or not c.alive for c in packed):
                break
        return budget_left

    def _finalize(
        self,
        classes: list[_Class],
        tokens_used: int,
        budget: int | None,
        trace: SubstitutionTrace,
    ) -> Selection:
        # Order: documents by best class rank; within a document, reading order.
        best_rank_by_doc: dict[str, int] = {}
        for cls in classes:
            doc = cls.top.document_id
            best_rank_by_doc[doc] = min(best_rank_by_doc.get(doc, cls.eff_rank), cls.eff_rank)
        ordered = sorted(
            classes,
            key=lambda c: (
                best_rank_by_doc[c.top.document_id],
                c.top.document_id,
                c.top.codepoint_offset,
            ),
        )

        selected: list[SelectedSlice] = []
        for i, cls in enumerate(ordered):
            nxt = ordered[i + 1] if i + 1 < len(ordered) else None
            contiguous = (
                nxt is not None
                and nxt.top.document_id == cls.top.document_id
                and nxt.top.codepoint_offset == cls.top.codepoint_end
            )
            evidence = tuple(
                sorted(cls.evidence.values(), key=lambda e: (e.size, e.codepoint_offset))
            )
            replaced = tuple(sorted(rid for rid in cls.replaced if rid != cls.top_id))
            selected.append(
                SelectedSlice(
                    slice=cls.top,
                    provenance=cls.provenance,
                    effective_rank=cls.eff_rank,
                    action="traded_up" if cls.levels > 0 else "kept",
                    replaced_ids=replaced,
                    levels=cls.levels,
                    evidence=evidence,
                    contiguous_with_next=contiguous,
                )
            )
        return Selection(
            slices=tuple(selected), tokens_used=tokens_used, budget=budget, trace=trace
        )


class DedupeOnly:
    """Dedup by keeping the leaf; no budget, no trade-up."""

    name = "dedupe_only"

    def select(self, request: SelectionRequest) -> Selection:
        stripped = SelectionRequest(
            candidates=request.candidates,
            forest=request.forest,
            budget=None,
            grid=request.grid,
            counter=request.counter,
            source=request.source,
            options=request.options,
            corpus=request.corpus,
            trust_stored_counts=request.trust_stored_counts,
        )
        return GreedyUpwardStrategy().select(stripped)


def budget_fit(
    hits: Sequence[RetrievedHit],
    *,
    budget: int | None,
    counter: TokenCounter | None = None,
    source: SliceSource | None = None,
    strategy: SelectionStrategy | None = None,
    grid: GridSpec = DEFAULT_GRID,
    corpus: str = "default",
    options: SelectionOptions | None = None,
) -> Selection:
    """Right-size a set of retrieved hits under a token budget, in process.

    Fuses the hits (tier interleave), builds the containment forest, and runs the
    strategy (:class:`GreedyUpwardStrategy` by default). With ``source=None``,
    trade-up is limited to the retrieved hits themselves; pass a
    :class:`SliceSource` (e.g. an in-memory store) to let trade-up reach
    un-retrieved ancestors.
    """
    from phorapter.forest import ContainmentForest

    counter = counter or get_counter(DEFAULT_COUNTER_ID)
    strategy = strategy or GreedyUpwardStrategy()
    options = options or SelectionOptions()

    by_size: dict[int, list[RetrievedHit]] = defaultdict(list)
    for hit in hits:
        by_size[hit.slice.size].append(hit)
    candidates = tuple(TierInterleave().fuse(by_size))
    forest = ContainmentForest.build(hits, grid)
    effective_source = (
        source if source is not None else _MappingSource({hit.ref: hit.slice for hit in hits})
    )
    request = SelectionRequest(
        candidates=candidates,
        forest=forest,
        budget=budget,
        grid=grid,
        counter=counter,
        source=effective_source,
        options=options,
        corpus=corpus,
    )
    return strategy.select(request)
