"""Selection engine: dedup, budget packing, greedy upward trade-up, edge cases."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from phoropter import DEFAULT_GRID, multi_view_slice
from phoropter.model import HitProvenance, RetrievedHit, Slice
from phoropter.selection import (
    DedupeOnly,
    SelectionOptions,
    budget_fit,
)


class CharCounter:
    """One token per code point — predictable budgeting without tiktoken."""

    counter_id = "test:chars"

    def __init__(self) -> None:
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        return len(text)


class DictSource:
    def __init__(self, slices: list[Slice]) -> None:
        self._by_ref = {s.ref: s for s in slices}

    def get_slices(self, refs, *, corpus):
        return {ref: self._by_ref[ref] for ref in refs if ref in self._by_ref}


def make_hit(doc, size, offset, score, rank=0):
    return RetrievedHit(
        slice=doc.slice_at(size, offset),
        corpus="default",
        score=score,
        rank_in_size=rank,
        provenance=HitProvenance.RETRIEVED,
    )


DOC = multi_view_slice("doc-a", "x" * 1024, DEFAULT_GRID)


def _replay_consistent(selection) -> bool:
    return selection.trace.replay_top_ids() == {s.id for s in selection.slices}


class TestDedup:
    def test_keep_the_leaf_folds_ancestors_into_evidence(self) -> None:
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 128, 0, 0.7)]
        sel = budget_fit(hits, budget=None, counter=CharCounter())
        assert len(sel.slices) == 1
        assert sel.slices[0].slice.size == 64  # the leaf is kept
        assert {e.size for e in sel.slices[0].evidence} == {64, 128}
        assert _replay_consistent(sel)

    def test_dedupe_only_strategy_ignores_budget(self) -> None:
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 128, 0, 0.7)]
        sel = DedupeOnly().select(budget_fit_request(hits, budget=5, counter=CharCounter()))
        assert [s.slice.size for s in sel.slices] == [64]


def budget_fit_request(hits, *, budget, counter):
    # Build the same request budget_fit would, for strategies under test.
    from phoropter.forest import ContainmentForest
    from phoropter.fusion import TierInterleave
    from phoropter.selection import SelectionRequest

    by_size: dict[int, list] = {}
    for h in hits:
        by_size.setdefault(h.slice.size, []).append(h)
    return SelectionRequest(
        candidates=tuple(TierInterleave().fuse(by_size)),
        forest=ContainmentForest.build(hits, DEFAULT_GRID),
        budget=budget,
        grid=DEFAULT_GRID,
        counter=counter,
        source=DictSource([h.slice for h in hits]),
    )


class TestTradeUp:
    def test_two_leaves_subsume_into_retrieved_parent(self) -> None:
        hits = [
            make_hit(DOC, 64, 0, 0.9),
            make_hit(DOC, 64, 64, 0.8),
            make_hit(DOC, 128, 0, 0.7),
        ]
        sel = budget_fit(hits, budget=1000, counter=CharCounter())
        assert len(sel.slices) == 1
        top = sel.slices[0]
        assert top.slice.size == 128
        assert top.action == "traded_up"
        assert top.provenance is HitProvenance.RETRIEVED  # the 128 was retrieved
        assert len(sel.trace.trade_ups) == 1
        assert _replay_consistent(sel)

    def test_trade_up_to_unretrieved_parent_is_fetched(self) -> None:
        # Retrieve only the two 64 leaves; the 128 parent is supplied by the source.
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 64, 64, 0.8)]
        source = DictSource([DOC.slice_at(128, 0)])
        sel = budget_fit(hits, budget=1000, counter=CharCounter(), source=source)
        assert len(sel.slices) == 1
        top = sel.slices[0]
        assert top.slice.size == 128
        assert top.provenance is HitProvenance.TRADED_UP  # 128 not retrieved
        assert sel.trace.trade_ups[-1].to_provenance == "fetched"
        assert _replay_consistent(sel)

    def test_retrieved_only_mode_does_not_fetch(self) -> None:
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 64, 64, 0.8)]
        source = DictSource([DOC.slice_at(128, 0)])
        sel = budget_fit(
            hits,
            budget=1000,
            counter=CharCounter(),
            source=source,
            options=SelectionOptions(expansion="retrieved_only"),
        )
        # Nothing to trade up to among retrieved hits: both leaves stay.
        assert sorted(s.slice.size for s in sel.slices) == [64, 64]
        assert sel.trace.trade_ups == ()

    def test_expansion_off_disables_trade_up(self) -> None:
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 64, 64, 0.8), make_hit(DOC, 128, 0, 0.7)]
        sel = budget_fit(
            hits, budget=1000, counter=CharCounter(), options=SelectionOptions(expansion="off")
        )
        assert sorted(s.slice.size for s in sel.slices) == [64, 64]
        assert sel.trace.trade_ups == ()

    def test_max_slice_size_caps_trade_up(self) -> None:
        hits = [make_hit(DOC, s, 0, 0.9) for s in DEFAULT_GRID.sizes]
        sel = budget_fit(
            hits, budget=100000, counter=CharCounter(), options=SelectionOptions(max_slice_size=256)
        )
        assert len(sel.slices) == 1
        assert sel.slices[0].slice.size == 256  # capped, not 1024


class TestBudgetEdges:
    def test_all_unaffordable_yields_empty_with_shortfall(self) -> None:
        hits = [make_hit(DOC, 128, 0, 0.9)]  # cost 128
        sel = budget_fit(hits, budget=10, counter=CharCounter())
        assert sel.slices == ()
        assert sel.trace.final.budget_exhausted
        assert sel.trace.initial_pack.smallest_unaffordable_tokens == 128
        assert _replay_consistent(sel)

    def test_partial_pack_skips_unaffordable_and_continues(self) -> None:
        # Budget fits one 64 but not the second; the walk continues past the skip.
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 64, 64, 0.8)]
        sel = budget_fit(
            hits, budget=100, counter=CharCounter(), options=SelectionOptions(expansion="off")
        )
        assert len(sel.slices) == 1
        assert len(sel.trace.initial_pack.skipped_unaffordable) == 1

    def test_over_budget_trade_up_rejected(self) -> None:
        # One 64 packed with budget 100; trading to the 128 costs +64 > 36 left.
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 128, 0, 0.7)]
        sel = budget_fit(hits, budget=100, counter=CharCounter())
        # 64 leaf kept (128 folded into evidence); trade-up to 128 over budget.
        assert [s.slice.size for s in sel.slices] == [64]
        assert any(r.reason == "over_budget" for r in sel.trace.rejections)


class TestStalenessAndMisses:
    def test_stale_parent_rejected(self) -> None:
        import dataclasses

        leaf = make_hit(DOC, 64, 0, 0.9)
        stale_128 = dataclasses.replace(DOC.slice_at(128, 0), descendant_markers=())
        source = DictSource([stale_128])
        sel = budget_fit([leaf], budget=1000, counter=CharCounter(), source=source)
        assert [s.slice.size for s in sel.slices] == [64]
        assert any(r.reason == "stale_parent" for r in sel.trace.rejections)

    def test_fetch_miss_keeps_top(self) -> None:
        leaf = make_hit(DOC, 64, 0, 0.9)
        sel = budget_fit([leaf], budget=1000, counter=CharCounter(), source=DictSource([]))
        assert [s.slice.size for s in sel.slices] == [64]
        assert any(r.reason == "fetch_miss" for r in sel.trace.rejections)


class TestShortDocumentDegenerateLevels:
    def test_degenerate_levels_skipped(self) -> None:
        # 100-cp doc: the 128 slice already spans the whole document, so 256/512/
        # 1024 are byte-identical no-ops and must not be counted as trade-ups.
        short = multi_view_slice("doc-s", "x" * 100, DEFAULT_GRID)
        hits = [make_hit(short, 64, 0, 0.9), make_hit(short, 64, 64, 0.8)]
        source = DictSource([short.slice_at(s, 0) for s in (128, 256, 512, 1024)])
        sel = budget_fit(hits, budget=100000, counter=CharCounter(), source=source)
        assert len(sel.slices) == 1
        top = sel.slices[0]
        assert top.slice.size == 128  # stops at 128, not 1024
        assert top.slice.codepoint_length == 100
        assert top.levels == 1  # one real content level, degenerate ones skipped


class TestTokenCounting:
    def test_counts_memoized_per_ref(self) -> None:
        counter = CharCounter()
        hits = [make_hit(DOC, 64, 0, 0.9), make_hit(DOC, 128, 0, 0.7)]
        budget_fit(hits, budget=None, counter=counter)
        # Two distinct slices touched; each counted at most once (memoized).
        assert counter.calls <= 2

    def test_stored_counts_used_when_present(self) -> None:
        doc = multi_view_slice(
            "doc-t", "hello world " * 30, DEFAULT_GRID, token_counter=CharCounter()
        )
        assert doc.slice_at(64, 0).token_count is not None
        counter = CharCounter()
        hits = [make_hit(doc, 64, 0, 0.9)]
        budget_fit(hits, budget=1000, counter=counter)
        # Stored counts trusted: the leaf's count is not recomputed. (Trade-up may
        # still count fetched parents, but there is no source here.)
        assert counter.calls == 0


class TestDeterminism:
    def test_identical_runs_identical_trace(self) -> None:
        hits = [make_hit(DOC, s, 0, 0.9 - 0.1 * i) for i, s in enumerate(DEFAULT_GRID.sizes)]
        a = budget_fit(hits, budget=500, counter=CharCounter())
        b = budget_fit(hits, budget=500, counter=CharCounter())
        assert repr(a.trace) == repr(b.trace)
        assert [s.id for s in a.slices] == [s.id for s in b.slices]


class TestSingleDocumentCollapse:
    def test_whole_chain_collapses_within_budget(self) -> None:
        hits = [make_hit(DOC, s, 0, 0.9) for s in DEFAULT_GRID.sizes]
        sel = budget_fit(hits, budget=100000, counter=CharCounter())
        assert len(sel.slices) == 1
        assert sel.slices[0].slice.size == 1024
        assert _replay_consistent(sel)


# ── property-based safety net ────────────────────────────────────────────────


@st.composite
def hit_sets(draw):
    n = draw(st.integers(min_value=800, max_value=2048))
    doc = multi_view_slice("doc-p", "x" * n, DEFAULT_GRID)
    sizes = draw(
        st.lists(st.sampled_from(DEFAULT_GRID.sizes), min_size=1, max_size=6, unique=False)
    )
    hits = []
    for i, size in enumerate(sizes):
        offsets = list(range(0, n, size))
        offset = offsets[draw(st.integers(min_value=0, max_value=len(offsets) - 1))]
        hits.append(make_hit(doc, size, offset, score=1.0 - i * 0.01, rank=i))
    # de-dup by ref to avoid trivial repeats
    seen = set()
    unique = []
    for h in hits:
        if h.ref not in seen:
            seen.add(h.ref)
            unique.append(h)
    return doc, unique


@settings(max_examples=80, deadline=None)
@given(data=hit_sets(), budget=st.integers(min_value=1, max_value=4000))
def test_properties_hold(data, budget: int) -> None:
    doc, hits = data
    all_slices = list(doc.slices)
    sel = budget_fit(hits, budget=budget, counter=CharCounter(), source=DictSource(all_slices))

    # Never exceeds budget.
    assert sel.tokens_used <= budget

    # Substitutions are information-preserving: a selected slice's span must
    # actually CONTAIN every slice it replaced (larger size AND covering offsets),
    # not merely be larger. This is the guard against marker-collision false
    # subsumption on repeated text.
    by_id = {str(s.ref.uuid()): s for s in all_slices}
    for selected in sel.slices:
        for replaced_id in selected.replaced_ids:
            replaced = by_id.get(replaced_id)
            if replaced is None:
                continue
            assert selected.slice.size > replaced.size
            assert selected.slice.document_id == replaced.document_id
            # The replaced slice's span must lie inside the selected slice's span.
            # (This is the property-level guard against marker-collision false
            # subsumption: a disjoint same-text slice would violate it.)
            assert selected.slice.codepoint_offset <= replaced.codepoint_offset
            assert replaced.codepoint_end <= selected.slice.codepoint_end

    # The lifecycle invariant is machine-checkable and holds.
    assert sel.trace.replay_top_ids() == {s.id for s in sel.slices}

    # Determinism: a re-run is byte-identical, trace included.
    again = budget_fit(hits, budget=budget, counter=CharCounter(), source=DictSource(all_slices))
    assert [s.id for s in again.slices] == [s.id for s in sel.slices]
    assert repr(again.trace) == repr(sel.trace)


class TestRepeatedTextSubsumption:
    """Regression: repeated text must not fabricate a subsumption in trade-up.

    Byte-identical slices at disjoint offsets share a marker, so the parent's
    descendant-marker list vouches for a slice it does not enclose. Subsumption
    must decide containment positionally, not by marker alone.
    """

    def test_disjoint_repeated_leaf_is_not_absorbed(self) -> None:
        doc = multi_view_slice("doc-a", "x" * 1024, DEFAULT_GRID)
        # Two disjoint 64 leaves; the second is far outside the 128@0 parent.
        hits = [make_hit(doc, 64, 0, 0.95), make_hit(doc, 64, 192, 0.90)]
        source = DictSource([doc.slice_at(128, 0)])
        sel = budget_fit(hits, budget=200, counter=CharCounter(), source=source)
        # 64@0 trades up to 128@0 ([0,128)); 64@192 ([192,256)) must survive.
        spans = sorted((s.slice.size, s.slice.codepoint_offset) for s in sel.slices)
        assert spans == [(64, 192), (128, 0)]
        # The region [192,256) is still covered.
        assert any(
            s.slice.codepoint_offset == 192 and s.slice.codepoint_end == 256 for s in sel.slices
        )

    def test_many_disjoint_repeated_leaves_all_survive(self) -> None:
        doc = multi_view_slice("doc-a", "x" * 1024, DEFAULT_GRID)
        offsets = [0, 128, 256, 384, 512, 640, 768, 896]
        hits = [make_hit(doc, 64, o, 0.9 - i * 0.01) for i, o in enumerate(offsets)]
        source = DictSource([doc.slice_at(128, 0)])
        sel = budget_fit(hits, budget=10000, counter=CharCounter(), source=source)
        # 64@0 -> 128@0 covers [0,128); the other seven disjoint leaves remain.
        assert len(sel.slices) == 8
        covered_ends = {s.slice.codepoint_end for s in sel.slices}
        assert covered_ends == {128, 192, 320, 448, 576, 704, 832, 960}


class TestTraceDeterminism:
    def test_dedup_trace_order_independent_of_input_order(self) -> None:
        doc = multi_view_slice("doc-a", "".join(chr(0x100 + i) for i in range(1024)), DEFAULT_GRID)
        base = [
            make_hit(doc, 64, 0, 0.9),
            make_hit(doc, 128, 0, 0.8),
            make_hit(doc, 64, 512, 0.7),
            make_hit(doc, 128, 512, 0.6),
        ]
        a = budget_fit(
            base, budget=5000, counter=CharCounter(), options=SelectionOptions(expansion="off")
        )
        b = budget_fit(
            list(reversed(base)),
            budget=5000,
            counter=CharCounter(),
            options=SelectionOptions(expansion="off"),
        )
        assert repr(a.trace) == repr(b.trace)


class TestJoinOverhead:
    def test_overhead_counted_against_budget(self) -> None:
        doc = multi_view_slice("doc-a", "".join(chr(0x100 + i) for i in range(1024)), DEFAULT_GRID)
        # Three disjoint 128 leaves, each 128 tokens + 40 overhead = 168; budget 520
        # admits three (504) but not with any trade-up headroom.
        hits = [make_hit(doc, 128, o, 0.9) for o in (0, 256, 512)]
        sel = budget_fit(
            hits,
            budget=520,
            counter=CharCounter(),
            options=SelectionOptions(expansion="off", join_overhead_tokens=40),
        )
        assert len(sel.slices) == 3
        # tokens_used includes the per-slice overhead and stays within budget.
        assert sel.tokens_used == 3 * (128 + 40)
        assert sel.tokens_used <= 520
        assert sel.trace.final.utilization == sel.tokens_used / 520
