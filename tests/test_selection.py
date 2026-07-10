"""Selection engine: dedup, budget packing, greedy upward trade-up, edge cases."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from phorapter import DEFAULT_GRID, multi_view_slice
from phorapter.model import HitProvenance, RetrievedHit, Slice
from phorapter.selection import (
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
    from phorapter.forest import ContainmentForest
    from phorapter.fusion import TierInterleave
    from phorapter.selection import SelectionRequest

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

    # Substitutions are strictly upward: a traded-up slice is larger than every id
    # it replaced.
    by_ref_size = {s.ref: s.size for s in all_slices}
    for selected in sel.slices:
        for replaced_id in selected.replaced_ids:
            replaced_size = next(
                (sz for ref, sz in by_ref_size.items() if str(ref.uuid()) == replaced_id), None
            )
            if replaced_size is not None:
                assert selected.slice.size > replaced_size

    # The lifecycle invariant is machine-checkable and holds.
    assert sel.trace.replay_top_ids() == {s.id for s in sel.slices}

    # Determinism: a re-run is byte-identical.
    again = budget_fit(hits, budget=budget, counter=CharCounter(), source=DictSource(all_slices))
    assert [s.id for s in again.slices] == [s.id for s in sel.slices]
