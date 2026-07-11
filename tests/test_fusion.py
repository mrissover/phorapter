"""Rank fusion: tier interleave ordering, tie stability, degradation."""

from phoropter import DEFAULT_GRID, multi_view_slice
from phoropter.fusion import RawScorePool, TierInterleave
from phoropter.model import HitProvenance, RetrievedHit

DOC = multi_view_slice("doc-a", "x" * 4096, DEFAULT_GRID)


def hit(size, offset, score, rank):
    return RetrievedHit(
        slice=DOC.slice_at(size, offset),
        corpus="c",
        score=score,
        rank_in_size=rank,
        provenance=HitProvenance.RETRIEVED,
    )


def test_tier_interleave_orders_by_tier_then_size() -> None:
    # Three sizes, two ranks each. Tier 0 = best of each size (size ascending),
    # then tier 1. Scores across sizes are never compared.
    by_size = {
        64: [hit(64, 0, 0.5, 0), hit(64, 64, 0.4, 1)],
        128: [hit(128, 0, 0.99, 0), hit(128, 128, 0.98, 1)],
        256: [hit(256, 0, 0.1, 0), hit(256, 256, 0.05, 1)],
    }
    fused = TierInterleave().fuse(by_size)
    order = [(c.hit.slice.size, c.hit.slice.codepoint_offset) for c in fused]
    assert order == [(64, 0), (128, 0), (256, 0), (64, 64), (128, 128), (256, 256)]
    assert [c.fused_rank for c in fused] == list(range(6))


def test_within_size_sorted_by_score_then_id() -> None:
    # Out-of-order input; each size is re-sorted by score desc before interleaving.
    by_size = {64: [hit(64, 64, 0.3, 9), hit(64, 0, 0.9, 9)]}
    fused = TierInterleave().fuse(by_size)
    assert [c.hit.slice.codepoint_offset for c in fused] == [0, 64]


def test_equal_scores_break_by_point_id_deterministically() -> None:
    by_size = {64: [hit(64, 0, 0.5, 0), hit(64, 64, 0.5, 1)]}
    a = TierInterleave().fuse(by_size)
    b = TierInterleave().fuse(dict(reversed(list(by_size.items()))))
    assert [c.hit.ref for c in a] == [c.hit.ref for c in b]


def test_missing_size_degrades_gracefully() -> None:
    by_size = {64: [hit(64, 0, 0.5, 0)], 256: []}
    fused = TierInterleave().fuse(by_size)
    assert [c.hit.slice.size for c in fused] == [64]


def test_empty_input() -> None:
    assert TierInterleave().fuse({}) == []
    assert TierInterleave().fuse({64: []}) == []


def test_raw_score_pool_sorts_by_score() -> None:
    by_size = {
        64: [hit(64, 0, 0.2, 0)],
        1024: [hit(1024, 0, 0.9, 0)],
    }
    fused = RawScorePool().fuse(by_size)
    assert [c.hit.slice.size for c in fused] == [1024, 64]
