"""The golden worked example: one fully hand-derived selection, pinned end to end.

Scenario (budget = 800 tokens, one token per code point via the char counter):

Document "hb" (1024 code points), retrieved slices:
  H1  64@0    score 0.95  rank 0   50 tokens
  H2  64@64   score 0.90  rank 1   50 tokens
  H3  128@0   score 0.80  rank 0  120 tokens   (encloses H1, H2)
  H4  256@0   score 0.70  rank 0  700 tokens   (encloses H3 — deliberately huge)
Document "faq" (256 code points), retrieved slices:
  H5  64@0    score 0.85  rank 0   60 tokens

The source (prefetched ancestor closure) also supplies un-retrieved parents:
  hb 128@0 (120), hb 256@0 (700), faq 128@0 (110), faq 256@0 (200).

Fusion (tier interleave). Per size, sorted by score descending:
64 -> [H1(0.95), H2(0.90), H5(0.85)]; 128 -> [H3]; 256 -> [H4]. Tier 0 (size
ascending) = H1, H3, H4; tier 1 = H2; tier 2 = H5. Resulting fused ranks:
H1=0, H3=1, H4=2, H2=3, H5=4.

Dedup (keep the leaf). Forest leaves are H1, H2, H5 (each has no retrieved slice
inside it). Their classes:
  C1 = leaf H1(64@0),  ancestors {128@0, 256@0} -> eff_rank min(0,1,2) = 0
  C2 = leaf H2(64@64), ancestors {128@0, 256@0} -> eff_rank min(3,1,2) = 1
  C5 = leaf H5(64@0 faq), ancestors {}          -> eff_rank 4
So two dedup entries (C1 and C2 fold in ancestors; C5 has none).

Initial pack, priority (eff_rank, doc, offset) = C1, C2, C5:
  C1 50 -> left 750;  C2 50 -> left 700;  C5 60 -> left 640. All fit.

Trade-up, round 1 (priority C1, C2, C5):
  C1: 64@0 -> 128@0 (retrieved). Parent encloses C2's 64@64 -> subsume it.
      delta = 120 - 50 - 50 = 20; left 640 - 20 = 620. C1 now on 128@0
      (provenance retrieved). C2 absorbed.
  C2: absorbed, skipped.
  C5: 64@0 -> 128@0 faq (fetched). No sibling to subsume.
      delta = 110 - 60 = 50; left 620 - 50 = 570.
Round 2 (priority C1, C5):
  C1: 128@0 -> 256@0 (fetched). delta = 700 - 120 = 580 > 570 left -> over budget.
      C1 is parked at 570 and stays on 128@0.
  C5: 128@0 faq -> 256@0 faq (fetched). delta = 200 - 110 = 90 <= 570; left 480.
      C5 now on 256@0 faq (the whole faq document).
Round 3:
  C1: parked at 570, budget 480 <= 570 -> still parked, no attempt.
  C5: on 256@0 (whole 256-cp doc); 512 and 1024 levels are degenerate (same
      span) -> saturated.
Round 4: no class can progress -> stop.

Final selection: hb 128@0 (traded up from two 64s), faq 256@0 (traded up two
levels). tokens_used = 120 + 200 = 320. Budget 800 was constrained (the hb 256
trade did not fit), so budget_exhausted is True.
"""

from __future__ import annotations

import dataclasses

from phorapter import DEFAULT_GRID, multi_view_slice
from phorapter.model import HitProvenance, RetrievedHit
from phorapter.selection import budget_fit

HB = multi_view_slice("hb", "x" * 1024, DEFAULT_GRID)
FAQ = multi_view_slice("faq", "y" * 256, DEFAULT_GRID)


class Chars:
    counter_id = "test:chars"

    def count(self, text: str) -> int:
        return len(text)


def _slice(doc, size, offset, tok):
    return dataclasses.replace(doc.slice_at(size, offset), token_count=tok)


def _hit(doc, size, offset, score, rank, tok):
    return RetrievedHit(
        slice=_slice(doc, size, offset, tok),
        corpus="default",
        score=score,
        rank_in_size=rank,
        provenance=HitProvenance.RETRIEVED,
    )


class _Source:
    def __init__(self, slices):
        self._by_ref = {s.ref: s for s in slices}

    def get_slices(self, refs, *, corpus):
        return {ref: self._by_ref[ref] for ref in refs if ref in self._by_ref}


def _run():
    hits = [
        _hit(HB, 64, 0, 0.95, 0, 50),
        _hit(HB, 64, 64, 0.90, 1, 50),
        _hit(HB, 128, 0, 0.80, 0, 120),
        _hit(HB, 256, 0, 0.70, 0, 700),
        _hit(FAQ, 64, 0, 0.85, 0, 60),
    ]
    source = _Source(
        [
            _slice(HB, 128, 0, 120),
            _slice(HB, 256, 0, 700),
            _slice(FAQ, 128, 0, 110),
            _slice(FAQ, 256, 0, 200),
        ]
    )
    return budget_fit(hits, budget=800, counter=Chars(), source=source)


def test_final_selection() -> None:
    sel = _run()
    assert sel.budget == 800
    assert sel.tokens_used == 320
    shape = [
        (
            s.slice.document_id,
            s.slice.size,
            s.slice.codepoint_offset,
            s.action,
            s.provenance,
            s.levels,
        )
        for s in sel.slices
    ]
    # Documents ordered by best class rank: hb (eff_rank 0) precedes faq (eff_rank 4).
    assert shape == [
        ("hb", 128, 0, "traded_up", HitProvenance.RETRIEVED, 1),
        ("faq", 256, 0, "traded_up", HitProvenance.TRADED_UP, 2),
    ]


def test_effective_ranks_and_ordering() -> None:
    sel = _run()
    assert [(s.slice.document_id, s.effective_rank) for s in sel.slices] == [("hb", 0), ("faq", 4)]


def test_trade_up_trace() -> None:
    sel = _run()
    trades = [
        (
            t.round,
            t.from_ref.size,
            t.to_ref.size,
            t.to_provenance,
            len(t.subsumed_ids),
            t.delta_tokens,
            t.budget_left,
        )
        for t in sel.trace.trade_ups
    ]
    assert trades == [
        (1, 64, 128, "retrieved", 1, 20, 620),  # hb: two 64s collapse into 128
        (1, 64, 128, "fetched", 0, 50, 570),  # faq: 64 -> 128
        (2, 128, 256, "fetched", 0, 90, 480),  # faq: 128 -> 256
    ]


def test_rejections() -> None:
    sel = _run()
    rejections = [(r.round, r.reason, r.target.size, r.delta_tokens) for r in sel.trace.rejections]
    # Exactly one over-budget (recorded once thanks to parking) and one saturation.
    assert rejections == [
        (2, "over_budget", 256, 580),
        (3, "saturated", 256, None),
    ]


def test_dedup_and_final_trace() -> None:
    sel = _run()
    assert len(sel.trace.dedup) == 2  # C1 and C2 fold in ancestors; faq leaf has none
    assert sel.trace.final.tokens_used == 320
    assert sel.trace.final.budget == 800
    assert sel.trace.final.utilization == 320 / 800
    assert sel.trace.final.budget_exhausted is True


def test_replay_invariant() -> None:
    sel = _run()
    assert sel.trace.replay_top_ids() == {s.id for s in sel.slices}


def test_hb_slice_replaced_both_leaves() -> None:
    sel = _run()
    hb = next(s for s in sel.slices if s.slice.document_id == "hb")
    replaced_ids = set(hb.replaced_ids)
    assert str(HB.slice_at(64, 0).ref.uuid()) in replaced_ids
    assert str(HB.slice_at(64, 64).ref.uuid()) in replaced_ids
    # Evidence carries the genuine retrieved scores of the folded-in slices.
    assert {e.size for e in hb.evidence} == {64, 128, 256}
