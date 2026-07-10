"""Cross-size rank fusion.

Retrieval returns a separate top-k list per grid size. Their raw scores are
**not comparable across sizes** — a 64-code-point slice and a 1024-code-point
slice embed into vectors whose cosine similarity is distorted by the length
difference, independent of content. So fusion trusts only the one thing that is
safe: the *within-size ordering*.

:class:`TierInterleave` (the default) does exactly that. It interleaves the
per-size lists by rank: the best hit of every size, then the second-best of every
size, and so on. No score is ever compared to a score from another size, and
there are no tuned constants. A size that returned nothing simply contributes no
ranks — degradation is automatic.

:class:`RawScorePool` is provided for comparison only; it pools every hit and
sorts by raw score, which is exactly the cross-size comparison the length
asymmetry makes unreliable. It is not the default and should not be.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from phorapter.model import CandidateHit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from phorapter.model import RetrievedHit

__all__ = ["RankFusion", "RawScorePool", "TierInterleave"]


def _score_key(hit: RetrievedHit) -> tuple[float, str]:
    """Sort key within one size: score descending, ties broken by point id ascending."""
    score = hit.score if hit.score is not None else float("-inf")
    return (-score, str(hit.ref.uuid()))


def _placement_key(hit: RetrievedHit) -> tuple[int, str, int]:
    """Deterministic within-tier / pooled tiebreak: size, then document, then offset."""
    return (hit.slice.size, hit.slice.document_id, hit.slice.codepoint_offset)


@runtime_checkable
class RankFusion(Protocol):
    """Fuses per-size result lists into one ranked candidate list."""

    name: str

    def fuse(self, by_size: Mapping[int, Sequence[RetrievedHit]]) -> list[CandidateHit]: ...


class TierInterleave:
    """Rank-tier interleave: the safe default (see module docstring)."""

    name = "tier_interleave"

    def fuse(self, by_size: Mapping[int, Sequence[RetrievedHit]]) -> list[CandidateHit]:
        ranked = {size: sorted(hits, key=_score_key) for size, hits in by_size.items() if hits}
        if not ranked:
            return []
        depth = max(len(hits) for hits in ranked.values())
        sizes_ascending = sorted(ranked)

        candidates: list[CandidateHit] = []
        for tier in range(depth):
            tier_hits = [ranked[size][tier] for size in sizes_ascending if tier < len(ranked[size])]
            tier_hits.sort(key=_placement_key)
            for hit in tier_hits:
                candidates.append(CandidateHit(hit=hit, fused_rank=len(candidates)))
        return candidates


class RawScorePool:
    """Pool every hit and sort by raw score. Experimental; unsafe across sizes.

    Shipped only so the length-asymmetry effect can be measured against the
    default. Do not use it as a production fusion.
    """

    name = "raw_score_pool"

    def fuse(self, by_size: Mapping[int, Sequence[RetrievedHit]]) -> list[CandidateHit]:
        pooled = [hit for hits in by_size.values() for hit in hits]
        pooled.sort(key=lambda h: (_score_key(h)[0], _placement_key(h)))
        return [CandidateHit(hit=hit, fused_rank=i) for i, hit in enumerate(pooled)]
