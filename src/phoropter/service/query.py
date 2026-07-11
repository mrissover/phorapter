"""Query service: multi-size retrieval fan-out and budgeted right-sizing.

The query path is the one place async I/O meets the synchronous, pure selection
engine. It:

1. embeds the query once;
2. fans out one top-k search per grid size, concurrently, each with its own
   timeout, tolerating partial failure — if at least one size answers the query
   succeeds with ``partial=True``; only a total fan-out failure raises;
3. fuses the per-size lists (tier interleave) and builds the containment forest;
4. prefetches the ancestor closure of the packable candidates by deterministic
   id (one ``fetch_by_ids`` per size) and wraps it in a synchronous slice source;
5. runs the selection strategy and returns its :class:`~phoropter.selection.Selection`
   unmodified, alongside the fan-out status and budget report.

The engine's output is never re-sorted or mutated — its determinism and safety
guarantees are preserved end to end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from phoropter.errors import StoreError
from phoropter.forest import ContainmentForest
from phoropter.fusion import TierInterleave
from phoropter.selection import (
    GreedyUpwardStrategy,
    Selection,
    SelectionOptions,
    SelectionRequest,
)
from phoropter.tokens import get_counter

if TYPE_CHECKING:
    import uuid

    from phoropter.model import RetrievedHit
    from phoropter.selection import SliceSource
    from phoropter.service.core import ServiceCore
    from phoropter.stores import CorpusConfig

__all__ = ["FanoutSize", "QueryOutcome", "QueryService"]


@dataclass(frozen=True, slots=True)
class FanoutSize:
    """The outcome of one size's fan-out search."""

    size: int
    ok: bool
    hits: int
    error: str | None


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """Everything the surfaces need to render a query response.

    ``selection`` is the engine's result, verbatim. ``partial`` is True when at
    least one size failed but the query still succeeded. ``budget_limit`` is the
    effective (capped) budget, or ``None`` for dedup-only.
    """

    corpus: str
    strategy: str
    selection: Selection
    partial: bool
    fanout: tuple[FanoutSize, ...]
    budget_limit: int | None
    counter_id: str
    include_text: bool
    include_trace: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


class QueryService:
    """Run a multi-size, budget-fitted query against one corpus."""

    def __init__(self, core: ServiceCore) -> None:
        self._core = core

    async def run(
        self,
        corpus: str,
        query: str,
        *,
        token_budget: int | None = None,
        top_k_per_size: int = 10,
        strategy: str = "greedy_upward",
        expansion: str = "fill",
        sizes: tuple[int, ...] | None = None,
        tokenizer: str | None = None,
        max_slice_size: int | None = None,
        include_text: bool = True,
        include_trace: bool = True,
    ) -> QueryOutcome:
        """Retrieve across sizes and right-size under the budget.

        Raises CORPUS_NOT_FOUND for an unknown corpus, UNKNOWN_TOKENIZER for a
        bad tokenizer override, EMBEDDER_UNAVAILABLE if the query cannot be
        embedded, and STORE_UNAVAILABLE only when *every* size fan-out fails.
        """
        core = self._core
        store = core.store
        config = await store.get_corpus_meta(corpus)  # CORPUS_NOT_FOUND if absent

        warnings: list[str] = []
        counter_id, trust_stored = self._resolve_counter(config, tokenizer)
        counter = get_counter(counter_id)  # UNKNOWN_TOKENIZER on a bad override

        budget_limit = self._cap_budget(token_budget, warnings)
        target_sizes = self._resolve_sizes(config, sizes, warnings)

        # 1. embed the query once (EMBEDDER_UNAVAILABLE on failure).
        vectors = await core.embedder.embed([query])
        query_vector = vectors[0]

        # 2. fan out one search per size, concurrently, tolerating partial failure.
        by_size, fanout = await self._fanout(corpus, target_sizes, query_vector, top_k_per_size)
        if not any(f.ok for f in fanout):
            raise StoreError(
                f"every size fan-out failed for corpus {corpus!r}: "
                + "; ".join(f"s{f.size}: {f.error}" for f in fanout)
            )

        all_hits: list[RetrievedHit] = [hit for hits in by_size.values() for hit in hits]

        # 3. fuse + forest.
        candidates = tuple(TierInterleave().fuse(by_size))
        forest = ContainmentForest.build(all_hits, config.grid)

        # 4. prefetch the ancestor closure of the packable candidates by id.
        source = await self._prefetch_ancestors(corpus, config, all_hits)

        # 5. run the strategy, unmodified.
        options = SelectionOptions(
            expansion=expansion,  # type: ignore[arg-type]  # validated at DTO boundary
            max_slice_size=max_slice_size,
        )
        request = SelectionRequest(
            candidates=candidates,
            forest=forest,
            budget=budget_limit,
            grid=config.grid,
            counter=counter,
            source=source,
            options=options,
            corpus=corpus,
            trust_stored_counts=trust_stored,
        )
        engine = GreedyUpwardStrategy()  # the only v1 strategy; name pinned below
        selection = engine.select(request)

        if selection.trace.final.budget_exhausted and not selection.slices:
            warnings.append("token budget too small to fit any slice; returned empty")

        return QueryOutcome(
            corpus=corpus,
            strategy=strategy if strategy else engine.name,
            selection=selection,
            partial=any(not f.ok for f in fanout),
            fanout=fanout,
            budget_limit=budget_limit,
            counter_id=counter_id,
            include_text=include_text,
            include_trace=include_trace,
            warnings=tuple(warnings),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _resolve_counter(self, config: CorpusConfig, tokenizer: str | None) -> tuple[str, bool]:
        """Pick the counter id; stored per-slice counts are trusted only under the pin."""
        if tokenizer is None or tokenizer == config.token_counter_id:
            return config.token_counter_id, True
        return tokenizer, False

    def _cap_budget(self, token_budget: int | None, warnings: list[str]) -> int | None:
        if token_budget is None:
            return None
        cap = self._core.settings.limits.max_token_budget
        if token_budget > cap:
            warnings.append(f"token_budget {token_budget} capped to server maximum {cap}")
            return cap
        return token_budget

    def _resolve_sizes(
        self, config: CorpusConfig, sizes: tuple[int, ...] | None, warnings: list[str]
    ) -> tuple[int, ...]:
        if sizes is None:
            return config.grid.sizes
        grid_sizes = set(config.grid.sizes)
        kept = tuple(s for s in config.grid.sizes if s in set(sizes))
        unknown = sorted(set(sizes) - grid_sizes)
        if unknown:
            warnings.append(f"ignoring sizes not in the corpus grid: {unknown}")
        return kept if kept else config.grid.sizes

    async def _fanout(
        self,
        corpus: str,
        sizes: tuple[int, ...],
        vector: list[float],
        k: int,
    ) -> tuple[dict[int, list[RetrievedHit]], tuple[FanoutSize, ...]]:
        """Search every size concurrently; per-size failures are captured, not raised."""
        timeout = self._core.settings.store.search_timeout_s

        async def one(size: int) -> list[RetrievedHit]:
            return await self._core.store.search_size(corpus, size, vector, k, timeout_s=timeout)

        results = await asyncio.gather(
            *(asyncio.wait_for(one(size), timeout + 1.0) for size in sizes),
            return_exceptions=True,
        )
        by_size: dict[int, list[RetrievedHit]] = {}
        fanout: list[FanoutSize] = []
        for size, result in zip(sizes, results, strict=True):
            if isinstance(result, BaseException):
                fanout.append(FanoutSize(size=size, ok=False, hits=0, error=str(result)))
            else:
                by_size[size] = result
                fanout.append(FanoutSize(size=size, ok=True, hits=len(result), error=None))
        return by_size, tuple(fanout)

    async def _prefetch_ancestors(
        self,
        corpus: str,
        config: CorpusConfig,
        hits: list[RetrievedHit],
    ) -> SliceSource:
        """Prefetch every grid ancestor of every hit by deterministic id, per size.

        The prefetched universe is exactly the candidate set trade-up may reach:
        the retrieved hits themselves plus, for each, every larger enclosing grid
        slice. Fetch misses are simply absent — the engine records them and moves on.
        Failures here never fail the request.
        """
        from phoropter.model import Slice, SliceRef
        from phoropter.service.core import PrefetchSource

        # Refs already present as retrieved hits do not need fetching.
        retrieved_refs = {hit.ref for hit in hits}
        wanted_by_size: dict[int, set[uuid.UUID]] = {size: set() for size in config.grid.sizes}
        for hit in hits:
            for ancestor in hit.ref.ancestors(config.grid):
                if ancestor in retrieved_refs:
                    continue
                wanted_by_size[ancestor.size].add(ancestor.uuid())

        materialized: dict[SliceRef, Slice] = {}
        for size, ids in wanted_by_size.items():
            if not ids:
                continue
            try:
                points = await self._core.store.fetch_by_ids(corpus, size, sorted(ids))
            except Exception:  # a failed prefetch never fails the request
                continue
            for point in points:
                materialized[point.slice.ref] = point.slice
        return PrefetchSource(materialized)
