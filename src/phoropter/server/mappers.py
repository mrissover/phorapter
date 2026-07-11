"""Hand-written conversions between core types and the generated REST DTOs.

The DTOs in :mod:`phoropter.server.schemas` are generated from
``api/openapi.yaml`` and must never be hand-edited; every non-trivial shaping of
a core value into a DTO (and back) lives here instead. Keeping this seam explicit
is what lets the contract, the generated models, and the engine's real output
shapes evolve independently and be checked against each other in CI.

The query mapping is faithful: it copies the engine's
:class:`~phoropter.selection.Selection` into the response verbatim, never
re-sorting or re-deriving anything, so the engine's determinism and safety
guarantees survive the trip to JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phoropter.model import HitProvenance
from phoropter.server import schemas as sc

if TYPE_CHECKING:
    from phoropter.selection import EvidenceItem, SelectedSlice, Selection
    from phoropter.service.corpora import CorpusInfo
    from phoropter.service.query import QueryOutcome
    from phoropter.stores import DocumentPage, DocumentRecord
    from phoropter.trace import SubstitutionTrace

__all__ = [
    "corpus_info_to_dto",
    "document_page_to_dto",
    "document_record_to_dto",
    "metadata_from_dto",
    "query_outcome_to_dto",
]


# ── corpora ──────────────────────────────────────────────────────────────────


def corpus_info_to_dto(info: CorpusInfo) -> sc.Corpus:
    """Map a :class:`~phoropter.service.corpora.CorpusInfo` to the ``Corpus`` DTO."""
    cfg = info.config
    return sc.Corpus(
        name=cfg.name,
        grid=sc.GridSpec(sizes=list(cfg.grid.sizes)),
        embedder=cfg.embedder_fingerprint,
        dimension=cfg.dimension,
        tokenizer=cfg.token_counter_id,
        document_count=info.document_count,
        points_by_size=[sc.SizeCount(size=s, count=c) for s, c in info.points_by_size],
        degraded=list(info.degraded),
    )


# ── documents ────────────────────────────────────────────────────────────────


def metadata_from_dto(
    entries: list[sc.DocumentMetadataEntry] | None,
) -> tuple[tuple[str, str], ...]:
    """Flatten metadata DTO entries to the core's ordered pair tuple."""
    if not entries:
        return ()
    return tuple((e.key, e.value) for e in entries)


def document_record_to_dto(record: DocumentRecord) -> sc.DocumentRecord:
    """Map a :class:`~phoropter.stores.DocumentRecord` to its DTO."""
    return sc.DocumentRecord(
        document_id=record.document_id,
        codepoint_length=record.codepoint_length,
        byte_length=record.byte_length,
        slice_count=record.slice_count,
        content_marker=record.content_marker,
        metadata=[sc.DocumentMetadataEntry(key=k, value=v) for k, v in record.metadata],
    )


def document_page_to_dto(page: DocumentPage) -> sc.DocumentPage:
    """Map a :class:`~phoropter.stores.DocumentPage` to its DTO."""
    return sc.DocumentPage(
        records=[document_record_to_dto(r) for r in page.records],
        next_cursor=page.next_cursor,
    )


# ── query ────────────────────────────────────────────────────────────────────

_PROVENANCE = {
    HitProvenance.RETRIEVED: sc.Provenance.retrieved,
    HitProvenance.TRADED_UP: sc.Provenance.traded_up,
}


def _evidence_to_dto(item: EvidenceItem) -> sc.EvidenceItem:
    return sc.EvidenceItem(
        id=item.id,
        size=item.size,
        codepoint_offset=item.codepoint_offset,
        score=item.raw_score,
        rank_in_size=item.rank_in_size,
        fused_rank=item.fused_rank,
    )


def _result_to_dto(sel: SelectedSlice, *, include_text: bool) -> sc.QueryResult:
    s = sel.slice
    # The best (lowest) fused rank across evidence is exactly the class's
    # effective rank; when this slice was itself a hit, rank/score are genuine.
    own = next((e for e in sel.evidence if e.ref == s.ref), None)
    retrieval = sc.RetrievalInfo(
        retrieved=own is not None,
        rank_in_size=own.rank_in_size if own is not None else None,
        score=own.raw_score if own is not None else None,
    )
    selection = sc.SelectionInfo(
        action=sc.Action(sel.action),
        replaced=list(sel.replaced_ids),
        levels=sel.levels,
    )
    return sc.QueryResult(
        id=sel.id,
        coords=sc.SliceCoords(
            document_id=s.document_id,
            size=s.size,
            codepoint_offset=s.codepoint_offset,
            codepoint_length=s.codepoint_length,
            codepoint_end=s.codepoint_end,
            own_marker=s.own_marker,
        ),
        text=s.text if include_text else None,
        token_count=s.token_count,
        provenance=_PROVENANCE[sel.provenance],
        effective_rank=sel.effective_rank,
        contiguous_with_next=sel.contiguous_with_next,
        retrieval=retrieval,
        selection=selection,
        evidence=[_evidence_to_dto(e) for e in sel.evidence],
    )


def _trace_to_dto(trace: SubstitutionTrace) -> sc.SubstitutionTrace:
    ip = trace.initial_pack
    return sc.SubstitutionTrace(
        fusion=sc.FusionTrace(
            sizes=list(trace.fusion.sizes), candidate_count=trace.fusion.candidate_count
        ),
        forest=sc.ForestTrace(
            hit_count=trace.forest.hit_count,
            edge_count=trace.forest.edge_count,
            anomaly_count=trace.forest.anomaly_count,
            participation_rate=trace.forest.participation_rate,
            max_depth=trace.forest.max_depth,
        ),
        dedup=[
            sc.DedupEntry(kept_id=d.kept_id, dropped_ids=list(d.dropped_ids)) for d in trace.dedup
        ],
        initial_pack=sc.InitialPackTrace(
            selected_ids=list(ip.selected_ids),
            skipped_unaffordable=[
                sc.SkippedUnaffordable(id=s.id, tokens=s.tokens, budget_left=s.budget_left)
                for s in ip.skipped_unaffordable
            ],
            smallest_unaffordable_tokens=ip.smallest_unaffordable_tokens,
        ),
        trade_ups=[
            sc.TradeUpEntry(
                round=t.round,
                from_id=t.from_id,
                to_id=t.to_id,
                to_provenance=sc.ToProvenance(t.to_provenance),
                subsumed_ids=list(t.subsumed_ids),
                delta_tokens=t.delta_tokens,
                budget_left=t.budget_left,
            )
            for t in trace.trade_ups
        ],
        rejections=[
            sc.RejectionEntry(
                round=r.round,
                target_id=str(r.target.uuid()),
                reason=sc.Reason(r.reason),
                delta_tokens=r.delta_tokens,
            )
            for r in trace.rejections
        ],
        final=sc.FinalTrace(
            tokens_used=trace.final.tokens_used,
            budget=trace.final.budget,
            utilization=trace.final.utilization,
            budget_exhausted=trace.final.budget_exhausted,
        ),
        warnings=list(trace.warnings),
    )


def query_outcome_to_dto(outcome: QueryOutcome) -> sc.QueryResponse:
    """Map a :class:`~phoropter.service.query.QueryOutcome` to the ``QueryResponse`` DTO.

    The engine's selection is copied faithfully — result order, provenance,
    evidence, and trace are taken exactly as produced.
    """
    selection: Selection = outcome.selection
    return sc.QueryResponse(
        corpus=outcome.corpus,
        strategy=outcome.strategy,
        budget=sc.BudgetReport(
            limit=outcome.budget_limit,
            used=selection.tokens_used,
            counter=outcome.counter_id,
        ),
        partial=outcome.partial,
        fanout=[
            sc.FanoutStatus(size=f.size, ok=f.ok, hits=f.hits, error=f.error)
            for f in outcome.fanout
        ],
        results=[
            _result_to_dto(sel, include_text=outcome.include_text) for sel in selection.slices
        ],
        trace=_trace_to_dto(selection.trace) if outcome.include_trace else None,
        warnings=list(outcome.warnings),
    )
