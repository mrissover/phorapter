# The right-sizing engine

This is the normative specification of Phoropter's query-time engine: how a query
and a token budget become a right-sized set of context slices, with a full trace
of every decision. Everything here is deterministic and pure — given the same
inputs, the same output and the same trace, byte for byte.

## Pipeline

```
per-size result lists                                (from the store, one top-k per grid size)
      │
      ▼  fusion (TierInterleave)          cross-size ranking without comparing scores
   fused candidates
      │
      ▼  containment forest               positional edges, marker-verified
   forest + candidates
      │
      ▼  selection (GreedyUpwardStrategy)  dedup → pack → trade up, under budget
   selected slices + substitution trace
```

The service layer performs all I/O — it fans out retrieval, builds the forest,
and **prefetches the ancestor closure** of the packable candidates by
deterministic ID — then hands a fully-populated `SelectionRequest` to the engine.
The engine never touches the network.

## 1. Fusion: ranking across sizes

Retrieval returns one top-k list per grid size. Their scores are **not comparable
across sizes**: a short slice and a long slice from the same region embed into
vectors whose similarity is distorted by the length difference, independent of
content. So fusion trusts only the *within-size ordering*.

`TierInterleave` (the default) interleaves the per-size lists by rank: the best
of every size, then the second-best of every size, and so on. Within a tier,
slices are ordered by size ascending (then document, then offset, for a total
order). A size that returned nothing contributes no ranks — degradation is
automatic. The result is a list of candidates each stamped with a `fused_rank`.
No score is ever compared to a score from another size, and there are no tuned
constants.

`RawScorePool` is provided for research comparison only; it pools all hits and
sorts by raw score, which is exactly the unreliable cross-size comparison. It is
not the default.

## 2. The containment forest

See [concepts.md](concepts.md) for the structure. The engine uses it to answer
two questions: which retrieved slices nest inside which (for dedup), and what the
enclosing slices of a kept slice are (for trade-up). Containment is decided
positionally and verified by markers; a marker mismatch is a
`ContainmentAnomaly` (a cross-generation read), never an edge.

## 3. Selection: `GreedyUpwardStrategy`

### (a) Dedup — keep the leaf

For each forest **leaf** (a retrieved slice with nothing retrieved inside it),
form a *class*:

- `top` = the leaf slice (this is what may grow via trade-up),
- `evidence` = the leaf plus its retrieved ancestors (the enclosing retrieved
  slices), each carrying its genuine per-size score,
- `effective_rank` = the best (lowest) `fused_rank` across the evidence.

Retrieved ancestors are thus *folded into evidence* rather than kept as separate
results: their content is a superset of the leaf's, so keeping the leaf loses
nothing, and their ranking signal survives through the effective rank. A dropped
ancestor may later reappear as a trade-up target.

### (b) Initial pack — first-fit

Visit classes in priority order `(effective_rank, document_id, offset)`. Pack
each whose cost (token count, plus optional join overhead) fits the remaining
budget; skip the rest, recording the shortfall, and keep walking (a cheaper
lower-ranked class may still fit). Slice text is never truncated — a partial
slice has no marker identity. If nothing fits, the selection is empty and the
trace says so.

With no budget (`budget=None`), this degenerates to dedup only: every class is
returned in priority order, no packing, no trade-up.

### (c) Trade-up — round-robin, one level per class per round

While the budget allows, grow packed classes upward. Each round visits the alive
classes in priority order; each attempts **one content level** of trade-up:

1. **Next target.** Walk the grid sizes above the current top, skipping
   *degenerate* levels whose clipped span equals the current one (for a document
   shorter than a grid size, the larger slices are byte-identical no-ops). Honor
   `max_slice_size` and a hard bound of one level per grid step. No target →
   the class is *saturated*.
2. **Resolve the parent** from the prefetched source. If it is a retrieved slice,
   its provenance is `retrieved` and its genuine score joins the evidence; if the
   source supplies an un-retrieved ancestor, provenance is `fetched`. A missing
   ref is a `fetch_miss` — the class keeps its top and skips that level.
3. **Staleness guard (mandatory).** The current top's marker must appear in the
   parent's descendant markers. If not, it is a `stale_parent` (a read racing a
   replace); the class keeps its top and skips that level.
4. **Subsumption netting.** The parent may enclose the tops of *other* alive
   classes (marker-verified), or coincide with one already sitting on it. All
   such classes are absorbed in this one move, and the cost is netted:

   ```
   delta = cost(parent) − cost(top) − sum(cost(absorbed tops))
   ```

   Accept iff `delta ≤ budget_left`. On accept, the class's top becomes the
   parent, the absorbed classes' evidence merges in, and the effective rank takes
   the minimum. On reject (`over_budget`), the class is *parked* at the current
   budget level and retried only if the budget later frees back above it (a
   negative-delta merge elsewhere can free budget) — so a hopeless target is
   recorded once, not every round.

The round-robin stops when a full round makes no progress (no acceptance and no
state change) or every class is saturated. Because every state change is monotone
and bounded, it always terminates.

### (d) Final ordering

Documents are ordered by their best class rank; within a document, slices are in
reading (code-point) order. Adjacent selected slices of the same document are
flagged `contiguous_with_next` so a caller can render them seamlessly. Slices are
never merged — a merged span would have no grid identity, marker, or token count.

### Scores on the output

A selected slice that is a retrieved hit carries that hit's genuine retrieval
score. A slice reached by trading up to an **un-retrieved** parent carries no
retrieval score (`null`) — never a fabricated one — but its `evidence` list
still holds the real, size-scoped scores of everything it stands in for. Rank
flows upward because the evidence physically does; scores never do.

## The substitution trace

Every run returns a `SubstitutionTrace` recording each phase: which sizes fused,
the forest shape, each dedup class, the initial pack (selected and skipped), each
trade-up with its subsumption accounting, each rejection with its reason, and the
final budget arithmetic. The trace is the explainability contract — a caller can
see exactly what the engine did and why.

The lifecycle is machine-checkable: `SubstitutionTrace.replay_top_ids()` replays
the pack and every trade, verifying that each replaced or subsumed id was a live
top when consumed and that no trade introduced a duplicate. The ids it returns
must be exactly the ids of the final selection. This invariant is asserted across
the property-based tests.

## Determinism

Same corpus snapshot, same per-size result lists, same configuration and budget
⇒ byte-identical selection, ordering, and trace. Every iteration runs over an
explicitly sorted sequence; fan-out results are reassembled in size order, never
completion order. The guarantee is conditional on the retrieval inputs, which the
trace captures precisely.

## Extending the engine

`SelectionStrategy` is the extension point. A new strategy — for example, global
budget packing over the laminar candidate tree, rather than greedy per-class —
receives the same `SelectionRequest` (candidates, forest, budget, the prefetched
source) and returns the same `Selection`. The prefetched ancestor closure is
exactly the candidate universe such a strategy needs, so no interface change is
required. Downward selection (choosing which part of a large slice to keep when it
will not fit) is deliberately out of scope: it needs a semantic signal that
structure alone cannot supply.
