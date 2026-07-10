# Design decisions

An append-only log of the load-bearing decisions, with the reasoning that
settled each. Newer decisions may refine older ones; nothing here is silently
reversed.

---

### D-001 — The slicing grid is a validated value type

**Context.** Slice geometry depends on the grid being origin-aligned with each
size dividing the next; violating either breaks exact containment.
**Decision.** `GridSpec` validates both rules in its constructor and is
immutable — an invalid grid cannot exist. Its fingerprint is checked at every
store upsert, so data sliced on one grid can never mix with another.
**Consequence.** The laminar-family guarantee holds by construction everywhere
downstream; there is no module-level size constant to drift out of sync.

### D-002 — Grid units are Unicode code points

**Context.** Boundaries could be measured in bytes, code points, or tokens.
**Decision.** Code points. Bytes would split multi-byte characters; tokens would
tie identity to one tokenizer version.
**Consequence.** Slicing is stable and language-neutral, and Python's native
string indexing *is* code-point indexing. Budgets, which are in tokens, are
bridged separately (see D-011).

### D-003 — Markers are SHA-256 over exact bytes, never normalized

**Context.** Content addressing needs a canonical form.
**Decision.** A marker is SHA-256 of the slice's exact UTF-8 bytes, as given — no
Unicode normalization, newline canonicalization, or BOM handling, at any layer.
**Consequence.** Equal markers mean byte-identical content (the exactness the
whole method rests on), any language computes identical markers, and slice texts
reconstruct the document losslessly. Callers who want normalized text own that
choice before ingest.

### D-004 — Slice identity is coordinates; markers are content

**Context.** Two slices can carry identical text (repeated content, or a
document shorter than a grid size).
**Decision.** A slice's identity is `(document_id, size, codepoint_offset)`.
Markers attest content, they are not identity.
**Consequence.** Containment is decided positionally and merely *verified* by
markers, so repeated text can never fabricate a containment edge.

### D-005 — Deterministic slice IDs under a project namespace

**Context.** Re-ingesting a document must update in place, and orphaned slices of
a shrunken document must be found without scanning.
**Decision.** Each slice's store ID is a UUIDv5 over its coordinates under a
fixed project namespace, with separator-escaped fields so the mapping is
injective for arbitrary document IDs.
**Consequence.** Replace is an in-place upsert; the orphan set after a shrink is
an exact set difference. IDs are intentionally distinct from any other
implementation's — markers, not IDs, are the cross-implementation anchor.

### D-006 — Core types are frozen dataclasses; validation lives at the edges

**Context.** A modest corpus produces hundreds of thousands of slice objects.
**Decision.** Core value types are frozen, slotted dataclasses with no per-object
validation. Rich validation (and any schema framework) lives only at the server's
request boundary.
**Consequence.** The hot path stays cheap, and the core imports and runs in a
stripped environment (its sole runtime dependency, the tokenizer, is imported
lazily). An import contract enforces the purity.

### D-007 — Containment is positional, markers are the integrity guard

**Context.** A retrieved parent could be from a different generation than its
positional child (a read racing a document replace).
**Decision.** The forest links a child to a positional parent only if the child's
marker is in the parent's descendant list; a mismatch is recorded as a
`ContainmentAnomaly` and the child looks further up. Anomalies are never fatal.
**Consequence.** Substitution never crosses generations, and the failure mode
(stale reads) is observable rather than silently wrong.

### D-008 — Minimal-parent edges, not the transitive closure

**Context.** Trade-up wants the *nearest* enclosing slice; reporting all
ancestor-descendant pairs is O(n²) and redundant for that.
**Decision.** The forest stores one minimal parent per node. The transitive
closure is available on request (for reporting) by walking the chains.
**Consequence.** Construction is `O(H·|sizes|)`, and trade-up reads the minimal
parent directly.

### D-009 — Cross-size ranking by rank-tier interleave; scores never fused

**Context.** Retrieval scores from different sizes are distorted by length
asymmetry and are not comparable.
**Decision.** Fusion interleaves the per-size lists by within-size rank
(`TierInterleave`); no score is compared across sizes, and no synthetic
"global" score is ever produced. A traded-up slice with no retrieval score of
its own carries `null`, plus an evidence list of genuine size-scoped scores.
**Consequence.** The one unreliable signal (cross-size score comparison) is never
relied upon; ranking flows upward through evidence, scores do not.

### D-010 — Dedup keeps the leaf; trade-up grows it

**Context.** When a child and an enclosing ancestor are both retrieved, one must
be kept.
**Decision.** Keep the leaf (the smallest), folding retrieved ancestors into its
evidence. Trade-up then grows the kept slice upward when the budget allows.
**Consequence.** Dedup is budget-independent and the whole engine moves in one
monotone direction (smallest, then grow), which keeps it simple and terminating.
A dropped ancestor can re-enter as a trade-up target carrying its real score.

### D-011 — Token counts are pinned per corpus and stored per slice

**Context.** Budgets are in tokens; slices are in code points; tokenizers vary.
**Decision.** A corpus pins one tokenizer at creation. Per-slice counts are
computed at ingest and stored, so budgeting is a lookup. A request may override
the tokenizer, in which case counts are recomputed on the fly (memoized per
request); counts from different tokenizers are never mixed.
**Consequence.** Budget arithmetic is fast and exact under the pinned tokenizer,
and honest (never approximate across tokenizers).

### D-012 — Trade-up is round-robin with subsumption netting and parking

**Context.** Greedily growing one class to the top would starve the others; and a
parent that encloses several kept slices should absorb them in one move.
**Decision.** Round-robin, one content level per class per round. A trade-up
subsumes every alive class the parent encloses, netting their token costs into a
single delta. A rejected (over-budget) class is parked and retried only if the
budget later frees above its parked level.
**Consequence.** Growth is balanced across classes, sibling collapses are charged
once, and the loop records each hopeless target once and terminates.

### D-013 — The engine is synchronous and pure; the service prefetches

**Context.** Trade-up may need ancestor slices that were not retrieved.
**Decision.** The service prefetches the ancestor closure by deterministic ID and
hands it to the engine as a `SliceSource`. The engine performs no I/O.
**Consequence.** Selection is deterministic and trivially testable, and the
prefetched closure is exactly the candidate universe a future global-packing
strategy would need — so new strategies need no interface change.

### D-014 — Store, embedder, and tokenizer are SPIs behind entry points

**Context.** Hard-coding a vector store or embedding model would limit adoption
and bias the method.
**Decision.** Each is a small interface discovered through an entry-point group,
with a shipped default (Qdrant; Ollama/OpenAI-compatible; tiktoken). A store
adapter proves compliance by passing a shared conformance suite.
**Consequence.** The method is embedder- and store-agnostic; third parties add
providers without touching the core.

### D-015 — One collection per (corpus, size) in the default store

**Context.** Retrieval is always top-k *per size*.
**Decision.** The Qdrant adapter uses one collection per (corpus, size), plus a
meta and a document registry. Descendant markers travel packed in the payload;
document facts live once on the registry.
**Consequence.** Each size gets its own vector geometry, a single unhealthy
collection degrades exactly one size (the unit of the partial-failure policy),
and containment stays decidable from a retrieval payload alone.

### D-016 — Core error codes surface unchanged through the API

**Context.** Errors raised deep in the core need a stable, actionable identity at
the API boundary.
**Decision.** Every error carries a stable machine-readable code; the REST error
taxonomy includes the core codes rather than remapping them.
**Consequence.** Clients act on codes, not prose, and the same identity holds
from the core through the wire.
