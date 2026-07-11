# Concepts

This document is the normative reference for Phoropter's core mechanics: the
slicing grid, structural markers, and deterministic slice identity. The
query-time machinery built on top of them (containment forest, dedup, trade-up)
is specified in [engine.md](engine.md).

## Multi-view, origin-aligned slicing

Phoropter indexes every document at **multiple slice sizes** simultaneously — the
default grid is 64, 128, 256, 512, and 1024 **Unicode code points**. Every size
tiles the document from code point 0 with stride equal to the size (no overlap);
the final slice of each size may be shorter. Concatenating any one size's slices
in order reproduces the document exactly, in both code points and bytes.

Two rules make the grid special:

1. **Origin alignment.** A slice of size `S` always starts at a code-point offset
   that is a multiple of `S`.
2. **Divisibility chain.** Each grid size divides every larger grid size.
   (The default grid uses powers of two, but nothing requires that — `(3, 6, 24)`
   is equally valid.)

Together these guarantee the **laminar family** property: *any two grid slices of
one document are either nested or disjoint.* Consequences, all exact and all by
construction:

- Containment over any set of slices forms a **forest** (no partial overlaps).
- A slice's ancestors are totally ordered — it has exactly one enclosing slice
  per larger size, whose offset is computable in closed form:
  `parent_offset = (offset // parent_size) * parent_size`.
- The smaller-size slices contained in a slice covering `[O, E)` at size `S'`
  are exactly those at offsets `range(O, E, S')` — an enumeration, not a search.

**Why code points, not bytes or tokens?** A byte grid would cut multi-byte UTF-8
characters in half, corrupting the text handed to embedding models. A token grid
would tie slice identity to one tokenizer's version. Code points are stable,
language-neutral, and natively how Python indexes strings. (Slices freely cut
across words, sentences, and even grapheme clusters — boundary quality is traded
for exactness, deliberately.)

## Structural markers

A slice's **marker** is the SHA-256 hash of the exact UTF-8 bytes of its text,
as 64 lowercase hex characters. Each slice also carries **descendant markers**:
the markers of every strictly-smaller grid slice fully contained in its span, in
canonical order (child size ascending, then offset ascending).

Markers make containment *verifiable from content alone*: a slice's marker
appearing in another slice's descendant list attests that its bytes are literally
a sub-span of the larger slice's bytes. This is what licenses the engine's core
move — **upward substitution**, replacing a slice with its enclosing parent, is
information-preserving *by construction*: the child's bytes are physically inside
the parent. Similarity scores cannot license that move: similarity is symmetric,
so it cannot even express which of two slices contains the other, at any
threshold.

### The no-normalization policy (normative)

Markers are computed over the bytes **as given**: no Unicode normalization
(NFC/NFD), no newline canonicalization, no BOM stripping — at any layer, ever.

1. **Reproducibility.** Any language that can UTF-8-encode a string computes the
   identical marker; normalization behavior forks across libraries and versions.
2. **Lossless reconstruction.** Slice texts must concatenate back to the exact
   document. A normalizing hash would claim equality between byte-different texts.
3. **Exactness.** "Equal markers ⇒ byte-identical content" is the load-bearing
   guarantee. Normalized equality is a different, weaker claim.

Callers who want normalized text normalize *before* ingest and own that choice.
Text that cannot be UTF-8-encoded (lone surrogates) is rejected at slicing time:
such text has no well-defined markers.

### Identity vs content

A slice's **identity** is its coordinates: `(document_id, size, codepoint_offset)`.
Markers are **content attestation**, not identity — two slices at different
coordinates can carry identical text and identical markers while remaining
different slices. Query-time containment therefore decides *positionally* (grid
arithmetic over coordinates) and uses markers as an integrity guard; repeated
text can never fabricate a containment edge. See [engine.md](engine.md).

## The prefix property (gating invariant)

For any document, the **left-anchored spine** — the offset-0 slice of every grid
size — must nest as byte-prefixes: the 64-slice's UTF-8 bytes are a prefix of
the 128-slice's bytes, and so on up the grid, with each smaller marker present
in every larger slice's descendant list. The prefix is proper whenever the
larger slice's span extends further; for documents shorter than a grid size,
consecutive spine levels cover the same span and are byte-identical with
identical markers (the engine treats such equal-span levels as degenerate and
skips them during trade-up). A violation is structurally impossible
if slicing is correct, so a detected violation means a bug in grid alignment,
code-point-to-byte mapping, or descendant assignment — and nothing downstream can
be trusted until it is fixed.

The prefix property is enforced by gate tests (`tests/test_prefix_property.py`)
that run first in CI and block everything else: the four canonical corpora
(ASCII, CJK, emoji, multi-script) plus property-based generalization to arbitrary
text and arbitrary valid grids. A second gate (`tests/test_parity.py`) asserts
byte-for-byte parity with fixtures generated by the reference implementation.

## Deterministic slice identity

Every slice has a deterministic UUIDv5 under the project namespace
(`PHOROPTER_NAMESPACE`, itself derived as
`uuid5(NAMESPACE_DNS, "phoropter.impluvium")` and pinned by test). The
name string is `escaped(document_id)|size|offset` (with `corpus` prefixed when an
adapter co-mingles corpora in one collection); `\` and `|` in free-text fields
are escaped, making the mapping injective for arbitrary document IDs.

Determinism is what makes the document lifecycle simple and safe:

- **Replace** re-slices the new text; unchanged grid cells re-derive identical
  UUIDs and upsert in place.
- **Shrink** leaves orphans that are an exact set difference (old IDs minus new
  IDs) — precise tombstoning, no scanning.
- The write order is **upsert new, then tombstone orphans**. A query racing a
  replace may briefly see slices from both generations; the engine's marker
  integrity guard prevents any cross-generation substitution during that window
  (a stale parent's descendant list will not contain the new child's marker, and
  vice versa).

## Token counting

Budgets arrive in tokens; slices are measured in code points. A `TokenCounter`
(identified by a stable id such as `tiktoken:o200k_base`) bridges the two. The
counter is pinned per corpus at creation time; per-slice token counts are
computed at ingest and stored, so budget arithmetic at query time is lookups,
not tokenization. A request may specify a different counter, in which case
counts are computed on the fly (memoized per request); counts from different
counters are never mixed.
