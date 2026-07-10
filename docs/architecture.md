# Architecture

> Skeleton — sections marked *(planned)* are filled in as their milestones land.

Phorapter is one package with two faces:

- **Core library** (`pip install phorapter`) — pure, synchronous, stdlib-only
  (sole runtime dependency: tiktoken, imported lazily). Embeddable in any
  pipeline in-process.
- **Server** (`pip install "phorapter[server,qdrant]"`) — an async service layer
  with two thin skins, REST (FastAPI) and MCP (FastMCP), over pluggable vector
  stores and embedders.

```
                pip install phorapter                      pip install "phorapter[server]"
 ┌─────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ CORE (stdlib-only, sync, I/O-free)          │   │ SERVER                               │
 │  grid.py      GridSpec (validated,          │   │  FastAPI /v1 REST ──┐                │
 │               fingerprint)                  │   │  FastMCP tools ─────┤ zero business  │
 │  slicer.py    multi_view_slice()            │   │                     │ logic in skins │
 │  markers.py   sha256(UTF-8), no-normalize   │   │        ServiceCore (async)           │
 │  ids.py       UUIDv5(PHORAPTER_NAMESPACE)   │   │   CorpusService / DocumentService /  │
 │  forest.py    containment forest            │   │   QueryService                       │
 │  fusion.py    cross-size rank fusion        │   │        │                 │           │
 │  selection.py SelectionStrategy SPI +       │◄──┤  VectorStoreAdapter   Embedder SPI   │
 │               GreedyUpwardStrategy, trace   │   │  (Qdrant default,     (Ollama,       │
 │  tokens.py    TokenCounter SPI (tiktoken)   │   │   InMemory, plugins)   OpenAI-compat)│
 │  stores/memory.py  zero-dep test double     │   │                                      │
 └─────────────────────────────────────────────┘   └──────────────────────────────────────┘
```

## Module map

| Module | Responsibility | Status |
|---|---|---|
| `grid` | `GridSpec`: validated size ladder, closed-form parent arithmetic, fingerprint | ✔ |
| `model` | Frozen value types: `SliceRef`, `Slice`, `SlicedDocument`, `RetrievedHit`, `CandidateHit` | ✔ |
| `markers` | SHA-256 content markers, no-normalization policy | ✔ |
| `ids` | Deterministic UUIDv5 slice identity | ✔ |
| `slicer` | `multi_view_slice()`: encode-once multi-size slicing with descendants | ✔ |
| `tokens` | `TokenCounter` protocol, tiktoken default, registry | ✔ |
| `forest` | Containment forest over retrieved hits | ✔ |
| `fusion` | Cross-size rank fusion (tier interleave) | ✔ |
| `selection` | `SelectionStrategy` SPI, greedy upward trade-up, substitution trace | ✔ |
| `stores` | `VectorStoreAdapter` SPI, in-memory + Qdrant adapters | ✔ |
| `embed` | `Embedder` SPI, Ollama + OpenAI-compatible providers | ✔ |
| `service` / `server` | Async orchestration; REST + MCP skins | ✔ |
| `eval` | Offline forest/budget metrics and regression gating | ✔ |

## Design rules

- **Core purity.** Core modules import only the standard library (tiktoken lazily
  in `tokens`). An import-linter contract forbids framework imports mechanically;
  the tiktoken laziness itself is enforced by a stripped-environment test that
  imports the package with tiktoken masked.
- **Validation at trust boundaries.** Core types are frozen slotted dataclasses
  with no per-instance validation — they are created in bulk. Pydantic validation
  lives exclusively in the server DTO layer.
- **All I/O in the service layer.** The selection engine is synchronous and pure;
  the service layer prefetches everything the engine may need (including
  trade-up ancestor slices, fetched by deterministic ID) before invoking it.
  Purity buys byte-identical determinism and trivial testability.
- **One config source.** Everything that governs slicing/indexing/query for a
  corpus is frozen into a persisted `CorpusConfig` at corpus creation. Server
  defaults apply exactly once, at creation — never silently at read time.
- **No fabricated scores.** Store scores are ordinal within one
  (corpus, size, query) triple only. Nothing in the system compares raw scores
  across sizes or synthesizes a "global" score.

## The query pipeline (specified in [engine.md](engine.md))

```
embed once → fan out per size (parallel, partial-failure tolerant)
  → tier-interleave fusion → containment forest (closed-form, minimal edges)
  → prefetch ancestor closure by deterministic ID
  → selection strategy (sync, pure): dedup → pack → trade up under budget
  → ordered slices + substitution trace
```

## Scope and non-goals (v1)

- **No downward selection.** When a large slice is relevant, choosing *which part*
  of it to keep under a tight budget requires a semantic signal that structure
  alone cannot provide. The `SelectionStrategy` interface is the plug-point for
  future strategies; v1 ships only moves that are safe by construction.
- **One slicing scheme.** Origin-aligned multi-size slicing only — no recursive,
  semantic, or content-defined chunking. Different chunkers produce misaligned
  boundaries that break exact containment.
- **No baked-in embedding model** and **no proprietary store coupling** — both are
  SPIs; Qdrant and Ollama/OpenAI-compatible adapters are conveniences.
- **No Unicode normalization**, anywhere (see [concepts.md](concepts.md)).
- **No partial document edits or version history** — add/replace/delete, whole
  documents.
- **No auth/multi-tenancy machinery** beyond an optional static bearer token.
- **No answer-quality evaluation** — the eval harness measures structure and
  budget utilization, not generation quality.
