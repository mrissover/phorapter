# REST API

The REST surface lives under `/v1` and is served by `phoropter serve`. The
normative contract is [`api/openapi.yaml`](../api/openapi.yaml) (OpenAPI 3.1);
this document is the narrative companion — semantics, the error taxonomy, and the
partial-failure policy, with examples. The DTOs the server validates against are
generated from that contract, and two CI gates keep the running app and the
contract equivalent (a spec diff and a Schemathesis property suite).

## Design in one paragraph

Every slicing, indexing, and query decision for a corpus is frozen once, at
corpus creation, into the store's meta storage. The server reads that config back
on every request and holds no per-corpus state of its own — it is stateless and
horizontally scalable. Retrieval fans out across the corpus's grid sizes and
tolerates partial failure; the selection engine that right-sizes the results is
pure and deterministic.

## Authentication

Auth is an optional static bearer token, off by default. When the server is
configured with `server.api_key`, every endpoint except the liveness
(`/healthz`) and readiness (`/v1/health`) probes requires
`Authorization: Bearer <token>`; a missing or wrong token returns `401
UNAUTHORIZED`. Real authentication and authorization are reverse-proxy territory
— the static token exists to keep an unprotected deployment from being wide open,
not to be an identity system.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness. Never touches the store. |
| GET | `/v1/health` | Readiness: store reachable, embedder probed. |
| GET | `/v1/info` | Version, configured store/embedder, registered adapters. |
| GET | `/v1/corpora` | List corpus names. |
| POST | `/v1/corpora` | Create a corpus (grid + embedder + tokenizer frozen here). |
| GET | `/v1/corpora/{corpus}` | Inspect: frozen config, counts, degradation. |
| DELETE | `/v1/corpora/{corpus}` | Drop a corpus and everything under it. |
| GET | `/v1/corpora/{corpus}/documents` | List documents (cursor-paginated). |
| PUT | `/v1/corpora/{corpus}/documents/{id}` | Add or replace one document. |
| POST | `/v1/corpora/{corpus}/documents/batch` | Add or replace a bounded batch. |
| GET | `/v1/corpora/{corpus}/documents/{id}` | Inspect one document record. |
| DELETE | `/v1/corpora/{corpus}/documents/{id}` | Delete a document. |
| POST | `/v1/corpora/{corpus}/query` | Retrieve and right-size under a budget. |

## Creating a corpus

Creation is the only place per-corpus configuration is written. The grid, the
embedder pin (`provider:model`) and its probed vector dimension, and the token
counter are all frozen at this moment and are immutable thereafter. Omitted
fields fall back to the server's configured defaults.

```http
POST /v1/corpora
{ "name": "handbook" }
```

The token counter is *force-materialized* at creation — the server actually
resolves it and runs it once — so a typo'd tokenizer pin can never be frozen into
a corpus; it fails at creation with `422 UNKNOWN_TOKENIZER` instead. Changing the
grid or the embedding model is not an update; it is a new corpus and a reindex.

## Ingesting documents

`PUT .../documents/{id}` adds or replaces a document. It is synchronous and
idempotent: the document is sliced at every grid size, every slice is embedded,
and the points are upserted under deterministic ids, so re-ingesting identical
text is an in-place no-op. A replacement that shrinks the document tombstones
exactly the orphaned slices (the per-size set difference between the old and new
generations) — the upsert happens first, then the tombstone, so a query racing a
replace is protected by the marker staleness guard. Documents larger than
`limits.max_document_codepoints` are refused with `413 DOCUMENT_TOO_LARGE`.

The batch endpoint ingests up to `limits.max_batch_documents` documents and
returns a per-document status, so a partially-failing batch is legible rather
than all-or-nothing.

## Querying

```http
POST /v1/corpora/handbook/query
{ "query": "how do I rotate the signing key?", "token_budget": 800 }
```

The request body:

| Field | Default | Meaning |
|---|---|---|
| `query` | — | The natural-language query, embedded once. |
| `token_budget` | `null` | Budget for the returned context. Omitted → dedup only. Capped by `limits.max_token_budget`. |
| `top_k_per_size` | `10` | Top-k retrieved per grid size before fusion. |
| `strategy` | `greedy_upward` | Selection strategy name. |
| `expansion` | `fill` | Trade-up mode: `fill`, `retrieved_only`, or `off`. |
| `sizes` | all | Restrict the fan-out to these grid sizes. |
| `tokenizer` | corpus pin | Override the tokenizer for this request (counts recomputed on the fly). |
| `max_slice_size` | `null` | Cap on how large a traded-up slice may grow. |
| `include_text` | `true` | Include slice text in results. |
| `include_trace` | `true` | Include the substitution trace. |

An omitted or `null` budget means **dedup only**: nested duplicates are removed,
but nothing is traded up (an unbounded trade-up would balloon every result to the
largest grid size). A budget too small to fit even one slice is not an error — it
returns `200` with an empty `results` array and a warning. A budget above the
server cap is clamped, with a warning.

### The query response maps the engine's output faithfully

The response mirrors the selection engine's result verbatim — results are never
re-sorted, and nothing is re-derived. Each entry in `results` carries:

- `coords` — the slice's identity and geometry (`document_id`, `size`,
  `codepoint_offset`, `codepoint_length`, `codepoint_end`, `own_marker`);
- `text`, `token_count`;
- `provenance` — `retrieved` if this exact slice was a hit, `traded_up` if it was
  reached by growing a smaller hit into its parent;
- `effective_rank` — the best fused rank across the slice's supporting evidence;
- `contiguous_with_next` — true when the next result immediately follows this
  slice in the same document (for optional joined rendering);
- `retrieval` — `{ retrieved, rank_in_size, score }`. **Score is `null` for a
  traded-up slice** — cross-size scores are never fabricated;
- `selection` — `{ action, replaced, levels }`: whether the slice was `kept` or
  `traded_up`, the ids it replaced or subsumed, and how many grid levels it grew;
- `evidence` — the genuine per-size hits that support this slice, each with its
  own within-size `score` and `rank_in_size`.

Alongside `results`, the response carries `budget` (`{ limit, used, counter }`),
`partial`, a per-size `fanout` status, `warnings`, and — unless suppressed — the
full `trace`.

### The trace

The `trace` object is the explainability contract: it records every touched slice
in exactly one lifecycle path (`fusion` → `forest` → `dedup` → `initial_pack` →
`trade_ups` / `rejections` → `final`), plus the forest shape and the final budget
arithmetic. `trace.final.tokens_used` always equals `budget.used`.

## Partial-failure semantics

Retrieval fans out one search per grid size, concurrently, each with its own
timeout. The policy is deliberate:

- **At least one size answered → `200` with `partial: true`.** The `fanout` array
  reports which sizes succeeded and which failed and why. The engine degrades
  gracefully: a missing size simply contributes no candidates.
- **Every size failed → `503 STORE_UNAVAILABLE`.** Only a total fan-out failure
  fails the request.
- A failed trade-up ancestor prefetch never fails a request — the engine records
  a fetch miss and keeps the smaller slice.

## Error taxonomy

Every non-2xx response is the uniform envelope:

```json
{ "error": { "code": "...", "message": "...", "details": null, "request_id": "..." } }
```

`code` is stable and machine-readable; act on it rather than parsing prose.
`request_id` echoes the `X-Request-Id` header (assigned if the client did not
supply one) and appears on every response header too.

| Code | HTTP | When |
|---|---|---|
| `VALIDATION_ERROR` | 422 | Malformed request body or parameters. |
| `INVALID_GRID` | 422 | The requested grid is not a valid divisibility chain. |
| `SLICING_ERROR` | 422 | A document cannot be sliced (e.g. lone surrogates). |
| `UNKNOWN_TOKENIZER` | 422 | The pinned or overridden tokenizer is unknown. |
| `CORPUS_NOT_FOUND` | 404 | No such corpus. |
| `DOCUMENT_NOT_FOUND` | 404 | No such document. |
| `CORPUS_EXISTS` | 409 | A corpus with this name already exists. |
| `EMBEDDER_MISMATCH` | 409 | Data offered to a corpus disagrees with its pinned embedder/grid. |
| `DOCUMENT_TOO_LARGE` | 413 | The document or batch exceeds a configured limit. |
| `EMBEDDER_UNAVAILABLE` | 503 | The embedding provider is unreachable. |
| `STORE_UNAVAILABLE` | 503 | The store is unreachable, or every query fan-out failed. |
| `UNAUTHORIZED` | 401 | Missing or wrong bearer token. |
| `INTERNAL` | 500 | An unexpected error. |

The core error codes surface unchanged through the envelope; the taxonomy is
extended by the server layer (for example `DOCUMENT_TOO_LARGE`,
`UNAUTHORIZED`), never remapped.
