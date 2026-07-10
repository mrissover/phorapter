# Adapters (SPIs)

Phorapter never hard-codes a vector store, an embedding model, or a tokenizer.
Each is a small provider interface (SPI) with a shipped default and a plugin
mechanism. This document is the reference for implementing your own.

- **`VectorStoreAdapter`** — where slices and vectors live. Default: Qdrant.
- **`Embedder`** — turns text into vectors. Defaults: Ollama, OpenAI-compatible.
- **`TokenCounter`** — counts tokens for budgeting. Default: tiktoken.

All three are discovered through Python entry-point groups, so a third-party
package can register a provider just by declaring an entry point — no changes to
phorapter.

| SPI | Entry-point group | Shipped |
|---|---|---|
| Vector store | `phorapter.stores` | `memory`, `qdrant` |
| Embedder | `phorapter.embedders` | `fake`, `ollama`, `openai_compat` |
| Token counter | (registered in-process) | `tiktoken:*` |

## VectorStoreAdapter

The SPI speaks **logical coordinates**: a corpus name and a grid size. How you
realize those in storage — one collection per (corpus, size), one table with
discriminator columns, separate databases — is entirely your business. Nothing
above the interface depends on the physical layout.

```python
class VectorStoreAdapter(ABC):
    async def ping(self) -> bool                                   # reachable? (False, never raise)
    async def bootstrap(self) -> None                              # create shared infra, idempotent
    async def create_corpus(self, config: CorpusConfig) -> None    # once; CorpusExistsError on dup
    async def drop_corpus(self, corpus: str) -> None
    async def get_corpus_meta(self, corpus: str) -> CorpusConfig   # CorpusNotFoundError if absent
    async def list_corpora(self) -> tuple[str, ...]                # sorted
    async def verify_corpus(self, corpus: str) -> list[str]        # [] = healthy; else reasons
    async def upsert_slices(self, corpus, points, *, grid_fingerprint) -> None
    async def put_document_meta(self, corpus, record) -> None
    async def get_document_meta(self, corpus, document_id) -> DocumentRecord
    async def list_documents(self, corpus, *, limit=100, cursor=None) -> DocumentPage
    async def list_point_ids(self, corpus, document_id) -> dict[int, set[UUID]]
    async def delete_points(self, corpus, ids_by_size) -> None     # idempotent
    async def delete_document(self, corpus, document_id) -> None
    async def search_size(self, corpus, size, vector, k, *, timeout_s=None) -> list[RetrievedHit]
    async def fetch_by_ids(self, corpus, size, ids) -> list[SlicePoint]
    async def corpus_stats(self, corpus) -> CorpusStats
```

### Normative contract (the conformance suite enforces all of this)

- **Deterministic search ordering.** `search_size` returns hits sorted by score
  descending, ties broken by point id ascending, with a 0-based `rank_in_size`
  assigned in that order. Determinism is what lets the whole engine be
  reproducible.
- **Payload-complete hits.** Every hit and every fetched point carries a fully
  reconstructed `Slice` — all coordinates, text, markers, and token count. The
  engine never re-derives slice content from a second source; containment is
  decidable from the retrieval payloads alone.
- **Frozen config.** `create_corpus` persists the `CorpusConfig` once.
  `upsert_slices` refuses data whose grid fingerprint or vector dimension
  disagrees with it (`CorpusMismatchError`). There is never a silent migration.
- **Exact orphan sets.** `list_point_ids` is the replace/tombstone workhorse:
  after re-slicing a shrunken document, the orphan set is exactly the per-size
  difference between the old ids and the new generation's ids.
- **Errors.** Unknown corpus → `CorpusNotFoundError`; duplicate creation →
  `CorpusExistsError`; unknown document → `DocumentNotFoundError`; backend
  unreachable or failing → `StoreError`.

### The slice payload contract (schema_version 1)

The payload an adapter stores per slice (the packed form is what Qdrant keeps;
other backends may map fields to columns):

```json
{
  "document_id": "handbook-07", "size": 512,
  "codepoint_offset": 1024, "codepoint_length": 512,
  "byte_offset": 1090, "byte_length": 545,
  "document_codepoint_length": 48211,
  "text": "...",
  "own_marker": "<64 lowercase hex>",
  "descendant_markers": "<base64 of concatenated 32-byte digests>",
  "token_count": 389,
  "schema_version": 1
}
```

Descendant markers are packed with `pack_markers` / `unpack_markers`: the base64
of the concatenated full 32-byte SHA-256 digests, order-preserving and lossless.
Full digests (not truncated prefixes) keep containment exact. Document-level
facts live once on the `DocumentRecord`, never duplicated onto every slice.

### The conformance suite

`tests/spi/test_store_contract.py` drives the full SPI through realistic
lifecycles and asserts every guarantee above. It is parametrized over store
implementations — the in-memory store always, Qdrant under the `integration`
marker. **A third-party adapter claims compliance by parametrizing this suite
over its own factory.** Import the suite, add a `pytest.param` for your store's
async-context-manager factory, and run it; a green run is the definition of a
compliant adapter.

### Default: Qdrant

One collection per (corpus, size) — `{prefix}__{corpus}__s{size}` — plus
`{prefix}__meta` (frozen configs) and `{prefix}__docs` (the document registry).
Per-size collections mean top-k retrieval maps to one collection, each size gets
its own HNSW geometry, and a single unhealthy collection degrades exactly one
size (the unit of the partial-failure policy). Point IDs are the deterministic
slice UUIDs, so re-ingest is an in-place upsert. Every slice collection carries a
keyword payload index on `document_id` for tombstoning and delete-by-document.

### Reference: in-memory

`InMemoryStore` is the behavioral yardstick: zero third-party dependencies,
exhaustive brute-force cosine search (so its ordering is exact by construction),
full SPI semantics. It also exposes a synchronous `get_slices` — the engine's
`SliceSource` — so in-process library users can run budget-fitting trade-up
directly against it without an event loop.

## Embedder

```python
class Embedder(ABC):
    provider: str
    model: str
    async def embed(self, texts: Sequence[str]) -> list[list[float]]  # order-preserving
    async def dimension(self) -> int                                  # probed once, cached
    async def health(self) -> EmbedderHealth                          # never raises
    def fingerprint(self) -> str                                      # "provider:model"
```

Phorapter learns a corpus's embedding dimension by **probing** — embedding a
fixed text and measuring the vector — never by trusting a provider metadata
endpoint. The `"provider:model"` fingerprint is pinned into the corpus at
creation; data embedded by anything else is refused. `embed` owns its own
batching and retries; callers hand over the full text list and never see
transport mechanics.

- **`FakeEmbedder`** — deterministic hash-derived unit vectors, dependency-free.
  Used by tests and offline development; there is no semantic signal, only
  determinism.
- **`OllamaEmbedder`** — `POST /api/embed` (batch), falling back to the legacy
  `/api/embeddings` if the server predates it.
- **`OpenAICompatEmbedder`** — `POST /v1/embeddings` with a bearer token; works
  against OpenAI, vLLM, text-embeddings-inference, LM Studio, and similar.

## TokenCounter

```python
class TokenCounter(Protocol):
    counter_id: str                       # e.g. "tiktoken:o200k_base"
    def count(self, text: str) -> int
```

A counter is pinned per corpus (`counter_id`) at creation; per-slice counts are
computed at ingest and stored, so budgeting is a lookup, not a tokenization, at
query time. `get_counter` resolves any `tiktoken:<encoding>` id dynamically;
register other counters with `register_counter`. tiktoken is imported lazily, so
the core library imports and runs without it.

## Registering a plugin

Declare an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."phorapter.stores"]
myrstore = "mypkg.store:MyStore"

[project.entry-points."phorapter.embedders"]
myembedder = "mypkg.embed:MyEmbedder"
```

`load_store_class(name)` and the embedder registry's `load_entry_points()`
resolve them. Later registrations win, so a plugin can deliberately shadow a
built-in name.
