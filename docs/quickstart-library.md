# Quickstart: the library

The core is a pure, synchronous Python library — no server, no async, one runtime
dependency (tiktoken). Use it to slice documents, detect containment, and
right-size a set of retrieved hits under a token budget, in process.

```bash
pip install phorapter
```

## Slice a document

```python
from phorapter import multi_view_slice, DEFAULT_GRID

doc = multi_view_slice("handbook", open("handbook.txt").read(), DEFAULT_GRID)
print(len(doc.slices), "slices across", DEFAULT_GRID.sizes, "code-point sizes")

for s in doc.slices_of_size(256):
    print(s.codepoint_offset, s.own_marker[:12], len(s.descendant_markers), "descendants")
```

Each slice carries dual coordinates (code point and byte), a SHA-256 marker over
its exact bytes, and the markers of every smaller slice contained in it. See
[concepts.md](concepts.md).

## Right-size retrieved hits under a budget

If you already have a retrieval pipeline, hand its multi-size hits to
`budget_fit`. It dedups (keeps the smallest of nested slices) and trades small
slices up to their enclosing parents while the budget allows, returning the
selected slices and a full decision trace.

```python
from phorapter import budget_fit, RetrievedHit, HitProvenance

# Build RetrievedHit objects from your store's results (one per retrieved slice).
hits = [
    RetrievedHit(slice=s, corpus="docs", score=score, rank_in_size=rank)
    for s, score, rank in my_multi_size_results()
]

selection = budget_fit(hits, budget=2000)  # tiktoken o200k_base by default

for chosen in selection.slices:
    tag = chosen.provenance.value  # "retrieved" or "traded_up"
    print(f"[{chosen.slice.document_id} @ {chosen.slice.codepoint_offset}, "
          f"size {chosen.slice.size}, {tag}] {chosen.slice.text[:60]}...")

print("tokens used:", selection.tokens_used, "of", selection.budget)
```

`selection.trace` records every decision — which hits were deduped, which traded
up, which were rejected and why — so you can explain or debug the outcome. See
[engine.md](engine.md).

### Trade-up beyond the retrieved set

By default, trade-up can only reuse slices that were retrieved. To let it pull in
un-retrieved enclosing slices, pass a `SliceSource` — for example an
`InMemoryStore` you populated at ingest, or any object with a
`get_slices(refs, *, corpus)` method:

```python
from phorapter.stores.memory import InMemoryStore

store = InMemoryStore()          # populated with your slices+vectors at ingest
selection = budget_fit(hits, budget=2000, source=store)
```

## Integrating with LangChain / LlamaIndex

The library is framework-agnostic. Slice and index your documents with your
existing embedder and vector store, retrieve at multiple sizes, wrap the results
as `RetrievedHit`s, and call `budget_fit` as a post-retrieval reranking/packing
step. The returned slice texts are ready to concatenate into a prompt (use
`contiguous_with_next` to join adjacent slices seamlessly).

For a managed server with REST and MCP surfaces instead of in-process use, see
[quickstart-server.md](quickstart-server.md).
