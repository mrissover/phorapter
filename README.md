# Phorapter

**Right-sized retrieval context under a token budget.**

Phorapter is a multi-view RAG server (and embeddable Python library) that indexes every
document at multiple slice sizes on a shared origin-aligned grid. At query time it
retrieves across all sizes, detects — *exactly, by construction* — when one retrieved
slice is contained inside another, discards the redundant ones, and trades small slices
up to their larger enclosing slices while a token budget allows. The result: the most
coherent context that fits the budget, with a full trace of every decision.

> **Status: pre-alpha.** The core is under active development; APIs will change.

## Why multi-view slicing?

Conventional RAG pipelines pick one chunk size at ingest time and live with it forever:
small chunks match precisely but strip context, large chunks preserve context but crowd
the model's window. Phorapter defers that choice to query time. Because every size is
sliced on the same origin-aligned grid, a larger slice *provably* contains its smaller
slices — containment is decided by offset arithmetic and verified by content hashes
(SHA-256 markers), not by similarity heuristics. Swapping a small relevant slice for its
enclosing parent never loses relevant content; it only spends tokens. That makes token
budgeting a sequence of safe, explainable moves.

## What ships

- `pip install phorapter` — the core library: multi-size slicer, structural markers,
  containment forest, budget-fitting selection engine. Stdlib-only, synchronous, embeddable
  in any pipeline.
- `pip install "phorapter[server,qdrant]"` — the server: FastAPI REST + MCP surfaces over
  a pluggable vector store (Qdrant default) and pluggable embedders (Ollama and
  OpenAI-compatible included).

## Documentation

- **Start here:** [concepts](docs/concepts.md) (the method) and
  [architecture](docs/architecture.md) (the system).
- **Quickstarts:** [library](docs/quickstart-library.md) (in-process) ·
  [server](docs/quickstart-server.md) (REST + MCP).
- **Reference:** [engine](docs/engine.md) (right-sizing) ·
  [api](docs/api.md) + [`api/openapi.yaml`](api/openapi.yaml) (REST) ·
  [mcp](docs/mcp.md) · [adapters](docs/adapters.md) (SPIs) ·
  [operations](docs/operations.md) · [evaluation](docs/evaluation.md).
- **Rationale:** [decisions](docs/decisions.md) (the design log).

## License

MIT © Impluvium Software
