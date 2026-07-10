# Quickstart: the server

The server wraps the core with REST and MCP surfaces over a pluggable vector store
(Qdrant by default) and embedder (Ollama by default).

```bash
pip install "phorapter[server,qdrant]"
```

## Bring up the stack

The dev compose file starts Qdrant, Ollama (with the default embedding model
pulled), and the server:

```bash
docker compose -f docker-compose.dev.yml up --build
```

Or run the server directly against your own Qdrant and Ollama:

```bash
export PHORAPTER_STORE__URL=http://localhost:6333
export PHORAPTER_EMBEDDER__PROVIDER=ollama
export PHORAPTER_EMBEDDER__MODEL=nomic-embed-text
phorapter check     # confirm the store and embedder are reachable
phorapter serve
```

## Create a corpus, add a document, query it

```bash
# 1. Create a corpus. The grid, embedder, and tokenizer are frozen here from
#    server defaults; the embedding dimension is probed and pinned.
curl -X POST localhost:8000/v1/corpora -H 'content-type: application/json' \
  -d '{"name": "docs"}'

# 2. Add (or replace) a document. Re-PUT with the same id to replace it.
curl -X PUT localhost:8000/v1/corpora/docs/documents/handbook \
  -H 'content-type: application/json' \
  -d '{"text": "... your document text ..."}'

# 3. Query with a token budget. The server retrieves across every size, discards
#    contained duplicates, and trades up to larger slices while the budget allows.
curl -X POST localhost:8000/v1/corpora/docs/query \
  -H 'content-type: application/json' \
  -d '{"query": "how do I rotate the signing key?", "token_budget": 2000, "include_trace": true}'
```

The query response returns the selected slices in priority order — each with its
source document, offsets, provenance (`retrieved` or `traded_up`), and genuine
retrieval scores — a budget report, and (with `include_trace`) the full decision
trace. See [api.md](api.md) for the complete contract, and
[`api/openapi.yaml`](../api/openapi.yaml) for the machine-readable spec.

## Use it from an LLM (MCP)

The server mounts an MCP endpoint at `/mcp`, exposing a `phorapter_query` tool
(and `phorapter_list_corpora`). An MCP-capable model can call it directly to pull
budgeted, right-sized context. See [mcp.md](mcp.md).

```bash
phorapter mcp        # or run the MCP server over stdio for a local client
```

## Evaluate behavior on your corpus

```bash
phorapter eval forest --url http://localhost:8000 --corpus docs --queries queries.jsonl
phorapter eval budget --url http://localhost:8000 --corpus docs --queries queries.jsonl --budget 2000
```

See [evaluation.md](evaluation.md).
