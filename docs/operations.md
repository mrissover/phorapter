# Operations

How to configure, deploy, and run a Phoropter server.

## Configuration

Settings come from three sources, highest precedence first: explicit values,
environment variables prefixed `PHOROPTER_`, then an optional `phoropter.toml` in
the working directory. Nested fields use a double underscore in the environment,
e.g. `PHOROPTER_STORE__URL`.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `server` | `host` / `port` | `127.0.0.1` / `8000` | bind address |
| `server` | `api_key` | (unset) | if set, all `/v1` routes require `Authorization: Bearer <key>` |
| `store` | `kind` | `qdrant` | store adapter name (`qdrant`, `memory`, or a plugin) |
| `store` | `url` / `api_key` / `prefix` | `http://localhost:6333` / – / `phoropter` | Qdrant connection and collection namespace |
| `store` | `search_timeout_s` | `5` | per-size search timeout |
| `embedder` | `provider` / `model` | `ollama` / `nomic-embed-text` | embedder (`ollama`, `openai_compat`, `fake`, or a plugin) |
| `embedder` | `base_url` / `api_key` | provider default | endpoint and credential |
| `embedder` | `batch_size` / `max_concurrency` | `32` / `4` | embedding throughput knobs |
| `defaults` | `grid_sizes` / `tokenizer` / `top_k_per_size` | `[64,128,256,512,1024]` / `tiktoken:o200k_base` / `10` | applied once, at corpus creation |
| `limits` | `max_document_codepoints` | `500000` | reject larger documents (413) |
| `limits` | `max_batch_documents` | `100` | batch ingest ceiling |
| `limits` | `max_token_budget` | `32000` | query budget cap |
| `mcp` | `enable_document_tools` | `false` | expose document write tools over MCP |
| `mcp` | `default_token_budget` | `4000` | budget used when an MCP query omits one |

The `defaults` apply **exactly once**, when a corpus is created; from then on the
grid, embedder, and tokenizer are frozen into that corpus's persisted config.
Changing a default never affects an existing corpus.

## Running

```bash
phoropter serve          # REST API + MCP mounted at /mcp, under uvicorn
phoropter mcp            # MCP server over stdio (for a local MCP client)
phoropter check          # validate startup (store reachable, embedder probed); exit 0 or 1
phoropter eval --help    # offline evaluation harness (see evaluation.md)
```

`serve` reads the configured store and embedder, validates them at startup, and
mounts the MCP streamable-HTTP app at `/mcp`. `check` is what the container
healthcheck runs.

## Deployment

The `Dockerfile` builds a wheel and installs it with the `server` and `qdrant`
extras, bakes the default tiktoken vocabulary into the image (so budgeting works
without network access), runs as a non-root user, and exposes port 8000. Its
`HEALTHCHECK` runs `phoropter check`.

`docker-compose.dev.yml` brings up Qdrant, Ollama, a one-shot model pull, and the
server together for local development:

```bash
docker compose -f docker-compose.dev.yml up --build
```

In production, point `PHOROPTER_STORE__URL` at your Qdrant cluster and
`PHOROPTER_EMBEDDER__*` at your embedding service. The server is stateless — all
corpus configuration lives in the store — so it scales horizontally behind a load
balancer.

## Health and readiness

- `GET /healthz` — liveness (process up).
- `GET /v1/health` — readiness with per-component detail (store reachable,
  embedder probed, per-corpus status).
- `GET /v1/info` — version, defaults, and the registered strategies, counters,
  and providers.

## Security

Authentication is an **optional static bearer token** (`server.api_key`); when
set, every `/v1` route requires it and `/healthz` stays open. This is deliberately
minimal — real authentication, authorization, rate limiting, and TLS termination
belong in a reverse proxy or API gateway in front of the server.

## Logging

Logs are structured (one JSON object per line by default; set
`PHOROPTER_LOGGING__JSON=false` for human-readable). Each request logs its method,
path, status, duration, and a `request_id` that is also returned in the
`X-Request-Id` response header and echoed in error envelopes for correlation.

## Embedder and grid changes

An embedder or grid is pinned per corpus at creation and cannot change in place —
mixing vector spaces or grids silently corrupts retrieval. To move a corpus to a
new embedding model or grid, create a new corpus and re-ingest. A dimension or
grid mismatch on upsert is refused with `409 EMBEDDER_MISMATCH`.
