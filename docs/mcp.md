# MCP tools

Phoropter exposes an [MCP](https://modelcontextprotocol.io) server so a
language-model client can retrieve right-sized context directly. It runs two
ways:

- **Streamable HTTP**, mounted at `/mcp` by `phoropter serve` (same process as
  the REST API);
- **stdio**, via `phoropter mcp`, for a local MCP client that launches the
  server as a subprocess.

Both surfaces call the same service core as REST — there is no separate business
logic — so a corpus created over REST is immediately queryable over MCP and vice
versa.

## Tools

### `phoropter_query` (always available)

Retrieve right-sized context for a natural-language query. The server searches
the corpus at several slice sizes, removes duplicate nested passages, and — within
a token budget — trades small matching passages up to their enclosing parent
passages so the returned context is as complete as the budget allows.

Arguments:

| Argument | Required | Meaning |
|---|---|---|
| `corpus` | yes | The corpus to search. |
| `query` | yes | The natural-language query. |
| `token_budget` | no | Context tokens to spend. Omitted → the server's `mcp.default_token_budget` (4000 by default). |

The budget is always capped by the server's `limits.max_token_budget`. The tool
returns structured content mirroring the REST query response (results with
document ids and character ranges, budget accounting, partial-failure flag), plus
a `text` block for direct reading. Each text block is headed
`[document_id @ start..end, size]`. Raw cosine scores are deliberately omitted
from the text rendering: they are ordinal within one slice size only and are
misleading when read across sizes.

### `phoropter_list_corpora` (always available)

List the names of the corpora available on the server.

### `phoropter_add_document` / `phoropter_delete_document` (opt-in)

Write tools that add/replace and delete documents. They ship but are **disabled
by default**. Enable them by setting `mcp.enable_document_tools = true` (or
`PHOROPTER_MCP__ENABLE_DOCUMENT_TOOLS=true`). When disabled they are not
registered at all, so a client never sees them.

## Budget semantics

The budget is the number of context tokens the answer may spend, measured under
the corpus's pinned token counter. A larger budget lets the server trade more
small passages up to their fuller parents; a smaller budget keeps the tightest
matching passages. A budget too small to fit even one passage returns an empty
result with a note, never an error — the model can retry with a larger budget.

Because the underlying selection engine is deterministic, the same query against
the same corpus snapshot with the same budget returns the same passages every
time.

## Enabling and mounting

```toml
# phoropter.toml
[mcp]
enable_document_tools = false   # set true to expose the write tools
default_token_budget = 4000
```

Run the combined REST + MCP server:

```console
$ phoropter serve            # REST under /v1, MCP under /mcp
```

Or run MCP alone over stdio for a local client:

```console
$ phoropter mcp
```
