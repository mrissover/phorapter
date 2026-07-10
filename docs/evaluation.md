# Evaluation

The evaluation harness measures the *structure and budgeting behavior* of
Phorapter on a real corpus — how dense and deep the containment forest is, how
much of the token budget is used, and how much trade-up happens. It does **not**
measure answer quality; that is a downstream, application-specific concern.

The harness (the `eval` extra) drives a running server's query endpoint over a
set of queries and aggregates the substitution trace each returns. Because the
selection engine is deterministic, the same corpus, queries, and budget produce
the same numbers every run — which is what makes regression detection meaningful.

## Running

```bash
# forest density: participation rate, chain depth, edge counts
phorapter eval forest  --url http://localhost:8000 --corpus docs --queries queries.jsonl

# budget behavior: results/query, trade-ups/query, utilization
phorapter eval budget  --url http://localhost:8000 --corpus docs --queries queries.jsonl --budget 2000

# regression: compare aggregates to a saved baseline; exit 1 on drift
phorapter eval regress --url http://localhost:8000 --corpus docs --queries queries.jsonl \
    --budget 2000 --baseline eval/summary.json
```

`--queries` is a file of JSONL objects (a `query` or `text` key) or one query per
line. `--out <dir>` writes `per_query.csv` (one row per query) and `summary.json`
(the aggregates). `--api-key` supplies a bearer token if the server requires one.

## Metrics

**Forest density** (per query, from the trace's `forest` block):

- *participation rate* — the fraction of retrieved slices that take part in at
  least one containment edge. A high rate means the multi-size structure is
  pervasive, so trade-up has a lot to work with.
- *max chain depth* — the length of the longest containment chain (how many
  sizes the same region surfaces at). Deeper chains mean more trade-up headroom.
- *edge count* — total containment edges per query (upward-substitution
  opportunities).

**Budget behavior** (from `final` and `trade_ups`):

- *results per query* — how many slices the selection returns.
- *trade-ups per query* — how many upward substitutions the engine performed.
- *utilization* — `tokens_used / budget`, how fully the budget was packed.
- *partial queries* — how many completed with a partial fan-out (a size failed).

## Regression gating

`regress` compares the current aggregates against a committed `summary.json` and
exits non-zero if any numeric aggregate drifts beyond the tolerance (relative,
default 1%) or the depth distribution changes. Wire it into CI against a pinned
corpus and query file to catch unintended behavior changes: because the engine is
deterministic, a drift is a real change in what the server does, not noise.

To refresh a baseline after an intended change, run any subcommand with `--out`
and commit the new `summary.json`, noting the change in the PR.
