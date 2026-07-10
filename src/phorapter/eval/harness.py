"""Offline evaluation harness.

Measures the *structure and budgeting behavior* of phorapter on a real corpus —
not answer quality. It drives a running server's query endpoint over a set of
queries and aggregates the substitution trace each returns: how dense and deep
the containment forest is, how much of the budget is used, and how much trade-up
happens.

Because the engine is deterministic, the same corpus, queries, and budget yield
the same numbers on every run — so ``regress`` can treat any change in the
aggregates as a real behavior change, not noise.

The harness talks HTTP (the ``eval`` extra pulls in httpx). A caller may inject
an ``httpx.AsyncClient`` bound to an in-process app for testing.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx

__all__ = [
    "EvalSummary",
    "QueryEval",
    "aggregate",
    "compare_baseline",
    "evaluate",
    "load_queries",
    "write_outputs",
]


@dataclass(frozen=True, slots=True)
class QueryEval:
    """Per-query evaluation metrics, read from one query response's trace."""

    query: str
    results: int
    forest_hits: int
    forest_edges: int
    participation_rate: float
    max_depth: int
    trade_ups: int
    rejections: int
    tokens_used: int
    budget: int | None
    utilization: float | None
    partial: bool


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregate metrics across a query set (means, extremes, distributions)."""

    queries: int
    budget: int | None
    mean_participation_rate: float
    min_participation_rate: float
    mean_max_depth: float
    depth_distribution: dict[str, int]
    mean_edges: float
    mean_results: float
    mean_trade_ups: float
    mean_utilization: float | None
    partial_queries: int


def load_queries(path: str | Path) -> list[str]:
    """Read queries from a file: JSONL objects (``query``/``text`` key) or plain lines."""
    text = Path(path).read_text(encoding="utf-8")
    queries: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            value = obj.get("query") or obj.get("text")
            if value:
                queries.append(str(value))
        else:
            queries.append(line)
    return queries


async def evaluate(
    client: httpx.AsyncClient,
    corpus: str,
    queries: Sequence[str],
    *,
    budget: int | None,
    top_k_per_size: int = 10,
) -> list[QueryEval]:
    """Run each query against ``/v1/corpora/{corpus}/query`` and collect trace metrics."""
    out: list[QueryEval] = []
    for query in queries:
        payload: dict[str, Any] = {
            "query": query,
            "top_k_per_size": top_k_per_size,
            "include_text": False,
            "include_trace": True,
        }
        if budget is not None:
            payload["token_budget"] = budget
        response = await client.post(f"/v1/corpora/{corpus}/query", json=payload)
        response.raise_for_status()
        body = response.json()
        trace = body["trace"]
        forest = trace["forest"]
        final = trace["final"]
        out.append(
            QueryEval(
                query=query,
                results=len(body["results"]),
                forest_hits=forest["hit_count"],
                forest_edges=forest["edge_count"],
                participation_rate=forest["participation_rate"],
                max_depth=forest["max_depth"],
                trade_ups=len(trace["trade_ups"]),
                rejections=len(trace["rejections"]),
                tokens_used=final["tokens_used"],
                budget=final["budget"],
                utilization=final["utilization"],
                partial=body["partial"],
            )
        )
    return out


def aggregate(evals: Sequence[QueryEval], *, budget: int | None) -> EvalSummary:
    """Reduce per-query metrics to an :class:`EvalSummary`."""
    if not evals:
        return EvalSummary(
            queries=0,
            budget=budget,
            mean_participation_rate=0.0,
            min_participation_rate=0.0,
            mean_max_depth=0.0,
            depth_distribution={},
            mean_edges=0.0,
            mean_results=0.0,
            mean_trade_ups=0.0,
            mean_utilization=None,
            partial_queries=0,
        )
    depth_dist: dict[str, int] = {}
    for e in evals:
        key = str(e.max_depth)
        depth_dist[key] = depth_dist.get(key, 0) + 1
    utils = [e.utilization for e in evals if e.utilization is not None]
    return EvalSummary(
        queries=len(evals),
        budget=budget,
        mean_participation_rate=statistics.fmean(e.participation_rate for e in evals),
        min_participation_rate=min(e.participation_rate for e in evals),
        mean_max_depth=statistics.fmean(e.max_depth for e in evals),
        depth_distribution=dict(sorted(depth_dist.items(), key=lambda kv: int(kv[0]))),
        mean_edges=statistics.fmean(e.forest_edges for e in evals),
        mean_results=statistics.fmean(e.results for e in evals),
        mean_trade_ups=statistics.fmean(e.trade_ups for e in evals),
        mean_utilization=statistics.fmean(utils) if utils else None,
        partial_queries=sum(1 for e in evals if e.partial),
    )


def write_outputs(evals: Sequence[QueryEval], summary: EvalSummary, out_dir: str | Path) -> None:
    """Write ``per_query.csv`` and ``summary.json`` under ``out_dir``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fields = list(QueryEval.__annotations__)
    lines = [",".join(fields)]
    for e in evals:
        row = asdict(e)
        lines.append(",".join(_csv_cell(row[f]) for f in fields))
    (directory / "per_query.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(c in text for c in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def compare_baseline(
    summary: EvalSummary, baseline_path: str | Path, *, tolerance: float = 0.01
) -> list[str]:
    """Return the list of metrics that drifted from a baseline beyond ``tolerance``.

    An empty list means no regression. Numeric aggregates are compared by
    relative difference; the depth distribution is compared exactly.
    """
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    current = asdict(summary)
    drifts: list[str] = []
    for key, base_value in baseline.items():
        cur_value = current.get(key)
        if isinstance(base_value, (int, float)) and isinstance(cur_value, (int, float)):
            denom = abs(base_value) if base_value else 1.0
            if abs(cur_value - base_value) / denom > tolerance:
                drifts.append(f"{key}: {base_value} -> {cur_value}")
        elif base_value != cur_value:
            drifts.append(f"{key}: {base_value!r} -> {cur_value!r}")
    return drifts
