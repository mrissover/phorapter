"""Offline evaluation harness (requires the ``eval`` extra).

Measures forest density and budget utilization on a real corpus by driving a
running server and aggregating query traces. See :mod:`phorapter.eval.harness`
for the metrics and ``phorapter eval --help`` for the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import TYPE_CHECKING

from phorapter.eval.harness import (
    EvalSummary,
    QueryEval,
    aggregate,
    compare_baseline,
    evaluate,
    load_queries,
    write_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EvalSummary",
    "QueryEval",
    "aggregate",
    "compare_baseline",
    "evaluate",
    "load_queries",
    "main",
    "write_outputs",
]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default="http://localhost:8000", help="server base URL")
    parser.add_argument("--corpus", required=True, help="corpus name")
    parser.add_argument("--queries", required=True, help="queries file (JSONL or plain lines)")
    parser.add_argument(
        "--budget", type=int, default=None, help="token budget (omit for dedup-only)"
    )
    parser.add_argument("--top-k", type=int, default=10, help="top-k per size")
    parser.add_argument("--api-key", default=None, help="bearer token if the server requires one")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phorapter eval", description="Offline evaluation harness"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("forest", "report containment-forest density (participation, depth, edges)"),
        ("budget", "report budget utilization and trade-up behavior"),
        ("regress", "compare aggregates against a baseline summary.json; exit 1 on drift"),
    ):
        p = sub.add_parser(mode, help=help_text)
        _add_common(p)
        p.add_argument("--out", default=None, help="write per_query.csv + summary.json here")
        if mode == "regress":
            p.add_argument(
                "--baseline", required=True, help="baseline summary.json to compare against"
            )
            p.add_argument("--tolerance", type=float, default=0.01, help="relative drift tolerance")
    return parser


async def _run(args: argparse.Namespace) -> int:
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    queries = load_queries(args.queries)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60.0) as client:
        evals = await evaluate(
            client, args.corpus, queries, budget=args.budget, top_k_per_size=args.top_k
        )
    summary = aggregate(evals, budget=args.budget)

    if args.out:
        write_outputs(evals, summary, args.out)

    if args.mode == "forest":
        print(f"queries:               {summary.queries}")
        print(f"mean participation:    {summary.mean_participation_rate:.3f}")
        print(f"min participation:     {summary.min_participation_rate:.3f}")
        print(f"mean max depth:        {summary.mean_max_depth:.2f}")
        print(f"depth distribution:    {summary.depth_distribution}")
        print(f"mean edges/query:      {summary.mean_edges:.1f}")
        return 0
    if args.mode == "budget":
        util = "n/a" if summary.mean_utilization is None else f"{summary.mean_utilization:.3f}"
        print(f"queries:               {summary.queries}")
        print(f"budget:                {summary.budget}")
        print(f"mean results/query:    {summary.mean_results:.1f}")
        print(f"mean trade-ups/query:  {summary.mean_trade_ups:.1f}")
        print(f"mean utilization:      {util}")
        print(f"partial queries:       {summary.partial_queries}")
        return 0
    # regress
    drifts = compare_baseline(summary, args.baseline, tolerance=args.tolerance)
    if drifts:
        print("REGRESSION — aggregates drifted from baseline:")
        for d in drifts:
            print(f"  {d}")
        return 1
    print("no regression: aggregates match baseline within tolerance")
    print(json.dumps({"queries": summary.queries, "budget": summary.budget}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``phorapter eval``."""
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(args))
