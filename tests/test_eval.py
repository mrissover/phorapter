"""Evaluation harness: metric aggregation, regression comparison, end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from phorapter.eval.harness import (
    QueryEval,
    aggregate,
    compare_baseline,
    evaluate,
    load_queries,
    write_outputs,
)


def _qe(**kw) -> QueryEval:
    base = {
        "query": "q",
        "results": 3,
        "forest_hits": 10,
        "forest_edges": 6,
        "participation_rate": 0.8,
        "max_depth": 3,
        "trade_ups": 2,
        "rejections": 1,
        "tokens_used": 400,
        "budget": 500,
        "utilization": 0.8,
        "partial": False,
    }
    base.update(kw)
    return QueryEval(**base)  # type: ignore[arg-type]


class TestLoadQueries:
    def test_plain_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "q.txt"
        p.write_text("first query\n\nsecond query\n", encoding="utf-8")
        assert load_queries(p) == ["first query", "second query"]

    def test_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "q.jsonl"
        p.write_text('{"query": "a"}\n{"text": "b"}\n', encoding="utf-8")
        assert load_queries(p) == ["a", "b"]


class TestAggregate:
    def test_means_and_distribution(self) -> None:
        evals = [_qe(max_depth=3, participation_rate=0.6), _qe(max_depth=5, participation_rate=1.0)]
        summary = aggregate(evals, budget=500)
        assert summary.queries == 2
        assert summary.mean_participation_rate == pytest.approx(0.8)
        assert summary.min_participation_rate == pytest.approx(0.6)
        assert summary.depth_distribution == {"3": 1, "5": 1}
        assert summary.mean_max_depth == pytest.approx(4.0)

    def test_empty(self) -> None:
        summary = aggregate([], budget=None)
        assert summary.queries == 0


class TestRegression:
    def test_no_drift(self, tmp_path: Path) -> None:
        summary = aggregate([_qe()], budget=500)
        baseline = tmp_path / "b.json"
        write_outputs([_qe()], summary, tmp_path)
        baseline = tmp_path / "summary.json"
        assert compare_baseline(summary, baseline) == []

    def test_drift_detected(self, tmp_path: Path) -> None:
        write_outputs(
            [_qe(participation_rate=0.8)],
            aggregate([_qe(participation_rate=0.8)], budget=500),
            tmp_path,
        )
        drifted = aggregate([_qe(participation_rate=0.4)], budget=500)
        drifts = compare_baseline(drifted, tmp_path / "summary.json")
        assert any("participation" in d for d in drifts)

    def test_outputs_written(self, tmp_path: Path) -> None:
        write_outputs(
            [_qe(), _qe(max_depth=5)], aggregate([_qe(), _qe(max_depth=5)], budget=500), tmp_path
        )
        assert (tmp_path / "per_query.csv").exists()
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["queries"] == 2


class TestEndToEnd:
    async def test_evaluate_against_in_process_app(self) -> None:
        # A real app (in-memory store + fake embedder) driven over ASGI — no network.
        from phorapter.config import Settings
        from phorapter.embed import FakeEmbedder
        from phorapter.server.rest import create_app
        from phorapter.service.core import ServiceCore
        from phorapter.stores.memory import InMemoryStore

        settings = Settings(store={"kind": "memory"}, embedder={"provider": "fake"})
        core = ServiceCore(
            store=InMemoryStore(), embedder=FakeEmbedder(dimension=32), settings=settings
        )
        app = create_app(settings, core=core)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/v1/corpora", json={"name": "docs"})
            await client.put(
                "/v1/corpora/docs/documents/d1",
                json={"text": "the quick brown fox jumps over the lazy dog " * 20},
            )
            evals = await evaluate(client, "docs", ["quick fox", "lazy dog"], budget=500)

        assert len(evals) == 2
        assert all(e.budget == 500 for e in evals)
        # Deterministic engine: a re-run yields identical metrics.
        transport2 = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport2, base_url="http://test") as client:
            again = await evaluate(client, "docs", ["quick fox", "lazy dog"], budget=500)
        assert [e.tokens_used for e in again] == [e.tokens_used for e in evals]
