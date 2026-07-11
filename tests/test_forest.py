"""Containment forest: positional containment, minimal edges, anomalies, metrics."""

import dataclasses

import pytest

from phoropter import DEFAULT_GRID, multi_view_slice
from phoropter.forest import ContainmentForest, ForestError, contains
from phoropter.model import HitProvenance, RetrievedHit


def hit(doc, size, offset, score=1.0, rank=0):
    s = doc.slice_at(size, offset)
    assert s is not None, (size, offset)
    return RetrievedHit(
        slice=s, corpus="c", score=score, rank_in_size=rank, provenance=HitProvenance.RETRIEVED
    )


DOC = multi_view_slice("doc-a", "x" * 1024, DEFAULT_GRID)


class TestMinimalEdges:
    def test_full_spine_has_four_minimal_edges(self) -> None:
        # The offset-0 spine across all five sizes: 4 minimal edges (each level to
        # the next), not the 10 of the transitive closure.
        hits = [hit(DOC, s, 0) for s in DEFAULT_GRID.sizes]
        forest = ContainmentForest.build(hits, DEFAULT_GRID)
        assert len(forest.edges()) == 4
        assert forest.max_depth() == 5

    def test_closure_yields_all_ancestor_edges(self) -> None:
        hits = [hit(DOC, s, 0) for s in DEFAULT_GRID.sizes]
        forest = ContainmentForest.build(hits, DEFAULT_GRID, closure=True)
        assert len(forest.edges()) == 10  # C(5, 2)

    def test_minimal_parent_and_ancestors(self) -> None:
        hits = [hit(DOC, s, 0) for s in DEFAULT_GRID.sizes]
        forest = ContainmentForest.build(hits, DEFAULT_GRID)
        leaf = DOC.slice_at(64, 0).ref
        assert forest.minimal_parent(leaf) == DOC.slice_at(128, 0).ref
        ancestors = forest.retrieved_ancestors(leaf)
        assert [r.size for r in ancestors] == [128, 256, 512, 1024]

    def test_leaves_and_roots(self) -> None:
        hits = [hit(DOC, s, 0) for s in DEFAULT_GRID.sizes]
        forest = ContainmentForest.build(hits, DEFAULT_GRID)
        assert [h.slice.size for h in forest.leaves()] == [64]
        assert [h.slice.size for h in forest.roots()] == [1024]


class TestPositionalIdentity:
    def test_repeated_text_does_not_fabricate_an_edge(self) -> None:
        # Two 64-slices with byte-identical text at different offsets, plus a 128
        # parent of the FIRST. The parent must not link to the second twin, even
        # though their markers match, because identity is coordinates.
        doc = multi_view_slice("doc-a", "x" * 256, DEFAULT_GRID)
        assert doc.slice_at(64, 0).own_marker == doc.slice_at(64, 64).own_marker
        hits = [hit(doc, 64, 0), hit(doc, 64, 128), hit(doc, 128, 0)]
        forest = ContainmentForest.build(hits, DEFAULT_GRID)
        edges = forest.edges()
        # The 128 at offset 0 spans [0,128): it parents 64@0, not 64@128.
        assert len(edges) == 1
        assert edges[0].child.slice.codepoint_offset == 0

    def test_cross_document_identical_text_no_edge(self) -> None:
        a = multi_view_slice("doc-a", "x" * 128, DEFAULT_GRID)
        b = multi_view_slice("doc-b", "x" * 128, DEFAULT_GRID)
        assert a.slice_at(64, 0).own_marker == b.slice_at(64, 0).own_marker
        hits = [hit(a, 64, 0), hit(b, 128, 0)]
        forest = ContainmentForest.build(hits, DEFAULT_GRID)
        assert forest.edges() == ()

    def test_contains_predicate(self) -> None:
        parent = hit(DOC, 128, 0)
        child = hit(DOC, 64, 0)
        assert contains(parent, child)
        assert not contains(child, parent)


class TestAnomalies:
    def test_marker_mismatch_records_anomaly_not_edge(self) -> None:
        # Tamper the parent's descendant list: positional containment holds but the
        # marker no longer vouches -> anomaly, no edge, not fatal.
        parent = hit(DOC, 128, 0)
        tampered = dataclasses.replace(parent.slice, descendant_markers=())
        parent = dataclasses.replace(parent, slice=tampered)
        child = hit(DOC, 64, 0)
        forest = ContainmentForest.build([parent, child], DEFAULT_GRID)
        assert forest.edges() == ()
        assert len(forest.anomalies) == 1
        assert forest.anomalies[0].child_ref == child.ref

    def test_anomaly_walks_past_stale_parent_to_valid_ancestor(self) -> None:
        # If the 128 is stale but the 256 is valid, the child's minimal parent is
        # the 256, and the stale 128 is recorded as an anomaly.
        child = hit(DOC, 64, 0)
        stale_128 = dataclasses.replace(
            hit(DOC, 128, 0), slice=dataclasses.replace(DOC.slice_at(128, 0), descendant_markers=())
        )
        good_256 = hit(DOC, 256, 0)
        forest = ContainmentForest.build([child, stale_128, good_256], DEFAULT_GRID)
        assert forest.minimal_parent(child.ref) == good_256.slice.ref
        assert len(forest.anomalies) == 1


class TestMetrics:
    def test_participation_and_depth(self) -> None:
        # A 128->64 pair plus a standalone 256 elsewhere.
        doc = multi_view_slice("doc-a", "x" * 512, DEFAULT_GRID)
        pair = [hit(doc, 128, 0), hit(doc, 64, 0)]
        standalone = [hit(doc, 256, 256)]
        forest = ContainmentForest.build(pair + standalone, DEFAULT_GRID)
        assert forest.participation_rate() == pytest.approx(2 / 3)
        assert forest.max_depth() == 2

    def test_empty_forest(self) -> None:
        forest = ContainmentForest.build([], DEFAULT_GRID)
        assert forest.participation_rate() == 0.0
        assert forest.max_depth() == 0
        assert forest.edges() == ()

    def test_duplicate_refs_collapsed(self) -> None:
        forest = ContainmentForest.build([hit(DOC, 64, 0), hit(DOC, 64, 0)], DEFAULT_GRID)
        assert len(forest.hits) == 1


def test_off_grid_hit_rejected() -> None:
    bad_slice = dataclasses.replace(DOC.slice_at(64, 0), codepoint_offset=7)
    bad = RetrievedHit(slice=bad_slice, corpus="c", score=1.0, rank_in_size=0)
    with pytest.raises(ForestError):
        ContainmentForest.build([bad], DEFAULT_GRID)
