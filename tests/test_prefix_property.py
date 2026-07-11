"""GATE: the prefix property and the slicing invariants it rests on.

For any document, the left-anchored spine — the slices at code-point offset 0
across all grid sizes — must nest as byte-prefixes of one another (proper
prefixes whenever the larger slice extends further; byte-identical for documents
shorter than the larger size), and each smaller slice's marker must appear in
every larger slice's descendant-marker list.
A failure indicates a bug in grid alignment, code-point-to-byte mapping, or
descendant assignment, and nothing downstream (containment, dedup, trade-up) can
be trusted until it passes.

The four canonical corpora are exact ports of the reference implementation's
gating test; the hypothesis suite generalizes the invariants to arbitrary text
and arbitrary valid divisibility-chain grids (nothing may assume powers of two).
"""

from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from phoropter import GridSpec, SlicedDocument, multi_view_slice

pytestmark = pytest.mark.gate


def multi_script_document(min_code_points: int) -> str:
    """Exact port of the reference corpus builder: whole samples until the target is met."""
    sample = "Hello, world! 世界 こんにちは Привет שלום مرحبا नमस्ते 🙂🌍🚀 "
    parts: list[str] = []
    code_points = 0
    while code_points < min_code_points:
        parts.append(sample)
        code_points += len(sample)
    return "".join(parts)


def verify_prefix_property_at_offset_zero(text: str) -> None:
    doc = multi_view_slice("doc", text)
    spine = {s.size: s for s in doc.spine()}
    assert len(spine) == len(doc.grid.sizes), "expected exactly one slice at offset 0 per size"

    sizes = doc.grid.sizes
    for smaller_size, larger_size in pairwise(sizes):
        smaller, larger = spine[smaller_size], spine[larger_size]
        smaller_bytes = smaller.text.encode("utf-8")
        larger_bytes = larger.text.encode("utf-8")
        assert larger_bytes[: len(smaller_bytes)] == smaller_bytes, (
            f"size {smaller_size} bytes not a prefix of size {larger_size}"
        )
        assert smaller.own_marker in larger.descendant_markers, (
            f"size {smaller_size} marker missing from size {larger_size} descendant list"
        )


class TestCanonicalCorpora:
    def test_multi_script(self) -> None:
        verify_prefix_property_at_offset_zero(multi_script_document(1024))

    def test_ascii_only(self) -> None:
        verify_prefix_property_at_offset_zero("x" * 1024)

    def test_cjk_heavy(self) -> None:
        verify_prefix_property_at_offset_zero("世" * 1024)

    def test_emoji_heavy(self) -> None:
        verify_prefix_property_at_offset_zero("🙂" * 1024)

    def test_short_document_spine_levels_are_byte_identical(self) -> None:
        # For a document shorter than a grid size, consecutive spine levels are
        # equal spans: identical bytes, identical markers. The trade-up engine
        # depends on recognizing these degenerate levels — pin the behavior.
        verify_prefix_property_at_offset_zero("x" * 100)
        doc = multi_view_slice("doc", "x" * 100)
        spine = {s.size: s for s in doc.spine()}
        for size in (128, 256, 512, 1024):
            assert spine[size].text == doc.text
            assert spine[size].own_marker == spine[128].own_marker
            assert spine[size].codepoint_length == 100


# --- hypothesis generalization -------------------------------------------------

utf8_text = st.text(alphabet=st.characters(codec="utf-8"), max_size=600)


@st.composite
def divisibility_chain_grids(draw: st.DrawFn) -> GridSpec:
    """Arbitrary valid grids: random base, random multipliers — not just powers of two."""
    base = draw(st.integers(min_value=1, max_value=8))
    length = draw(st.integers(min_value=1, max_value=4))
    sizes = [base]
    for _ in range(length - 1):
        sizes.append(sizes[-1] * draw(st.integers(min_value=2, max_value=4)))
    return GridSpec(tuple(sizes))


def assert_all_invariants(doc: SlicedDocument) -> None:
    text, grid = doc.text, doc.grid
    full_bytes = text.encode("utf-8")
    assert doc.byte_length == len(full_bytes)
    by_coords = {(s.size, s.codepoint_offset): s for s in doc.slices}

    for size in grid.sizes:
        pieces = doc.slices_of_size(size)
        # Lossless reconstruction in code points AND bytes.
        assert "".join(p.text for p in pieces) == text
        # Byte coordinates are consistent and gap-free per size.
        running = 0
        for p in pieces:
            assert p.codepoint_offset % p.size == 0, "origin alignment violated"
            assert p.byte_offset == running
            assert full_bytes[p.byte_offset : p.byte_offset + p.byte_length] == p.text.encode(
                "utf-8"
            )
            running += p.byte_length
        assert running == doc.byte_length or not pieces

    for s in doc.slices:
        # Descendants: exactly the closed-form enumeration, in canonical order.
        expected = []
        for child_size in grid.levels_below(s.size):
            expected.extend(
                by_coords[(child_size, o)].own_marker
                for o in range(s.codepoint_offset, s.codepoint_end, child_size)
            )
        assert list(s.descendant_markers) == expected

        # Prefix property at every aligned anchor (not only offset 0): whenever a
        # smaller slice starts exactly where a larger slice starts, its bytes are
        # a prefix of the larger slice's bytes.
        for parent_size in grid.levels_above(s.size):
            if s.codepoint_offset % parent_size != 0:
                continue
            parent = by_coords[(parent_size, s.codepoint_offset)]
            small = s.text.encode("utf-8")
            assert parent.text.encode("utf-8")[: len(small)] == small
            assert s.own_marker in parent.descendant_markers


@settings(max_examples=120, deadline=None)
@given(text=utf8_text)
def test_invariants_hold_for_arbitrary_text_on_default_grid(text: str) -> None:
    assert_all_invariants(multi_view_slice("doc", text))


@settings(max_examples=120, deadline=None)
@given(text=utf8_text, grid=divisibility_chain_grids())
def test_invariants_hold_for_arbitrary_text_and_grid(text: str, grid: GridSpec) -> None:
    assert_all_invariants(multi_view_slice("doc", text, grid))


@settings(max_examples=60, deadline=None)
@given(text=st.text(alphabet=st.characters(codec="utf-8"), min_size=1, max_size=600))
def test_markers_are_deterministic_across_runs(text: str) -> None:
    a = multi_view_slice("doc", text)
    b = multi_view_slice("doc", text)
    assert [s.own_marker for s in a.slices] == [s.own_marker for s in b.slices]
    assert [s.descendant_markers for s in a.slices] == [s.descendant_markers for s in b.slices]
