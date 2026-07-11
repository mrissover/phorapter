"""Slicer: geometry, byte accounting, descendants, reconstruction, token counts."""

import pytest

from phoropter import DEFAULT_GRID, GridSpec, SlicingError, multi_view_slice


class CharCounter:
    """Deterministic fake counter for tests: one token per code point."""

    counter_id = "test:chars"

    def count(self, text: str) -> int:
        return len(text)


class TestInputHandling:
    def test_empty_text_yields_no_slices(self) -> None:
        doc = multi_view_slice("d", "")
        assert doc.slices == ()
        assert doc.codepoint_length == 0
        assert doc.byte_length == 0

    def test_missing_document_id_rejected(self) -> None:
        with pytest.raises(SlicingError):
            multi_view_slice("", "text")

    def test_non_string_text_rejected(self) -> None:
        with pytest.raises(SlicingError):
            multi_view_slice("d", b"bytes")  # type: ignore[arg-type]

    def test_lone_surrogate_rejected(self) -> None:
        with pytest.raises(SlicingError):
            multi_view_slice("d", "a\ud800b")


class TestGeometry:
    def test_slice_counts_for_1200_codepoints(self) -> None:
        doc = multi_view_slice("d", "x" * 1200)
        counts = {size: len(doc.slices_of_size(size)) for size in DEFAULT_GRID.sizes}
        assert counts == {64: 19, 128: 10, 256: 5, 512: 3, 1024: 2}

    def test_short_final_slice_lengths(self) -> None:
        doc = multi_view_slice("d", "x" * 1200)
        finals = {s.size: s for s in doc.slices if s.codepoint_end == 1200}
        assert finals[64].codepoint_length == 1200 - 18 * 64  # 48
        assert finals[128].codepoint_length == 1200 - 9 * 128  # 48
        assert finals[256].codepoint_length == 1200 - 4 * 256  # 176
        assert finals[512].codepoint_length == 1200 - 2 * 512  # 176
        assert finals[1024].codepoint_length == 1200 - 1024  # 176
        assert all(s.is_short for s in finals.values())

    def test_boundaries_are_origin_aligned(self) -> None:
        doc = multi_view_slice("d", "x" * 1200)
        assert all(s.codepoint_offset % s.size == 0 for s in doc.slices)

    def test_single_char_document_has_one_slice_per_size(self) -> None:
        doc = multi_view_slice("d", "a")
        assert len(doc.slices) == len(DEFAULT_GRID.sizes)
        assert all(s.codepoint_length == 1 for s in doc.slices)

    def test_canonical_order_size_then_offset(self) -> None:
        doc = multi_view_slice("d", "x" * 300)
        keys = [(s.size, s.codepoint_offset) for s in doc.slices]
        assert keys == sorted(keys, key=lambda k: (DEFAULT_GRID.sizes.index(k[0]), k[1]))

    def test_document_codepoint_length_on_every_slice(self) -> None:
        doc = multi_view_slice("d", "x" * 300)
        assert all(s.document_codepoint_length == 300 for s in doc.slices)


class TestByteAccounting:
    def test_multiscript_byte_lengths(self) -> None:
        # "a" = 1 byte, "世" = 3 bytes, "🙂" = 4 bytes
        doc = multi_view_slice("d", "a世🙂")
        for s in doc.slices:
            assert s.codepoint_length == 3
            assert s.byte_length == 8
        assert doc.byte_length == 8

    def test_byte_offsets_match_actual_encoding(self) -> None:
        text = ("a世🙂" * 100)[:250]
        doc = multi_view_slice("d", text)
        full = text.encode("utf-8")
        for s in doc.slices:
            assert s.byte_offset == len(text[: s.codepoint_offset].encode("utf-8"))
            assert full[s.byte_offset : s.byte_offset + s.byte_length] == s.text.encode("utf-8")


class TestDescendants:
    def test_full_1024_slice_has_30_descendants(self) -> None:
        doc = multi_view_slice("d", "x" * 1024)
        top = doc.slice_at(1024, 0)
        assert top is not None
        assert len(top.descendant_markers) == 16 + 8 + 4 + 2
        markers_64 = [s.own_marker for s in doc.slices_of_size(64)]
        assert all(m in top.descendant_markers for m in markers_64)

    def test_descendant_canonical_order(self) -> None:
        doc = multi_view_slice("d", "x" * 1024)
        top = doc.slice_at(1024, 0)
        assert top is not None
        expected = []
        for size in (64, 128, 256, 512):
            expected.extend(s.own_marker for s in doc.slices_of_size(size))
        assert list(top.descendant_markers) == expected

    def test_short_document_descendants_preserve_duplicates(self) -> None:
        # 100 code points: sizes 128/256/512 each produce one full-document slice with
        # identical text, hence identical markers — duplicates are preserved here
        # (dedup, when wanted, is a storage concern).
        doc = multi_view_slice("d", "x" * 100)
        top = doc.slice_at(1024, 0)
        assert top is not None
        assert len(top.descendant_markers) == 5  # 64@0, 64@64, 128@0, 256@0, 512@0
        assert len(set(top.descendant_markers)) == 3  # 64@0, 64@64, and the shared full-doc marker

    def test_smallest_size_has_no_descendants(self) -> None:
        doc = multi_view_slice("d", "x" * 300)
        assert all(s.descendant_markers == () for s in doc.slices_of_size(64))

    def test_mid_document_parent_contains_exactly_its_span(self) -> None:
        doc = multi_view_slice("d", "x" * 1200)
        parent = doc.slice_at(256, 1024)  # short region at the tail: [1024, 1200)
        assert parent is not None
        expected = [
            doc.slice_at(64, o).own_marker  # type: ignore[union-attr]
            for o in (1024, 1088, 1152)
        ] + [doc.slice_at(128, o).own_marker for o in (1024, 1152)]  # type: ignore[union-attr]
        assert list(parent.descendant_markers) == expected


class TestReconstruction:
    @pytest.mark.parametrize("text", ["x" * 1200, ("Hello 世界 🙂 " * 120)[:1100]])
    def test_lossless_at_every_size(self, text: str) -> None:
        doc = multi_view_slice("d", text)
        for size in DEFAULT_GRID.sizes:
            pieces = doc.slices_of_size(size)
            assert "".join(p.text for p in pieces) == text
            assert b"".join(p.text.encode("utf-8") for p in pieces) == text.encode("utf-8")


class TestTokenCounts:
    def test_counts_present_when_counter_supplied(self) -> None:
        doc = multi_view_slice("d", "x" * 300, token_counter=CharCounter())
        assert doc.token_counter_id == "test:chars"
        assert all(s.token_count == s.codepoint_length for s in doc.slices)

    def test_counts_absent_without_counter(self) -> None:
        doc = multi_view_slice("d", "x" * 300)
        assert doc.token_counter_id is None
        assert all(s.token_count is None for s in doc.slices)


class TestCustomGrids:
    def test_non_power_of_two_grid(self) -> None:
        grid = GridSpec((3, 6, 24))
        doc = multi_view_slice("d", "abcdefghij" * 5, grid)  # 50 cps
        assert len(doc.slices_of_size(3)) == 17
        assert len(doc.slices_of_size(6)) == 9
        assert len(doc.slices_of_size(24)) == 3
        top = doc.slice_at(24, 24)
        assert top is not None
        assert len(top.descendant_markers) == 8 + 4  # size-3 and size-6 children of [24, 48)

    def test_spine_helper(self) -> None:
        doc = multi_view_slice("d", "x" * 1024)
        spine = doc.spine()
        assert [s.size for s in spine] == [64, 128, 256, 512, 1024]
