"""Marker payload codec: lossless, order-preserving round-trip."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from phoropter.markers import marker_for_text
from phoropter.stores import pack_markers, unpack_markers

hex_marker = st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)


def test_empty_tuple_round_trips() -> None:
    assert unpack_markers(pack_markers(())) == ()
    assert pack_markers(()) == ""


def test_single_marker_round_trips() -> None:
    m = (marker_for_text("hello"),)
    assert unpack_markers(pack_markers(m)) == m


def test_order_is_preserved() -> None:
    markers = tuple(marker_for_text(str(i)) for i in range(5))
    assert unpack_markers(pack_markers(markers)) == markers
    # A different order packs differently and round-trips to that order.
    reversed_markers = markers[::-1]
    assert unpack_markers(pack_markers(reversed_markers)) == reversed_markers
    assert pack_markers(markers) != pack_markers(reversed_markers)


def test_rejects_non_marker() -> None:
    with pytest.raises(ValueError, match="not a marker"):
        pack_markers(("deadbeef",))  # too short
    with pytest.raises(ValueError, match="not a marker"):
        pack_markers(("X" * 64,))  # not hex
    with pytest.raises(ValueError, match="not a marker"):
        pack_markers((marker_for_text("x").upper(),))  # uppercase not allowed


def test_unpack_rejects_corrupt_input() -> None:
    with pytest.raises(ValueError, match="not a base64"):
        unpack_markers("not*base64*")
    with pytest.raises(ValueError, match="not a multiple"):
        unpack_markers(pack_markers((marker_for_text("x"),))[:-4])  # truncated digest


@given(markers=st.lists(hex_marker, max_size=40).map(tuple))
def test_round_trip_property(markers: tuple[str, ...]) -> None:
    assert unpack_markers(pack_markers(markers)) == markers
