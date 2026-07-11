"""The multi-view slicer.

Slices a document at every grid size on the shared origin-aligned grid, computes
each slice's marker, and assigns descendant markers by closed-form offset
arithmetic. Python strings are natively code-point indexed, so ``text[o:o+s]``
*is* a code-point slice.

**Encode-once construction.** The document is UTF-8-encoded exactly once, as the
sequence of smallest-size spans. Every larger slice's bytes are the physical
concatenation of a contiguous run of those spans (grid boundaries at every size
are multiples of the smallest size, by the divisibility chain). Markers are
hashed incrementally over the runs, so the byte prefix property along the
left-anchored spine holds *by construction of the implementation*, not merely by
the math.

**Descendants are enumeration, not search.** For a parent slice covering
``[O, E)``, the contained smaller-size slices at size ``S'`` are exactly those at
offsets ``range(O, E, S')`` — all of them. Origin alignment plus the divisibility
chain guarantee each one ends at or before ``E``.
"""

from __future__ import annotations

import hashlib
from itertools import accumulate
from typing import TYPE_CHECKING

from phoropter.errors import SlicingError
from phoropter.grid import DEFAULT_GRID, GridSpec
from phoropter.model import Slice, SlicedDocument

if TYPE_CHECKING:
    from phoropter.tokens import TokenCounter

__all__ = ["multi_view_slice"]


def multi_view_slice(
    document_id: str,
    text: str,
    grid: GridSpec = DEFAULT_GRID,
    *,
    token_counter: TokenCounter | None = None,
) -> SlicedDocument:
    """Slice ``text`` at every size of ``grid``, with markers and descendant markers.

    Returns a :class:`~phoropter.model.SlicedDocument` whose ``slices`` are in
    canonical order (size ascending, then offset ascending). Empty text yields
    zero slices. Text that cannot be UTF-8-encoded (lone surrogates) is rejected:
    such a document has no well-defined markers.

    When ``token_counter`` is provided, each slice carries its token count under
    that counter, and the counter's id is recorded on the document.
    """
    if not document_id:
        raise SlicingError("document_id must be a non-empty string")
    if not isinstance(text, str):
        raise SlicingError(f"text must be str, got {type(text).__name__}")

    n = len(text)  # len(str) is the code-point count
    counter_id = token_counter.counter_id if token_counter is not None else None
    if n == 0:
        return SlicedDocument(
            document_id=document_id,
            text=text,
            codepoint_length=0,
            byte_length=0,
            grid=grid,
            token_counter_id=counter_id,
            slices=(),
        )

    s_min = grid.smallest
    spans: list[bytes] = []
    for i in range(0, n, s_min):
        try:
            spans.append(text[i : i + s_min].encode("utf-8"))
        except UnicodeEncodeError as e:
            raise SlicingError(
                f"document {document_id!r} is not UTF-8-encodable "
                f"(lone surrogate near code point {i + e.start}); "
                "markers are undefined for such text"
            ) from e

    # byte_prefix[j] = UTF-8 byte offset of code point j * s_min; final entry = total bytes.
    byte_prefix = list(accumulate((len(s) for s in spans), initial=0))
    total_bytes = byte_prefix[-1]

    # Pass 1: geometry and markers for every (size, offset).
    markers: dict[int, dict[int, str]] = {size: {} for size in grid.sizes}
    geometry: list[tuple[int, int, int, int, int]] = []  # (size, O, E, byte_off, byte_len)
    for size in grid.sizes:
        by_offset = markers[size]
        for offset in range(0, n, size):
            end = min(offset + size, n)
            # Slice boundaries are multiples of s_min except a clipped end, which is n.
            j0 = offset // s_min
            j1 = (end + s_min - 1) // s_min
            digest = hashlib.sha256()
            for span in spans[j0:j1]:
                digest.update(span)
            by_offset[offset] = digest.hexdigest()
            geometry.append((size, offset, end, byte_prefix[j0], byte_prefix[j1] - byte_prefix[j0]))

    # Pass 2: materialize slices with descendant markers (canonical order:
    # child size ascending, then offset ascending).
    slices: list[Slice] = []
    for size, offset, end, byte_off, byte_len in geometry:
        descendants: list[str] = []
        for child_size in grid.levels_below(size):
            child_markers = markers[child_size]
            descendants.extend(child_markers[o] for o in range(offset, end, child_size))
        slice_text = text[offset:end]
        slices.append(
            Slice(
                document_id=document_id,
                size=size,
                codepoint_offset=offset,
                codepoint_length=end - offset,
                byte_offset=byte_off,
                byte_length=byte_len,
                document_codepoint_length=n,
                text=slice_text,
                own_marker=markers[size][offset],
                descendant_markers=tuple(descendants),
                token_count=token_counter.count(slice_text) if token_counter is not None else None,
            )
        )

    return SlicedDocument(
        document_id=document_id,
        text=text,
        codepoint_length=n,
        byte_length=total_bytes,
        grid=grid,
        token_counter_id=counter_id,
        slices=tuple(slices),
    )
