"""The multi-size origin-aligned slicing grid.

A :class:`GridSpec` is a validated, immutable ladder of slice sizes measured in
Unicode code points. Two rules make everything downstream exact:

1. **Origin alignment** — every slice of size ``S`` starts at a code-point offset
   that is a multiple of ``S``, measured from offset 0 of the document.
2. **Divisibility chain** — each size divides every larger size.

Together they guarantee the *laminar family* property: any two grid slices of one
document are either nested or disjoint. Containment over any set of slices is
therefore a forest, a slice's ancestors are totally ordered, and every ancestor's
offset is computable in closed form — no search, no similarity heuristics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from phoropter.errors import GridError

__all__ = ["DEFAULT_GRID", "GridSpec"]


@dataclass(frozen=True, slots=True)
class GridSpec:
    """A validated ladder of slice sizes (code points), strictly ascending, each dividing the next.

    Invalid grids cannot be constructed: validation runs in ``__post_init__`` and
    raises :class:`~phoropter.errors.GridError`.
    """

    sizes: tuple[int, ...]

    def __init__(self, sizes: Sequence[int]) -> None:
        object.__setattr__(self, "sizes", tuple(sizes))
        self.__post_init__()

    def __post_init__(self) -> None:
        sizes = self.sizes
        if not sizes:
            raise GridError("grid must contain at least one size")
        for s in sizes:
            if not isinstance(s, int) or isinstance(s, bool) or s <= 0:
                raise GridError(f"grid sizes must be positive integers, got {s!r}")
        for smaller, larger in pairwise(sizes):
            if larger <= smaller:
                raise GridError(f"grid sizes must be strictly ascending: {smaller} !< {larger}")
            if larger % smaller != 0:
                raise GridError(
                    f"divisibility chain broken: {smaller} does not divide {larger}; "
                    "every grid size must divide all larger sizes"
                )

    @property
    def smallest(self) -> int:
        return self.sizes[0]

    @property
    def largest(self) -> int:
        return self.sizes[-1]

    def __contains__(self, size: object) -> bool:
        return size in self.sizes

    def parent_offset(self, child_offset: int, parent_size: int) -> int:
        """Offset of the size-``parent_size`` slice containing code point ``child_offset``."""
        return (child_offset // parent_size) * parent_size

    def levels_above(self, size: int) -> tuple[int, ...]:
        """Grid sizes strictly larger than ``size``, ascending.

        ``size`` must be a grid size.
        """
        if size not in self.sizes:
            raise GridError(f"{size} is not a size of this grid {self.sizes}")
        return tuple(s for s in self.sizes if s > size)

    def levels_below(self, size: int) -> tuple[int, ...]:
        """Grid sizes strictly smaller than ``size``, ascending.

        ``size`` must be a grid size.
        """
        if size not in self.sizes:
            raise GridError(f"{size} is not a size of this grid {self.sizes}")
        return tuple(s for s in self.sizes if s < size)

    def fingerprint(self) -> str:
        """Stable hash of the grid, persisted per corpus and checked at every upsert.

        Two stores sliced with different grids must never be mixed; comparing
        fingerprints is how adapters refuse that.
        """
        canonical = "sizes=" + ",".join(str(s) for s in self.sizes)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


DEFAULT_GRID = GridSpec((64, 128, 256, 512, 1024))
"""The default five-size grid. Sizes are code points, not bytes and not tokens."""
