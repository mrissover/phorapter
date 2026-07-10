"""Phorapter: multi-view slicing, exact containment, and token-budgeted context right-sizing.

The top-level package re-exports the curated public API of the core library.
Server components live under ``phorapter.server`` / ``phorapter.service`` and require
the ``server`` extra.
"""

from importlib.metadata import PackageNotFoundError, version

from phorapter.errors import GridError, PhorapterError, SlicingError, TokenizerError
from phorapter.forest import ContainmentAnomaly, ContainmentForest, Edge, contains
from phorapter.fusion import RankFusion, RawScorePool, TierInterleave
from phorapter.grid import DEFAULT_GRID, GridSpec
from phorapter.ids import PHORAPTER_NAMESPACE, slice_name, slice_uuid
from phorapter.markers import MARKER_HEX_LENGTH, marker_for_bytes, marker_for_text
from phorapter.model import (
    CandidateHit,
    HitProvenance,
    RetrievedHit,
    Slice,
    SlicedDocument,
    SliceRef,
)
from phorapter.selection import (
    DedupeOnly,
    EvidenceItem,
    GreedyUpwardStrategy,
    SelectedSlice,
    Selection,
    SelectionOptions,
    SelectionRequest,
    SelectionStrategy,
    SliceSource,
    budget_fit,
)
from phorapter.slicer import multi_view_slice
from phorapter.tokens import (
    DEFAULT_COUNTER_ID,
    TiktokenCounter,
    TokenCounter,
    get_counter,
    register_counter,
    registered_counter_ids,
)

try:
    __version__ = version("phorapter")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_COUNTER_ID",
    "DEFAULT_GRID",
    "MARKER_HEX_LENGTH",
    "PHORAPTER_NAMESPACE",
    "CandidateHit",
    "ContainmentAnomaly",
    "ContainmentForest",
    "DedupeOnly",
    "Edge",
    "EvidenceItem",
    "GreedyUpwardStrategy",
    "GridError",
    "GridSpec",
    "HitProvenance",
    "PhorapterError",
    "RankFusion",
    "RawScorePool",
    "RetrievedHit",
    "SelectedSlice",
    "Selection",
    "SelectionOptions",
    "SelectionRequest",
    "SelectionStrategy",
    "Slice",
    "SliceRef",
    "SliceSource",
    "SlicedDocument",
    "SlicingError",
    "TierInterleave",
    "TiktokenCounter",
    "TokenCounter",
    "TokenizerError",
    "__version__",
    "budget_fit",
    "contains",
    "get_counter",
    "marker_for_bytes",
    "marker_for_text",
    "multi_view_slice",
    "register_counter",
    "registered_counter_ids",
    "slice_name",
    "slice_uuid",
]
