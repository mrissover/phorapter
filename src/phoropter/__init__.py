"""Phoropter: multi-view slicing, exact containment, and token-budgeted context right-sizing.

The top-level package re-exports the curated public API of the core library.
Server components live under ``phoropter.server`` / ``phoropter.service`` and require
the ``server`` extra.
"""

from importlib.metadata import PackageNotFoundError, version

from phoropter.errors import GridError, PhoropterError, SlicingError, TokenizerError
from phoropter.forest import ContainmentAnomaly, ContainmentForest, Edge, contains
from phoropter.fusion import RankFusion, RawScorePool, TierInterleave
from phoropter.grid import DEFAULT_GRID, GridSpec
from phoropter.ids import PHOROPTER_NAMESPACE, slice_name, slice_uuid
from phoropter.markers import MARKER_HEX_LENGTH, marker_for_bytes, marker_for_text
from phoropter.model import (
    CandidateHit,
    HitProvenance,
    RetrievedHit,
    Slice,
    SlicedDocument,
    SliceRef,
)
from phoropter.selection import (
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
from phoropter.slicer import multi_view_slice
from phoropter.tokens import (
    DEFAULT_COUNTER_ID,
    TiktokenCounter,
    TokenCounter,
    get_counter,
    register_counter,
    registered_counter_ids,
)

try:
    __version__ = version("phoropter")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_COUNTER_ID",
    "DEFAULT_GRID",
    "MARKER_HEX_LENGTH",
    "PHOROPTER_NAMESPACE",
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
    "PhoropterError",
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
