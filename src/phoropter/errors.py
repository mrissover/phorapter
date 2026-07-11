"""Phoropter error hierarchy.

Every error carries a stable machine-readable ``code`` that surfaces unchanged
through the REST error envelope and MCP error messages, so clients can act on
codes rather than parse prose.
"""

from __future__ import annotations


class PhoropterError(Exception):
    """Base class for all phoropter errors."""

    code: str = "INTERNAL"


class GridError(PhoropterError):
    """The slicing grid is invalid (not ascending, or the divisibility chain is broken)."""

    code = "INVALID_GRID"


class SlicingError(PhoropterError):
    """A document cannot be sliced (missing id, non-text input, or unencodable content)."""

    code = "SLICING_ERROR"


class TokenizerError(PhoropterError):
    """The requested token counter is unknown or unavailable."""

    code = "UNKNOWN_TOKENIZER"


class StoreError(PhoropterError):
    """The vector store backend is unreachable or failed to execute an operation."""

    code = "STORE_UNAVAILABLE"


class CorpusNotFoundError(PhoropterError):
    """The named corpus does not exist in the store."""

    code = "CORPUS_NOT_FOUND"


class CorpusExistsError(PhoropterError):
    """A corpus with this name already exists; corpus configuration is frozen at creation."""

    code = "CORPUS_EXISTS"


class CorpusMismatchError(PhoropterError):
    """Data offered to a corpus does not match its pinned configuration.

    Raised when an upsert's grid fingerprint or vector dimension disagrees with
    what the corpus was created with. There is never a silent migration path:
    changing the grid or the embedding model means creating a new corpus and
    reindexing.
    """

    code = "EMBEDDER_MISMATCH"


class DocumentNotFoundError(PhoropterError):
    """The named document is not registered in the corpus."""

    code = "DOCUMENT_NOT_FOUND"


class EmbedderError(PhoropterError):
    """The embedding provider is unreachable, misconfigured, or returned an unusable response."""

    code = "EMBEDDER_UNAVAILABLE"
