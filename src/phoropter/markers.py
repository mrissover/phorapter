"""Structural markers: content addressing for slices.

A slice's **marker** is the SHA-256 hash of the exact UTF-8 bytes of its text,
formatted as 64 lowercase hex characters.

**No-normalization policy (normative).** The hash is computed over the bytes *as
given*: no Unicode normalization (NFC/NFD), no newline canonicalization, no BOM
stripping — ever, at any layer. Three reasons:

1. **Reproducibility across ecosystems.** Any language that can UTF-8-encode a
   string computes the identical marker; normalization variants would fork on
   library versions and settings.
2. **Lossless reconstruction.** Slice texts must concatenate back to the exact
   document; a normalizing hash would claim equality between slices whose bytes
   differ.
3. **Exactness.** "Equal markers ⇒ byte-identical content" is what makes
   containment detection exact rather than approximate. Normalization would
   dilute that to "canonically equivalent", which is a different (and weaker)
   claim.

Callers who want normalized text normalize *before* ingest and own that choice.
"""

from __future__ import annotations

import hashlib

__all__ = ["MARKER_HEX_LENGTH", "marker_for_bytes", "marker_for_text"]

MARKER_HEX_LENGTH = 64
"""Length of a marker string: SHA-256 as lowercase hex."""


def marker_for_bytes(data: bytes) -> str:
    """SHA-256 of ``data`` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def marker_for_text(text: str) -> str:
    """SHA-256 of the exact UTF-8 encoding of ``text`` (no normalization of any kind)."""
    return marker_for_bytes(text.encode("utf-8"))
