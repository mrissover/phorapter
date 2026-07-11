"""Markers: known-answer vectors and the no-normalization policy."""

import unicodedata

from phoropter import marker_for_bytes, marker_for_text


def test_sha256_empty_known_answer() -> None:
    assert (
        marker_for_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_abc_known_answer() -> None:
    assert (
        marker_for_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_x64_anchor() -> None:
    # Pinned anchor shared with the cross-implementation parity fixtures.
    assert (
        marker_for_text("x" * 64)
        == "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c"
    )


def test_marker_is_lowercase_hex() -> None:
    m = marker_for_text("hello")
    assert len(m) == 64
    assert m == m.lower()
    int(m, 16)


def test_no_unicode_normalization() -> None:
    composed = "é"  # é as one code point (NFC)
    decomposed = unicodedata.normalize("NFD", composed)  # e + combining accent
    assert composed != decomposed
    assert marker_for_text(composed) != marker_for_text(decomposed)


def test_multibyte_hashes_bytes_not_codepoints() -> None:
    assert marker_for_text("世") == marker_for_bytes("世".encode())
