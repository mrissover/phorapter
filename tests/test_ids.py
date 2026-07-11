"""Deterministic IDs: pinned vectors, namespace provenance, injective escaping."""

import uuid

from phoropter import PHOROPTER_NAMESPACE, slice_name, slice_uuid


def test_stdlib_uuid5_matches_rfc4122_appendix_vector() -> None:
    # Sanity anchor: proves the stdlib uuid5 we build on matches the RFC's own example.
    assert uuid.uuid5(uuid.NAMESPACE_DNS, "www.example.org") == uuid.UUID(
        "74738ff5-5367-5958-9aee-98fffdcd1876"
    )


def test_namespace_value_and_provenance() -> None:
    assert uuid.UUID("fabb402a-1e0f-55a8-83cd-2d0a124a3fca") == PHOROPTER_NAMESPACE
    assert uuid.uuid5(uuid.NAMESPACE_DNS, "phoropter.impluvium.software") == PHOROPTER_NAMESPACE


def test_pinned_slice_uuids() -> None:
    assert slice_uuid("doc-1", 64, 0) == uuid.UUID("e7e09240-3782-55fa-aaa9-327955e17cdc")
    assert slice_uuid("doc-1", 128, 0) == uuid.UUID("9a5a1a5e-5b61-56cc-9722-8d97e68129a7")


def test_same_coordinates_same_uuid() -> None:
    assert slice_uuid("d", 64, 128) == slice_uuid("d", 64, 128)


def test_different_coordinates_different_uuid() -> None:
    base = slice_uuid("d", 64, 0)
    assert slice_uuid("d", 64, 64) != base
    assert slice_uuid("d", 128, 0) != base
    assert slice_uuid("e", 64, 0) != base
    assert slice_uuid("d", 64, 0, corpus="c") != base


class TestEscapingInjectivity:
    def test_separator_in_document_id(self) -> None:
        # doc "a|64" at size 128 must not collide with doc "a" at size 64 (or anything else).
        assert slice_name("a|64", 128, 0) != slice_name("a", 64, 128)
        assert slice_uuid("a|64", 128, 0) != slice_uuid("a", 64, 128)

    def test_backslash_in_document_id(self) -> None:
        assert slice_name("a\\|b", 64, 0) != slice_name("a|b", 64, 0)
        assert slice_uuid("a\\|b", 64, 0) != slice_uuid("a|b", 64, 0)

    def test_corpus_vs_document_boundary(self) -> None:
        # (corpus="a", doc="b") must not collide with doc "a|b".
        assert slice_name("a|b", 64, 0) != slice_name("b", 64, 0, corpus="a")
        assert slice_uuid("a|b", 64, 0) != slice_uuid("b", 64, 0, corpus="a")

    def test_corpus_changes_uuid(self) -> None:
        assert slice_uuid("d", 64, 0, corpus="c1") != slice_uuid("d", 64, 0, corpus="c2")
