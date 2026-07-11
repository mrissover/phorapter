"""GridSpec: validation, closed-form arithmetic, fingerprints."""

import pytest

from phoropter import DEFAULT_GRID, GridError, GridSpec


class TestValidation:
    def test_default_grid(self) -> None:
        assert DEFAULT_GRID.sizes == (64, 128, 256, 512, 1024)
        assert DEFAULT_GRID.smallest == 64
        assert DEFAULT_GRID.largest == 1024

    def test_accepts_any_sequence(self) -> None:
        assert GridSpec([32, 64]).sizes == (32, 64)

    def test_single_size_is_valid(self) -> None:
        assert GridSpec((100,)).sizes == (100,)

    def test_non_power_of_two_chain_is_valid(self) -> None:
        # Nothing requires powers of two — only the divisibility chain.
        assert GridSpec((3, 6, 24, 48)).sizes == (3, 6, 24, 48)

    def test_empty_rejected(self) -> None:
        with pytest.raises(GridError):
            GridSpec(())

    @pytest.mark.parametrize("bad", [0, -64])
    def test_non_positive_rejected(self, bad: int) -> None:
        with pytest.raises(GridError):
            GridSpec((bad, 128))

    def test_non_ascending_rejected(self) -> None:
        with pytest.raises(GridError):
            GridSpec((128, 64))

    def test_duplicate_rejected(self) -> None:
        with pytest.raises(GridError):
            GridSpec((64, 64))

    def test_broken_divisibility_chain_rejected(self) -> None:
        with pytest.raises(GridError):
            GridSpec((64, 96))

    def test_bool_rejected(self) -> None:
        with pytest.raises(GridError):
            GridSpec((True, 2))  # type: ignore[arg-type]


class TestArithmetic:
    def test_parent_offset(self) -> None:
        g = DEFAULT_GRID
        assert g.parent_offset(0, 128) == 0
        assert g.parent_offset(64, 128) == 0
        assert g.parent_offset(128, 128) == 128
        assert g.parent_offset(1088, 512) == 1024
        assert g.parent_offset(1088, 1024) == 1024

    def test_levels_above_and_below(self) -> None:
        g = DEFAULT_GRID
        assert g.levels_above(64) == (128, 256, 512, 1024)
        assert g.levels_above(1024) == ()
        assert g.levels_below(64) == ()
        assert g.levels_below(1024) == (64, 128, 256, 512)

    def test_levels_require_grid_size(self) -> None:
        with pytest.raises(GridError):
            DEFAULT_GRID.levels_above(100)
        with pytest.raises(GridError):
            DEFAULT_GRID.levels_below(100)

    def test_contains(self) -> None:
        assert 64 in DEFAULT_GRID
        assert 100 not in DEFAULT_GRID


class TestFingerprint:
    def test_stable(self) -> None:
        assert GridSpec((64, 128)).fingerprint() == GridSpec([64, 128]).fingerprint()

    def test_distinguishes_grids(self) -> None:
        assert GridSpec((64, 128)).fingerprint() != GridSpec((64, 256)).fingerprint()

    def test_is_hex_sha256(self) -> None:
        fp = DEFAULT_GRID.fingerprint()
        assert len(fp) == 64
        int(fp, 16)
