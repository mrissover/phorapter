"""Scaffold sanity: the package installs, imports, and reports a version."""

import phorapter


def test_version_is_exposed() -> None:
    assert phorapter.__version__
    assert phorapter.__version__ != "0.0.0"
