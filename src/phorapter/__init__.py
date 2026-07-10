"""Phorapter: multi-view slicing, exact containment, and token-budgeted context right-sizing.

The top-level package re-exports the curated public API of the core library.
Server components live under ``phorapter.server`` / ``phorapter.service`` and require
the ``server`` extra.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("phorapter")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
