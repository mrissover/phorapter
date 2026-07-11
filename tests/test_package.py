"""Scaffold sanity: the package installs, imports, and reports a version."""

import subprocess
import sys
import textwrap

import phoropter


def test_version_is_exposed() -> None:
    assert phoropter.__version__
    assert phoropter.__version__ != "0.0.0"


def test_core_imports_without_tiktoken() -> None:
    """The core must import and run in a stripped environment.

    tiktoken is masked in a subprocess; importing phoropter must succeed, and
    only the first count() on a tiktoken-backed counter may fail (with
    TokenizerError). This is the enforcement of the lazy-import policy — the
    import-linter contract alone cannot catch a hoisted module-level import.
    """
    program = textwrap.dedent(
        """
        import sys
        sys.modules["tiktoken"] = None  # mask: any `import tiktoken` now fails

        import phoropter
        from phoropter import GridSpec, get_counter, multi_view_slice
        from phoropter.errors import TokenizerError

        doc = multi_view_slice("d", "hello world", GridSpec((4, 8)))
        assert len(doc.slices) > 0

        counter = get_counter("tiktoken:o200k_base")  # resolves; validation deferred
        try:
            counter.count("x")
        except TokenizerError:
            pass
        else:
            raise SystemExit("count() should have failed without tiktoken")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
