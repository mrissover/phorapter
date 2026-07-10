"""Phorapter command-line entry point.

Subcommands (``serve``, ``mcp``, ``check``, ``eval``) are added by their milestones;
until then the CLI reports the version and what is available.
"""

from __future__ import annotations

import argparse

from phorapter import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phorapter", description="Phorapter multi-view RAG server"
    )
    parser.add_argument("--version", action="version", version=f"phorapter {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
