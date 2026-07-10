#!/usr/bin/env python
"""Shipped-content guard.

Blocks a small set of disallowed strings from entering the repository
(personal-name remnants and references to unpublished manuscripts).
Run by pre-commit with the staged file list, or with no arguments to scan
every file tracked by git.

Exits non-zero and prints each offending file:line if a pattern matches.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Patterns are assembled from fragments so this file never matches itself.
DISALLOWED: list[tuple[str, re.Pattern[str]]] = [
    ("personal name", re.compile("ris" + "sover", re.IGNORECASE)),
    ("manuscript numbering", re.compile(r"\bpaper" + r"\s*[123]\b", re.IGNORECASE)),
    ("submission venue", re.compile("acm" + r"\s+ai\s+" + "letters", re.IGNORECASE)),
    ("submission status", re.compile("under" + r"\s+" + "submission", re.IGNORECASE)),
]

SELF = Path(__file__).resolve()
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".pyc"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [Path(line) for line in out.splitlines() if line.strip()]


def check(path: Path) -> list[str]:
    if path.resolve() == SELF or path.suffix.lower() in SKIP_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, IsADirectoryError):
        return []
    problems = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, pattern in DISALLOWED:
            if pattern.search(line):
                problems.append(f"{path}:{lineno}: disallowed content ({label})")
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] if argv else tracked_files()
    problems = [p for path in paths for p in check(path)]
    if problems:
        print("\n".join(problems))
        print(f"\nshipped-content guard: {len(problems)} problem(s) found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
