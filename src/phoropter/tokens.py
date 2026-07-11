"""Token counting.

Budgets arrive in tokens; slices are measured in code points. A
:class:`TokenCounter` bridges the two. Counters are identified by a stable
``counter_id`` (e.g. ``"tiktoken:o200k_base"``) that is pinned per corpus at
creation time: per-slice counts stored at ingest are valid only under the pinned
counter, and counts from different counters are never mixed in one request.

tiktoken is imported lazily, inside functions — ``import phoropter`` works in an
environment without it. That laziness is enforced by a stripped-environment test
(importing the package with tiktoken masked); the import-linter contract forbids
every *other* third-party import here and forbids tiktoken everywhere else in
the core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from phoropter.errors import TokenizerError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DEFAULT_COUNTER_ID",
    "TiktokenCounter",
    "TokenCounter",
    "get_counter",
    "register_counter",
    "registered_counter_ids",
]

DEFAULT_COUNTER_ID = "tiktoken:o200k_base"

_TIKTOKEN_PREFIX = "tiktoken:"


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in text under one specific tokenizer."""

    counter_id: str

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """A :class:`TokenCounter` backed by a tiktoken encoding (loaded lazily on first use)."""

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self.counter_id = _TIKTOKEN_PREFIX + encoding_name
        self._encoding_name = encoding_name
        self._encoding: object | None = None

    def _load(self) -> object:
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError as e:  # pragma: no cover - environment-dependent
                raise TokenizerError(
                    f"counter {self.counter_id!r} requires the tiktoken package"
                ) from e
            try:
                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except ValueError as e:
                raise TokenizerError(f"unknown tiktoken encoding {self._encoding_name!r}") from e
        return self._encoding

    def count(self, text: str) -> int:
        encoding = self._load()
        return len(encoding.encode(text))  # type: ignore[attr-defined]


_registry: dict[str, Callable[[], TokenCounter]] = {}
_instances: dict[str, TokenCounter] = {}


def register_counter(counter_id: str, factory: Callable[[], TokenCounter]) -> None:
    """Register a counter factory under a stable id (e.g. from a plugin)."""
    _registry[counter_id] = factory
    _instances.pop(counter_id, None)


def registered_counter_ids() -> tuple[str, ...]:
    """Ids resolvable by :func:`get_counter` (explicit registrations; any
    ``tiktoken:<encoding>`` id also resolves dynamically)."""
    return tuple(sorted(set(_registry) | {DEFAULT_COUNTER_ID, "tiktoken:cl100k_base"}))


def get_counter(counter_id: str) -> TokenCounter:
    """Resolve a counter by id.

    Explicit registrations win; any ``tiktoken:<encoding>`` id resolves
    dynamically. Unknown ids raise :class:`~phoropter.errors.TokenizerError`
    listing what is available. Unknown *encodings* under the ``tiktoken:``
    prefix fail here too whenever tiktoken is importable; only in an
    environment without tiktoken is encoding validation deferred to the first
    :meth:`~TokenCounter.count` call. Broken counters are never memoized, so a
    typo'd id cannot poison the cache.
    """
    if counter_id in _instances:
        return _instances[counter_id]
    if counter_id in _registry:
        instance = _registry[counter_id]()
    elif counter_id.startswith(_TIKTOKEN_PREFIX):
        tiktoken_counter = TiktokenCounter(counter_id[len(_TIKTOKEN_PREFIX) :])
        try:
            tiktoken_counter._load()
        except TokenizerError as e:
            if not isinstance(e.__cause__, ImportError):
                raise  # unknown encoding: fail at resolve time
            # tiktoken absent: defer to first count() so core stays importable
        instance = tiktoken_counter
    else:
        known = ", ".join(registered_counter_ids())
        raise TokenizerError(f"unknown token counter {counter_id!r}; registered: {known}")
    _instances[counter_id] = instance
    return instance
