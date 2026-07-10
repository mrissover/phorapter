"""Embedder SPI: :class:`Embedder`, provider registry, and the deterministic test embedder.

Phorapter never bakes in an embedding model. A corpus pins an embedder as
``"provider:model"`` at creation (:meth:`Embedder.fingerprint`) together with
the probed vector dimension; from then on, data embedded by anything else is
refused. Providers are registered under the ``phorapter.embedders``
entry-point group (see docs/adapters.md).

This module is dependency-free on purpose — the shipped HTTP providers
(:mod:`phorapter.embed.ollama`, :mod:`phorapter.embed.openai_compat`) live in
their own modules and are imported lazily by the registry, so
:class:`FakeEmbedder` and the SPI itself work without the ``server`` extra.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from phorapter.errors import EmbedderError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

__all__ = [
    "Embedder",
    "EmbedderHealth",
    "EmbedderRegistry",
    "FakeEmbedder",
    "create_embedder",
    "default_registry",
]

_PROBE_TEXT = "phorapter dimension probe"

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class EmbedderHealth:
    """Result of an embedder health check; ``detail`` explains a failure."""

    ok: bool
    provider: str
    model: str
    dimension: int | None = None
    detail: str | None = None


class Embedder(ABC):
    """The embedder SPI.

    Subclasses set ``provider`` and ``model`` and implement :meth:`embed`,
    which must preserve input order, return one vector per text, and take care
    of its own batching and retries — callers hand over the full text list and
    never see transport mechanics. Failures surface as
    :class:`~phorapter.errors.EmbedderError`.
    """

    provider: str
    model: str

    def __init__(self) -> None:
        self._dimension: int | None = None

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input text, in input order."""

    async def dimension(self) -> int:
        """The embedding dimension, probed once with a fixed text and cached.

        The probe is how corpus creation learns the dimension to pin — no
        provider metadata endpoint is trusted over the shape of an actual
        embedding.
        """
        if self._dimension is None:
            vectors = await self.embed([_PROBE_TEXT])
            if not vectors or not vectors[0]:
                raise EmbedderError(
                    f"embedder {self.fingerprint()!r} returned an empty probe embedding"
                )
            self._dimension = len(vectors[0])
        return self._dimension

    async def health(self) -> EmbedderHealth:
        """Probe the provider; never raises — failure is data, not an exception."""
        try:
            dim = await self.dimension()
        except Exception as e:  # health checks report failures, they do not propagate them
            return EmbedderHealth(ok=False, provider=self.provider, model=self.model, detail=str(e))
        return EmbedderHealth(ok=True, provider=self.provider, model=self.model, dimension=dim)

    def fingerprint(self) -> str:
        """The ``"provider:model"`` string pinned into a corpus at creation."""
        return f"{self.provider}:{self.model}"


class FakeEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests and offline development.

    Vectors are hash-derived unit vectors: the text's SHA-256 seeds a counter
    of digest blocks whose bytes become components in ``[-1, 1)``, normalized
    to unit length. The same text yields the same vector on every platform and
    every run; different texts yield (nearly always) different directions.
    There is no semantic signal — only determinism.
    """

    provider = "fake"

    def __init__(self, dimension: int = 32, *, model: str | None = None) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.model = model if model is not None else f"deterministic-{dimension}"
        self._dimension = dimension

    def _vector(self, text: str) -> list[float]:
        assert self._dimension is not None
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        components: list[float] = []
        counter = 0
        while len(components) < self._dimension:
            block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(block), 4):
                value = int.from_bytes(block[i : i + 4], "big")
                components.append(value / 2**31 - 1.0)
            counter += 1
        components = components[: self._dimension]
        norm = math.sqrt(sum(c * c for c in components))
        if norm == 0.0:  # pragma: no cover - hash output is never all-zero in practice
            components[0] = 1.0
            norm = 1.0
        return [c / norm for c in components]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class EmbedderRegistry:
    """Maps provider names to embedder factories.

    ``create(name, **kwargs)`` builds an embedder; unknown names raise
    :class:`~phorapter.errors.EmbedderError` listing what is registered.
    Third-party providers arrive either by explicit :meth:`register` calls or
    via :meth:`load_entry_points` (group ``phorapter.embedders``).
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Embedder]] = {}

    def register(self, provider: str, factory: Callable[..., Embedder]) -> None:
        self._factories[provider] = factory

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, provider: str, **kwargs: Any) -> Embedder:
        factory = self._factories.get(provider)
        if factory is None:
            known = ", ".join(self.providers()) or "(none)"
            raise EmbedderError(f"unknown embedder provider {provider!r}; registered: {known}")
        return factory(**kwargs)

    def load_entry_points(self) -> int:
        """Register every factory in the ``phorapter.embedders`` entry-point group.

        Returns the number of entry points loaded. Later registrations win, so
        a plugin can deliberately shadow a builtin name.
        """
        from importlib.metadata import entry_points

        count = 0
        for ep in entry_points(group="phorapter.embedders"):
            self.register(ep.name, ep.load())
            count += 1
        return count


def _ollama_factory(**kwargs: Any) -> Embedder:
    from phorapter.embed.ollama import OllamaEmbedder

    return OllamaEmbedder(**kwargs)


def _openai_compat_factory(**kwargs: Any) -> Embedder:
    from phorapter.embed.openai_compat import OpenAICompatEmbedder

    return OpenAICompatEmbedder(**kwargs)


default_registry = EmbedderRegistry()
"""The process-wide registry, pre-seeded with the shipped providers."""

default_registry.register("fake", FakeEmbedder)
default_registry.register("ollama", _ollama_factory)
default_registry.register("openai_compat", _openai_compat_factory)


def create_embedder(provider: str, **kwargs: Any) -> Embedder:
    """Build an embedder from :data:`default_registry`."""
    return default_registry.create(provider, **kwargs)


# ── shared helpers for HTTP providers (stdlib-only) ─────────────────────────


async def _with_retries(
    attempt: Callable[[], Awaitable[_T]],
    *,
    attempts: int,
    base_delay: float,
    retryable: Callable[[Exception], bool],
) -> _T:
    """Run ``attempt`` with exponential backoff; non-retryable errors raise immediately."""
    delay = base_delay
    for remaining in range(attempts - 1, -1, -1):
        try:
            return await attempt()
        except Exception as e:
            if remaining == 0 or not retryable(e):
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


async def _map_batches(
    texts: Sequence[str],
    *,
    batch_size: int,
    max_concurrency: int,
    call: Callable[[list[str]], Awaitable[list[list[float]]]],
) -> list[list[float]]:
    """Embed ``texts`` in batches with bounded concurrency, preserving input order."""
    batches = [list(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await call(batch)

    results = await asyncio.gather(*(bounded(batch) for batch in batches))
    return [vector for batch_vectors in results for vector in batch_vectors]
