"""Corpus lifecycle service: create (freeze config), list, inspect, drop.

Creation is where every per-corpus decision is frozen — the grid, the embedder
pin and its probed dimension, and the token counter — exactly once, into the
store's meta storage. From then on the server reads the config back and stays
stateless. The token counter is force-materialized here so a typo'd pin cannot
be persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from phoropter.embed import create_embedder
from phoropter.errors import EmbedderError
from phoropter.grid import GridSpec
from phoropter.stores import CorpusConfig

if TYPE_CHECKING:
    from phoropter.service.core import ServiceCore

__all__ = ["CorpusInfo", "CorpusService"]


@dataclass(frozen=True, slots=True)
class CorpusInfo:
    """A corpus's frozen config plus its current counts and degradation status."""

    config: CorpusConfig
    document_count: int
    points_by_size: tuple[tuple[int, int], ...]
    degraded: tuple[str, ...]


class CorpusService:
    """Create, list, inspect, and drop corpora."""

    def __init__(self, core: ServiceCore) -> None:
        self._core = core

    async def create(
        self,
        name: str,
        *,
        grid_sizes: tuple[int, ...] | None = None,
        tokenizer: str | None = None,
        embedder_provider: str | None = None,
        embedder_model: str | None = None,
    ) -> CorpusConfig:
        """Create a corpus and freeze its configuration.

        The grid and tokenizer default to the server's configured defaults. The
        embedder defaults to the server's configured embedder; an override
        builds a fresh embedder of the named provider/model and probes its
        dimension. The tokenizer is materialized before anything is persisted.
        Raises :class:`~phoropter.errors.CorpusExistsError` if the name is taken.
        """
        settings = self._core.settings
        sizes = grid_sizes if grid_sizes is not None else settings.defaults.grid_sizes
        grid = GridSpec(sizes)  # raises GridError (INVALID_GRID) on a bad ladder
        counter_id = tokenizer if tokenizer is not None else settings.defaults.tokenizer
        self._core.materialize_counter(counter_id)  # raises UNKNOWN_TOKENIZER before persisting

        embedder = self._core.embedder
        if embedder_provider is not None or embedder_model is not None:
            provider = embedder_provider or settings.embedder.provider
            model = embedder_model or settings.embedder.model
            if provider == "fake":
                embedder = create_embedder("fake")
            elif provider == settings.embedder.provider:
                # Same provider as configured: reuse its transport, override model only.
                from phoropter.config import build_embedder

                overridden = settings.embedder.model_copy(update={"model": model})
                embedder = build_embedder(settings.model_copy(update={"embedder": overridden}))
            else:
                raise EmbedderError(
                    f"cannot create a corpus with embedder provider {provider!r}: "
                    f"the server is configured for {settings.embedder.provider!r}"
                )

        dimension = await embedder.dimension()  # probe (EmbedderError -> 503)
        config = CorpusConfig(
            name=name,
            grid=grid,
            embedder_fingerprint=embedder.fingerprint(),
            dimension=dimension,
            token_counter_id=counter_id,
        )
        await self._core.store.create_corpus(config)
        return config

    async def list_names(self) -> tuple[str, ...]:
        """Names of all corpora, sorted."""
        return await self._core.store.list_corpora()

    async def inspect(self, corpus: str) -> CorpusInfo:
        """The frozen config plus current document and per-size point counts.

        Raises :class:`~phoropter.errors.CorpusNotFoundError` for an unknown corpus.
        """
        store = self._core.store
        config = await store.get_corpus_meta(corpus)  # CORPUS_NOT_FOUND if absent
        stats = await store.corpus_stats(corpus)
        degraded = await store.verify_corpus(corpus)
        return CorpusInfo(
            config=config,
            document_count=stats.document_count,
            points_by_size=stats.points_by_size,
            degraded=tuple(degraded),
        )

    async def drop(self, corpus: str) -> None:
        """Drop a corpus and everything under it. CORPUS_NOT_FOUND for an unknown name."""
        await self._core.store.get_corpus_meta(corpus)  # raise CORPUS_NOT_FOUND, not a silent no-op
        await self._core.store.drop_corpus(corpus)
