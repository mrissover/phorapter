"""The service core: the async facade shared by the REST and MCP surfaces.

:class:`ServiceCore` owns the store adapter, the embedder, and the settings, and
exposes the three domain services (:class:`~phoropter.service.corpora.CorpusService`,
:class:`~phoropter.service.documents.DocumentService`,
:class:`~phoropter.service.query.QueryService`). Every I/O the surfaces need
flows through it; the surfaces themselves hold no business logic.

Startup validation (:meth:`ServiceCore.startup`) pings the store, bootstraps its
shared infrastructure, and probes the embedder's dimension so a misconfigured
deployment fails loudly at boot rather than on the first request.

The query path bridges async I/O to the synchronous, pure selection engine: it
prefetches the ancestor closure of the packable candidates by deterministic id
(one :meth:`~phoropter.stores.VectorStoreAdapter.fetch_by_ids` per size) and
hands the engine a fixed, in-memory :class:`~phoropter.selection.SliceSource`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phoropter.embed import Embedder
from phoropter.errors import TokenizerError
from phoropter.model import Slice, SliceRef
from phoropter.tokens import get_counter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from phoropter.config import Settings
    from phoropter.service.corpora import CorpusService
    from phoropter.service.documents import DocumentService
    from phoropter.service.query import QueryService
    from phoropter.stores import VectorStoreAdapter

__all__ = ["PrefetchSource", "ServiceCore", "StartupReport"]


class PrefetchSource:
    """A synchronous :class:`~phoropter.selection.SliceSource` over a fixed ref map.

    The query service prefetches the ancestor closure of the packable candidates
    (async, by deterministic id) and wraps the result in one of these. The pure
    engine then resolves trade-up ancestors from it without touching the network.
    Unknown refs are simply omitted — the engine records a fetch miss and moves on.
    """

    def __init__(self, slices: dict[SliceRef, Slice]) -> None:
        self._slices = slices

    def get_slices(self, refs: Iterable[SliceRef], *, corpus: str) -> dict[SliceRef, Slice]:
        return {ref: self._slices[ref] for ref in refs if ref in self._slices}


class StartupReport:
    """Outcome of :meth:`ServiceCore.startup` — store reachability and the probed dimension."""

    def __init__(
        self, *, store_ok: bool, embedder_dimension: int | None, detail: str | None
    ) -> None:
        self.store_ok = store_ok
        self.embedder_dimension = embedder_dimension
        self.detail = detail

    @property
    def ok(self) -> bool:
        return self.store_ok and self.embedder_dimension is not None


class ServiceCore:
    """Holds the store adapter, the embedder, and the settings, and wires the services.

    Construct with an already-built store and embedder (see
    :func:`phoropter.config.build_store` / :func:`phoropter.config.build_embedder`),
    then call :meth:`startup` once before serving.
    """

    def __init__(
        self, *, store: VectorStoreAdapter, embedder: Embedder, settings: Settings
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    @property
    def corpora(self) -> CorpusService:
        from phoropter.service.corpora import CorpusService

        return CorpusService(self)

    @property
    def documents(self) -> DocumentService:
        from phoropter.service.documents import DocumentService

        return DocumentService(self)

    @property
    def query(self) -> QueryService:
        from phoropter.service.query import QueryService

        return QueryService(self)

    # ── startup / shutdown ───────────────────────────────────────────────────

    async def startup(self) -> StartupReport:
        """Ping the store, bootstrap it, and probe the embedder dimension.

        Never raises on a dependency being down — the readiness endpoint reports
        the outcome, and ``phoropter check`` turns it into an exit code. A store
        that is up is bootstrapped (idempotent); a store that is down is not.
        """
        detail: str | None = None
        store_ok = False
        try:
            store_ok = await self.store.ping()
        except Exception as e:  # a raising ping is treated as unreachable
            detail = f"store ping failed: {e}"
        if store_ok:
            try:
                await self.store.bootstrap()
            except Exception as e:
                store_ok = False
                detail = f"store bootstrap failed: {e}"

        embedder_dimension: int | None = None
        try:
            embedder_dimension = await self.embedder.dimension()
        except Exception as e:
            detail = (detail + "; " if detail else "") + f"embedder probe failed: {e}"

        return StartupReport(
            store_ok=store_ok, embedder_dimension=embedder_dimension, detail=detail
        )

    async def aclose(self) -> None:
        """Release adapter resources (HTTP clients, connection pools) if they expose it."""
        for target in (self.store, self.embedder):
            closer = getattr(target, "aclose", None)
            if closer is not None:
                await closer()

    # ── shared helpers ───────────────────────────────────────────────────────

    def materialize_counter(self, counter_id: str) -> None:
        """Force-materialize a token counter, raising if the id is unusable.

        Called at corpus creation so a typo'd tokenizer pin can never be frozen
        into a :class:`~phoropter.stores.CorpusConfig`: an id that resolves but
        cannot count (an unknown tiktoken encoding surfacing only on first use)
        is caught here and surfaces as ``UNKNOWN_TOKENIZER`` (HTTP 422).
        """
        counter = get_counter(counter_id)  # raises TokenizerError on an unknown id
        try:
            counter.count("")
        except TokenizerError:
            raise
        except Exception as e:  # a backend that resolves but cannot encode
            raise TokenizerError(
                f"token counter {counter_id!r} could not be materialized: {e}"
            ) from e
