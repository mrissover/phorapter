"""Ollama embedding provider (requires httpx, shipped with the ``server`` extra).

Talks to Ollama's batch endpoint ``POST /api/embed`` (``input`` may be a list);
if a server predates it (404), the adapter falls back to the legacy
single-prompt ``POST /api/embeddings`` and remembers the downgrade for the
rest of its lifetime. Transient failures (connection errors, 5xx, 429) retry
with exponential backoff; anything else fails fast as
:class:`~phorapter.errors.EmbedderError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import httpx
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError('OllamaEmbedder requires httpx: pip install "phorapter[server]"') from e

from phorapter.embed import Embedder, _map_batches, _with_retries
from phorapter.errors import EmbedderError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OllamaEmbedder"]


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or status >= 500
    return isinstance(error, httpx.TransportError)


class OllamaEmbedder(Embedder):
    """:class:`~phorapter.embed.Embedder` over a local or remote Ollama server.

    ``transport`` exists for tests (``httpx.MockTransport``); production use
    leaves it ``None``. Call :meth:`aclose` when done — the HTTP client is
    created lazily and reused across calls.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        max_concurrency: int = 4,
        timeout_s: float = 60.0,
        retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_base_delay = retry_base_delay
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._use_legacy_endpoint = False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await _map_batches(
            texts,
            batch_size=self._batch_size,
            max_concurrency=self._max_concurrency,
            call=self._embed_batch,
        )

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            return await _with_retries(
                lambda: self._request_batch(batch),
                attempts=self._retry_attempts,
                base_delay=self._retry_base_delay,
                retryable=_is_retryable,
            )
        except httpx.HTTPError as e:
            raise EmbedderError(f"ollama embedding via {self.fingerprint()!r} failed: {e}") from e

    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        if not self._use_legacy_endpoint:
            response = await self._http().post(
                "/api/embed", json={"model": self.model, "input": batch}
            )
            if response.status_code != 404:
                response.raise_for_status()
                return self._extract(response.json(), "embeddings", expected=len(batch))
            # A 404 here means the batch endpoint does not exist on this
            # server; fall back to the legacy per-prompt endpoint permanently.
            self._use_legacy_endpoint = True
        vectors: list[list[float]] = []
        for text in batch:
            response = await self._http().post(
                "/api/embeddings", json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            body = response.json()
            embedding = body.get("embedding") if isinstance(body, dict) else None
            if not isinstance(embedding, list):
                raise EmbedderError(
                    f"ollama returned a malformed legacy embedding response "
                    f"for {self.fingerprint()!r}"
                )
            vectors.append([float(component) for component in embedding])
        return vectors

    def _extract(self, body: Any, key: str, *, expected: int) -> list[list[float]]:
        embeddings = body.get(key) if isinstance(body, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbedderError(
                f"ollama returned a malformed embedding response for {self.fingerprint()!r}: "
                f"expected {expected} vectors under {key!r}"
            )
        return [[float(component) for component in vector] for vector in embeddings]
