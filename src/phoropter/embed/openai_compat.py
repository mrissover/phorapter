"""OpenAI-compatible embedding provider (requires httpx, shipped with the ``server`` extra).

Talks to any service exposing the OpenAI embeddings contract —
``POST {base_url}/v1/embeddings`` with a bearer token, ``{"model", "input": [...]}``
in and ``{"data": [{"index", "embedding"}, ...]}`` out. That covers OpenAI
itself, vLLM, text-embeddings-inference, LM Studio, and similar. Transient
failures (connection errors, 5xx, 429) retry with exponential backoff; anything
else fails fast as :class:`~phoropter.errors.EmbedderError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import httpx
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError('OpenAICompatEmbedder requires httpx: pip install "phoropter[server]"') from e

from phoropter.embed import Embedder, _map_batches, _with_retries
from phoropter.errors import EmbedderError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["OpenAICompatEmbedder"]


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or status >= 500
    return isinstance(error, httpx.TransportError)


class OpenAICompatEmbedder(Embedder):
    """:class:`~phoropter.embed.Embedder` over any OpenAI-compatible embeddings endpoint.

    ``api_key`` is sent as a bearer token; some local servers ignore it, so it
    is optional. ``transport`` exists for tests (``httpx.MockTransport``);
    production use leaves it ``None``. Call :meth:`aclose` when done — the HTTP
    client is created lazily and reused across calls.
    """

    provider = "openai_compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com",
        api_key: str | None = None,
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
        self._api_key = api_key
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._retry_base_delay = retry_base_delay
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                headers=headers,
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
            raise EmbedderError(
                f"openai-compatible embedding via {self.fingerprint()!r} failed: {e}"
            ) from e

    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._http().post(
            "/v1/embeddings", json={"model": self.model, "input": batch}
        )
        response.raise_for_status()
        return self._extract(response.json(), expected=len(batch))

    def _extract(self, body: Any, *, expected: int) -> list[list[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            raise EmbedderError(
                f"openai-compatible endpoint returned a malformed response for "
                f"{self.fingerprint()!r}: expected {expected} vectors under 'data'"
            )
        # The contract indexes each embedding; sort by it rather than trusting order.
        try:
            ordered = sorted(data, key=lambda item: item["index"])
        except (KeyError, TypeError) as e:
            raise EmbedderError(
                f"openai-compatible endpoint omitted embedding indices for {self.fingerprint()!r}"
            ) from e
        vectors: list[list[float]] = []
        for item in ordered:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise EmbedderError(
                    f"openai-compatible endpoint returned a malformed embedding for "
                    f"{self.fingerprint()!r}"
                )
            vectors.append([float(component) for component in embedding])
        return vectors
