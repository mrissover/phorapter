"""Embedders: FakeEmbedder determinism, HTTP providers against a mock transport."""

import httpx
import pytest

from phorapter.embed import FakeEmbedder, create_embedder, default_registry
from phorapter.embed.ollama import OllamaEmbedder
from phorapter.embed.openai_compat import OpenAICompatEmbedder
from phorapter.errors import EmbedderError

# asyncio_mode = "auto" (pyproject) runs async test functions without per-test marks.


class TestFakeEmbedder:
    async def test_deterministic(self) -> None:
        e = FakeEmbedder(dimension=16)
        a = await e.embed(["hello", "world"])
        b = await e.embed(["hello", "world"])
        assert a == b

    async def test_dimension_and_unit_norm(self) -> None:
        e = FakeEmbedder(dimension=24)
        assert await e.dimension() == 24
        (vector,) = await e.embed(["something"])
        assert len(vector) == 24
        norm = sum(c * c for c in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9

    async def test_distinct_texts_distinct_vectors(self) -> None:
        e = FakeEmbedder(dimension=32)
        a, b = await e.embed(["alpha", "beta"])
        assert a != b

    async def test_fingerprint(self) -> None:
        assert FakeEmbedder(dimension=8).fingerprint() == "fake:deterministic-8"
        assert FakeEmbedder(dimension=8, model="custom").fingerprint() == "fake:custom"

    def test_registered_in_default_registry(self) -> None:
        assert "fake" in default_registry.providers()
        assert isinstance(create_embedder("fake", dimension=4), FakeEmbedder)


def _handler(record: list[httpx.Request], responses: list[httpx.Response]):
    calls = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        record.append(request)
        return next(calls)

    return handle


class TestOllamaEmbedder:
    async def test_batch_endpoint(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(requests, [httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})])
        )
        e = OllamaEmbedder("nomic-embed-text", transport=transport)
        vectors = await e.embed(["a", "b"])
        assert vectors == [[1.0, 0.0], [0.0, 1.0]]
        assert requests[0].url.path == "/api/embed"
        await e.aclose()

    async def test_falls_back_to_legacy_endpoint_on_404(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(
                requests,
                [
                    httpx.Response(404),
                    httpx.Response(200, json={"embedding": [1.0, 2.0]}),
                    httpx.Response(200, json={"embedding": [3.0, 4.0]}),
                ],
            )
        )
        e = OllamaEmbedder("m", transport=transport, batch_size=8)
        vectors = await e.embed(["a", "b"])
        assert vectors == [[1.0, 2.0], [3.0, 4.0]]
        assert [r.url.path for r in requests] == [
            "/api/embed",
            "/api/embeddings",
            "/api/embeddings",
        ]
        await e.aclose()

    async def test_retry_then_success(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(
                requests,
                [
                    httpx.Response(503),
                    httpx.Response(200, json={"embeddings": [[1.0]]}),
                ],
            )
        )
        e = OllamaEmbedder("m", transport=transport, retry_base_delay=0.0)
        vectors = await e.embed(["a"])
        assert vectors == [[1.0]]
        assert len(requests) == 2
        await e.aclose()

    async def test_malformed_response_raises(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"nope": 1}))
        e = OllamaEmbedder("m", transport=transport)
        with pytest.raises(EmbedderError):
            await e.embed(["a"])
        await e.aclose()

    async def test_empty_input_no_calls(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(_handler(requests, []))
        e = OllamaEmbedder("m", transport=transport)
        assert await e.embed([]) == []
        assert requests == []
        await e.aclose()


class TestOpenAICompatEmbedder:
    async def test_embed_and_auth_header(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(
                requests,
                [
                    httpx.Response(
                        200,
                        json={
                            "data": [
                                {"index": 0, "embedding": [1.0, 0.0]},
                                {"index": 1, "embedding": [0.0, 1.0]},
                            ]
                        },
                    )
                ],
            )
        )
        e = OpenAICompatEmbedder("text-embedding-3-small", api_key="sk-test", transport=transport)
        vectors = await e.embed(["a", "b"])
        assert vectors == [[1.0, 0.0], [0.0, 1.0]]
        assert requests[0].url.path == "/v1/embeddings"
        assert requests[0].headers["authorization"] == "Bearer sk-test"
        await e.aclose()

    async def test_reorders_by_index(self) -> None:
        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [9.0]},
                        {"index": 0, "embedding": [8.0]},
                    ]
                },
            )
        )
        e = OpenAICompatEmbedder("m", transport=transport)
        assert await e.embed(["first", "second"]) == [[8.0], [9.0]]
        await e.aclose()

    async def test_dimension_probe_is_cached(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(
                requests,
                [httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})],
            )
        )
        e = OpenAICompatEmbedder("m", transport=transport)
        assert await e.dimension() == 3
        assert await e.dimension() == 3  # cached: no second request
        assert len(requests) == 1
        await e.aclose()

    async def test_no_auth_header_without_key(self) -> None:
        requests: list[httpx.Request] = []
        transport = httpx.MockTransport(
            _handler(
                requests, [httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})]
            )
        )
        e = OpenAICompatEmbedder("m", transport=transport)
        await e.embed(["a"])
        assert "authorization" not in requests[0].headers
        await e.aclose()
