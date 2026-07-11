"""The FastAPI application: thin routers over :class:`ServiceCore`, plus the error envelope.

Routers hold no business logic — each validates its request against a generated
DTO, calls the matching service method, and maps the result back through
:mod:`phoropter.server.mappers`. Every :class:`~phoropter.errors.PhoropterError`
is caught by one exception handler that turns its stable ``code`` into an HTTP
status and the uniform ``{"error": {...}}`` envelope; unexpected exceptions
become a 500 with an ``INTERNAL`` code.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from phoropter import __version__
from phoropter.config import build_embedder, build_store
from phoropter.errors import PhoropterError
from phoropter.server import mappers
from phoropter.server import schemas as sc
from phoropter.server.middleware import (
    BearerAuthMiddleware,
    RequestIdMiddleware,
    RequestLogMiddleware,
    configure_logging,
)
from phoropter.service.core import ServiceCore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from phoropter.config import Settings

__all__ = ["CODE_TO_STATUS", "create_app"]

# Stable error code → HTTP status. Core codes surface unchanged; the taxonomy is
# extended (not remapped) by the server layer (DOCUMENT_TOO_LARGE, UNAUTHORIZED).
CODE_TO_STATUS: dict[str, int] = {
    "VALIDATION_ERROR": 422,
    "INVALID_GRID": 422,
    "SLICING_ERROR": 422,
    "UNKNOWN_TOKENIZER": 422,
    "CORPUS_NOT_FOUND": 404,
    "DOCUMENT_NOT_FOUND": 404,
    "CORPUS_EXISTS": 409,
    "EMBEDDER_MISMATCH": 409,
    "DOCUMENT_TOO_LARGE": 413,
    "EMBEDDER_UNAVAILABLE": 503,
    "STORE_UNAVAILABLE": 503,
    "UNAUTHORIZED": 401,
    "INTERNAL": 500,
}

# Reverse map for framework-raised HTTP errors (routing, malformed body). A
# malformed body (400) is special-cased to VALIDATION_ERROR by the handler.
_STATUS_TO_CODE: dict[int, str] = {
    400: "VALIDATION_ERROR",
    401: "UNAUTHORIZED",
    404: "CORPUS_NOT_FOUND",
    405: "VALIDATION_ERROR",
    413: "DOCUMENT_TOO_LARGE",
    422: "VALIDATION_ERROR",
    500: "INTERNAL",
    503: "STORE_UNAVAILABLE",
}


def _envelope(
    *, code: str, message: str, status: int, request_id: str | None, details: object | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": request_id,
            }
        },
    )


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def create_app(
    settings: Settings,
    *,
    core: ServiceCore | None = None,
    run_startup: bool = True,
) -> FastAPI:
    """Build the FastAPI app.

    A prebuilt ``core`` may be injected (tests do this with an in-memory store);
    otherwise a store and embedder are constructed from ``settings``. When
    ``run_startup`` is true, the store is pinged and bootstrapped and the
    embedder dimension is probed in the app's lifespan.
    """
    configure_logging(level=settings.logging.level, json=settings.logging.json_output)

    service = core or ServiceCore(
        store=build_store(settings),
        embedder=build_embedder(settings),
        settings=settings,
    )

    def lifespan(_: FastAPI) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def _ctx() -> AsyncIterator[None]:
            if run_startup:
                await service.startup()
            try:
                yield
            finally:
                await service.aclose()

        return _ctx()

    app = FastAPI(
        title="Phoropter",
        version="1.0",
        description="Multi-view retrieval with token-budgeted context right-sizing.",
        lifespan=lifespan,
    )
    app.state.service = service

    # add_middleware prepends, so the last one added is outermost and runs first.
    # Order the request reaches them: request-id (outermost) -> auth -> log
    # (innermost) -> route, so the id is set before auth and logging read it.
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(BearerAuthMiddleware, api_key=settings.server.api_key)
    app.add_middleware(RequestIdMiddleware)

    _register_handlers(app)
    _register_routes(app, service)
    return app


def _register_handlers(app: FastAPI) -> None:
    @app.exception_handler(PhoropterError)
    async def _phoropter_error(request: Request, exc: PhoropterError) -> JSONResponse:
        status = CODE_TO_STATUS.get(exc.code, 500)
        return _envelope(
            code=exc.code, message=str(exc), status=status, request_id=_request_id(request)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(
            code="VALIDATION_ERROR",
            message="request validation failed",
            status=422,
            request_id=_request_id(request),
            details={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Reshape framework HTTP errors (malformed body, 404/405 routing) into the
        # uniform envelope. A malformed body (Starlette's 400) is normalized to a
        # 422 validation error so it stays within the documented status set;
        # everything else keeps its status's code.
        status = 422 if exc.status_code == 400 else exc.status_code
        code = _STATUS_TO_CODE.get(status, "INTERNAL")
        return _envelope(
            code=code,
            message=str(exc.detail),
            status=status,
            request_id=_request_id(request),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        return _envelope(
            code="INTERNAL",
            message="an unexpected error occurred",
            status=500,
            request_id=_request_id(request),
        )


def _errors(*statuses: int) -> dict[int | str, dict[str, object]]:
    """Declare the error envelope for the given statuses so the runtime spec matches the contract."""
    return {status: {"model": sc.ErrorEnvelope} for status in statuses}


def _register_routes(app: FastAPI, service: ServiceCore) -> None:
    # ── health ──────────────────────────────────────────────────────────────

    @app.get("/healthz", response_model=sc.Liveness, tags=["health"])
    async def liveness() -> sc.Liveness:
        return sc.Liveness(status=sc.Status.ok)

    @app.get(
        "/v1/health",
        response_model=sc.Readiness,
        responses=_errors(503),
        tags=["health"],
    )
    async def readiness() -> sc.Readiness:
        report = await service.startup()
        status = sc.Status1.ok if report.ok else sc.Status1.degraded
        return sc.Readiness(
            status=status,
            store=report.store_ok,
            embedder=report.embedder_dimension is not None,
            detail=report.detail,
        )

    @app.get("/v1/info", response_model=sc.Info, tags=["health"])
    async def info() -> sc.Info:
        from phoropter.embed import default_registry
        from phoropter.stores import available_store_names

        s = service.settings
        return sc.Info(
            name="phoropter",
            version=__version__,
            store=s.store.kind,
            embedder=f"{s.embedder.provider}:{s.embedder.model}",
            stores_available=list(available_store_names()),
            embedders_available=list(default_registry.providers()),
        )

    # ── corpora ─────────────────────────────────────────────────────────────

    @app.get(
        "/v1/corpora",
        response_model=sc.CorpusList,
        responses=_errors(503),
        tags=["corpora"],
    )
    async def list_corpora() -> sc.CorpusList:
        return sc.CorpusList(corpora=list(await service.corpora.list_names()))

    @app.post(
        "/v1/corpora",
        response_model=sc.Corpus,
        status_code=201,
        responses=_errors(409, 422, 503),
        tags=["corpora"],
    )
    async def create_corpus(body: sc.CreateCorpusRequest) -> sc.Corpus:
        await service.corpora.create(
            body.name,
            grid_sizes=tuple(body.grid_sizes) if body.grid_sizes is not None else None,
            tokenizer=body.tokenizer,
            embedder_provider=body.embedder_provider,
            embedder_model=body.embedder_model,
        )
        return mappers.corpus_info_to_dto(await service.corpora.inspect(body.name))

    @app.get(
        "/v1/corpora/{corpus}",
        response_model=sc.Corpus,
        responses=_errors(404),
        tags=["corpora"],
    )
    async def get_corpus(corpus: str) -> sc.Corpus:
        return mappers.corpus_info_to_dto(await service.corpora.inspect(corpus))

    @app.delete(
        "/v1/corpora/{corpus}",
        status_code=204,
        responses=_errors(404),
        tags=["corpora"],
    )
    async def drop_corpus(corpus: str) -> Response:
        await service.corpora.drop(corpus)
        return Response(status_code=204)

    # ── documents ───────────────────────────────────────────────────────────

    @app.get(
        "/v1/corpora/{corpus}/documents",
        response_model=sc.DocumentPage,
        responses=_errors(404),
        tags=["documents"],
    )
    async def list_documents(
        corpus: str,
        limit: int = Query(100, ge=1, le=1000),
        cursor: str | None = Query(None),
    ) -> sc.DocumentPage:
        page = await service.documents.list_page(corpus, limit=limit, cursor=cursor)
        return mappers.document_page_to_dto(page)

    @app.put(
        "/v1/corpora/{corpus}/documents/{document_id}",
        response_model=sc.DocumentRecord,
        responses=_errors(404, 413, 422),
        tags=["documents"],
    )
    async def put_document(
        corpus: str, document_id: str, body: sc.PutDocumentRequest
    ) -> sc.DocumentRecord:
        record = await service.documents.put(
            corpus, document_id, body.text, metadata=mappers.metadata_from_dto(body.metadata)
        )
        return mappers.document_record_to_dto(record)

    @app.get(
        "/v1/corpora/{corpus}/documents/{document_id}",
        response_model=sc.DocumentRecord,
        responses=_errors(404),
        tags=["documents"],
    )
    async def get_document(corpus: str, document_id: str) -> sc.DocumentRecord:
        return mappers.document_record_to_dto(await service.documents.get(corpus, document_id))

    @app.delete(
        "/v1/corpora/{corpus}/documents/{document_id}",
        status_code=204,
        responses=_errors(404),
        tags=["documents"],
    )
    async def delete_document(corpus: str, document_id: str) -> Response:
        await service.documents.delete(corpus, document_id)
        return Response(status_code=204)

    @app.post(
        "/v1/corpora/{corpus}/documents/batch",
        response_model=sc.BatchDocumentsResponse,
        responses=_errors(404, 413, 422),
        tags=["documents"],
    )
    async def batch_documents(
        corpus: str, body: sc.BatchDocumentsRequest
    ) -> sc.BatchDocumentsResponse:
        limit = service.settings.limits.max_batch_documents
        if len(body.documents) > limit:
            from phoropter.service.documents import DocumentTooLargeError

            raise DocumentTooLargeError(
                f"batch has {len(body.documents)} documents, exceeding the limit of {limit}"
            )
        # A missing corpus should fail the whole batch, not each item.
        await service.corpora.inspect(corpus)
        results: list[sc.BatchDocumentStatus] = []
        for item in body.documents:
            try:
                record = await service.documents.put(
                    corpus,
                    item.document_id,
                    item.text,
                    metadata=mappers.metadata_from_dto(item.metadata),
                )
                results.append(
                    sc.BatchDocumentStatus(
                        document_id=item.document_id,
                        ok=True,
                        record=mappers.document_record_to_dto(record),
                    )
                )
            except PhoropterError as e:
                results.append(
                    sc.BatchDocumentStatus(
                        document_id=item.document_id,
                        ok=False,
                        error=sc.ErrorDetail(code=e.code, message=str(e)),
                    )
                )
        return sc.BatchDocumentsResponse(results=results)

    # ── query ───────────────────────────────────────────────────────────────

    @app.post(
        "/v1/corpora/{corpus}/query",
        response_model=sc.QueryResponse,
        responses=_errors(404, 422, 503),
        tags=["query"],
    )
    async def query(corpus: str, body: sc.QueryRequest) -> sc.QueryResponse:
        # These carry schema defaults but are typed Optional in the generated DTO;
        # narrow to the same defaults so the service receives concrete values.
        expansion = str(body.expansion) if body.expansion is not None else "fill"
        outcome = await service.query.run(
            corpus,
            body.query,
            token_budget=body.token_budget,
            top_k_per_size=body.top_k_per_size if body.top_k_per_size is not None else 10,
            strategy=body.strategy if body.strategy is not None else "greedy_upward",
            expansion=expansion,
            sizes=tuple(body.sizes) if body.sizes is not None else None,
            tokenizer=body.tokenizer,
            max_slice_size=body.max_slice_size,
            include_text=body.include_text if body.include_text is not None else True,
            include_trace=body.include_trace if body.include_trace is not None else True,
        )
        return mappers.query_outcome_to_dto(outcome)
