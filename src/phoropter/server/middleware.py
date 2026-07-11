"""Request-id, optional static bearer auth, and structured request logging.

Three small ASGI middlewares, added by the app factory:

- :class:`RequestIdMiddleware` stamps every request with a ``request_id`` (from
  the ``X-Request-Id`` header if the client supplied one, else a fresh UUID),
  exposes it on ``request.state`` for the error envelope, and echoes it back.
- :class:`BearerAuthMiddleware` enforces a single static bearer token when one is
  configured; it is a no-op when no key is set. Liveness and readiness probes are
  always exempt. Real authn/z is reverse-proxy territory (see docs/api.md).
- :class:`RequestLogMiddleware` emits one structured log line per request.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

__all__ = [
    "REQUEST_ID_HEADER",
    "BearerAuthMiddleware",
    "RequestIdMiddleware",
    "RequestLogMiddleware",
    "configure_logging",
    "unauthorized_response",
]

REQUEST_ID_HEADER = "X-Request-Id"
_PUBLIC_PATHS = frozenset({"/healthz", "/v1/health"})

_log = structlog.get_logger("phoropter.request")


def configure_logging(*, level: str = "INFO", json: bool = True) -> None:
    """Configure structlog once for the process."""
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            _level_to_int(level),
        ),
        cache_logger_on_first_use=True,
    )


def _level_to_int(level: str) -> int:
    import logging

    return getattr(logging, level.upper(), logging.INFO)


def unauthorized_response(request_id: str | None) -> JSONResponse:
    """The uniform 401 envelope, shared by the auth middleware and handlers."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "code": "UNAUTHORIZED",
                "message": "a valid bearer token is required",
                "details": None,
                "request_id": request_id,
            }
        },
    )


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign and echo a request id; expose it on ``request.state.request_id``."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Enforce a static bearer token when one is configured; a no-op otherwise."""

    def __init__(self, app: object, *, api_key: str | None) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._api_key = api_key

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self._api_key is None or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        header = request.headers.get("Authorization", "")
        expected = f"Bearer {self._api_key}"
        if header != expected:
            return unauthorized_response(getattr(request.state, "request_id", None))
        return await call_next(request)


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured log line per request with method, path, status, and duration."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        _log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
