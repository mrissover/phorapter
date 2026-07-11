"""Phoropter command-line entry point.

Subcommands:

- ``serve`` — run the REST API (with the MCP server mounted at ``/mcp``) under
  uvicorn;
- ``mcp`` — run the MCP server over stdio, for a local MCP client;
- ``check`` — validate startup (store reachable, embedder probed) and exit 0/1;
- ``eval`` — offline evaluation harness (``forest`` / ``budget`` / ``regress``)
  against a running server (requires the ``eval`` extra).

``--version`` reports the installed version and exits.
"""

from __future__ import annotations

import argparse
import asyncio

from phoropter import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phoropter", description="Phoropter multi-view RAG server"
    )
    parser.add_argument("--version", action="version", version=f"phoropter {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the REST + MCP server (uvicorn)")
    serve.add_argument("--host", default=None, help="override the configured host")
    serve.add_argument("--port", type=int, default=None, help="override the configured port")

    sub.add_parser("mcp", help="run the MCP server over stdio")
    sub.add_parser("check", help="validate startup and exit 0 (ok) or 1 (degraded)")
    sub.add_parser(
        "eval",
        help="offline evaluation harness (forest/budget/regress); see 'phoropter eval --help'",
        add_help=False,
    )
    return parser


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn

    from phoropter.config import Settings
    from phoropter.server.mcp import build_mcp
    from phoropter.server.rest import create_app
    from phoropter.service.core import ServiceCore

    settings = Settings()
    from phoropter.config import build_embedder, build_store

    core = ServiceCore(
        store=build_store(settings), embedder=build_embedder(settings), settings=settings
    )
    app = create_app(settings, core=core)
    # Mount the MCP server's streamable-HTTP app under /mcp on the same process.
    app.mount("/mcp", build_mcp(core).http_app())

    uvicorn.run(
        app,
        host=host or settings.server.host,
        port=port if port is not None else settings.server.port,
        log_config=None,
    )
    return 0


def _mcp() -> int:
    from phoropter.config import Settings, build_embedder, build_store
    from phoropter.server.mcp import build_mcp
    from phoropter.service.core import ServiceCore

    settings = Settings()
    core = ServiceCore(
        store=build_store(settings), embedder=build_embedder(settings), settings=settings
    )

    async def _prepare() -> None:
        await core.startup()

    asyncio.run(_prepare())
    build_mcp(core).run(transport="stdio")
    return 0


def _check() -> int:
    from phoropter.config import Settings, build_embedder, build_store
    from phoropter.service.core import ServiceCore

    settings = Settings()
    core = ServiceCore(
        store=build_store(settings), embedder=build_embedder(settings), settings=settings
    )

    async def _run() -> int:
        try:
            report = await core.startup()
        finally:
            await core.aclose()
        store = "ok" if report.store_ok else "UNREACHABLE"
        dim = report.embedder_dimension
        embedder = f"ok (dimension {dim})" if dim is not None else "UNAVAILABLE"
        print(f"store:    {store}")
        print(f"embedder: {embedder}")
        if report.detail:
            print(f"detail:   {report.detail}")
        return 0 if report.ok else 1

    return asyncio.run(_run())


def main(argv: list[str] | None = None) -> int:
    import sys

    # `eval` owns its own argument grammar; hand it the remaining argv untouched.
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "eval":
        from phoropter.eval import main as eval_main

        return eval_main(argv[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "mcp":
        return _mcp()
    if args.command == "check":
        return _check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
