"""The spec-equivalence gate plus a bounded Schemathesis contract suite.

Two independent checks keep the authored contract (``api/openapi.yaml``) and the
running FastAPI app honest:

1. **Equivalence** — every path + method in the authored contract exists in the
   runtime-produced spec, and their request/response bodies reference the same
   component schema names; and every authored component schema is present at
   runtime. This fails the build the moment a router drifts from the contract.
2. **Property-based conformance** — Schemathesis drives a bounded set of
   generated requests against the in-memory app and asserts every response
   conforms to the declared schema (including the uniform error envelope).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import schemathesis
import yaml
from hypothesis import HealthCheck, settings

from phoropter.config import (
    DefaultsSettings,
    EmbedderSettings,
    Settings,
    StoreSettings,
)
from phoropter.embed import FakeEmbedder
from phoropter.server.rest import create_app
from phoropter.service.core import ServiceCore
from phoropter.stores.memory import InMemoryStore

_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "api" / "openapi.yaml"


def _build_app() -> Any:
    s = Settings(
        store=StoreSettings(kind="memory"),
        embedder=EmbedderSettings(provider="fake", model="deterministic-32"),
        defaults=DefaultsSettings(grid_sizes=(64, 128, 256)),
    )
    core = ServiceCore(store=InMemoryStore(), embedder=FakeEmbedder(32), settings=s)
    return create_app(s, core=core, run_startup=False)


def _authored_spec() -> dict[str, Any]:
    return yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))


def _runtime_spec() -> dict[str, Any]:
    return _build_app().openapi()


def _ref_name(node: Any) -> str | None:
    """The trailing component name of a ``$ref``, chasing a single array wrapper."""
    if not isinstance(node, dict):
        return None
    ref = node.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    if node.get("type") == "array":
        return _ref_name(node.get("items", {}))
    return None


def _body_schema_name(operation: dict[str, Any], key: str) -> str | None:
    body = operation.get(key, {})
    content = body.get("content", {})
    json_body = content.get("application/json", {})
    return _ref_name(json_body.get("schema", {}))


def _response_schema_name(operation: dict[str, Any], status: str) -> str | None:
    responses = operation.get("responses", {})
    resp = responses.get(status)
    if resp is None:
        return None
    content = resp.get("content", {})
    json_body = content.get("application/json", {})
    return _ref_name(json_body.get("schema", {}))


# ── equivalence gate ─────────────────────────────────────────────────────────


def test_all_authored_paths_exist_at_runtime() -> None:
    authored = _authored_spec()
    runtime = _runtime_spec()
    for path, methods in authored["paths"].items():
        assert path in runtime["paths"], f"path {path} missing from runtime spec"
        for method in methods:
            if method in {"parameters"}:
                continue
            assert method in runtime["paths"][path], (
                f"{method.upper()} {path} missing from runtime spec"
            )


def test_request_and_response_schema_names_match() -> None:
    authored = _authored_spec()
    runtime = _runtime_spec()
    _http_methods = {"get", "post", "put", "delete", "patch"}
    for path, methods in authored["paths"].items():
        for method, op in methods.items():
            if method not in _http_methods:
                continue
            runtime_op = runtime["paths"][path][method]

            want_req = _body_schema_name(op, "requestBody")
            if want_req is not None:
                got_req = _body_schema_name(runtime_op, "requestBody")
                assert got_req == want_req, (
                    f"{method.upper()} {path} request schema: "
                    f"contract {want_req!r} != runtime {got_req!r}"
                )

            # Compare the success response schema (first 2xx the contract declares).
            success = next((s for s in op.get("responses", {}) if s.startswith("2")), None)
            if success is None:
                continue
            want_resp = _response_schema_name(op, success)
            if want_resp is None:
                continue
            got_resp = _response_schema_name(runtime_op, success)
            assert got_resp == want_resp, (
                f"{method.upper()} {path} {success} response schema: "
                f"contract {want_resp!r} != runtime {got_resp!r}"
            )


def test_authored_component_schemas_present_at_runtime() -> None:
    authored = _authored_spec()["components"]["schemas"]
    runtime = _runtime_spec()["components"]["schemas"]
    missing = sorted(set(authored) - set(runtime))
    assert not missing, f"authored component schemas absent from runtime spec: {missing}"


def test_error_responses_use_the_envelope() -> None:
    authored = _authored_spec()
    for path, methods in authored["paths"].items():
        for method, op in methods.items():
            if not isinstance(op, dict) or "responses" not in op:
                continue
            for status, resp in op["responses"].items():
                if not status.startswith(("4", "5")):
                    continue
                name = _ref_name(
                    resp.get("content", {}).get("application/json", {}).get("schema", {})
                )
                assert name == "ErrorEnvelope", (
                    f"{method.upper()} {path} {status} must use ErrorEnvelope, got {name!r}"
                )


# ── Schemathesis property-based conformance ──────────────────────────────────

schema = schemathesis.openapi.from_asgi("/openapi.json", _build_app())


from schemathesis.specs.openapi.checks import (  # noqa: E402
    response_schema_conformance,
    status_code_conformance,
)

# The two checks that assert contract equivalence: responses conform to their
# declared schema, and status codes are documented. The 405/resource-lifecycle
# heuristics are out of scope for this bounded suite.
_CONFORMANCE_CHECKS = [response_schema_conformance, status_code_conformance]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@schema.parametrize()
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
def test_schemathesis_conformance(case: Any) -> None:
    """Every generated request produces a response that conforms to its declared schema."""
    case.call_and_validate(checks=_CONFORMANCE_CHECKS)
