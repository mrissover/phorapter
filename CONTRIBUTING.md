# Contributing to Phorapter

Thanks for your interest! This document covers the development workflow, the test
tiers (including the gates that block everything else), and the documentation policy.

## Development setup

```bash
git clone <repo-url> && cd phorapter
python -m venv .venv
# Windows: .venv\Scripts\activate    POSIX: source .venv/bin/activate
pip install -e . --group dev
pre-commit install
```

Python 3.11+ is required. The test suite runs on Linux and Windows; both are CI lanes.

## Test tiers

| Tier | Marker | Runs | Purpose |
|---|---|---|---|
| **Gates** | `gate` | first, `--exitfirst`; CI blocks everything on them | Correctness invariants the whole system leans on: the prefix property of the slicing grid, and byte-parity of markers with the reference implementation fixtures |
| Unit | (none) | always | Everything else in-process, including the in-memory store conformance suite |
| Integration | `integration` | opt-in (`pytest -m integration`); needs Docker | Qdrant round-trips, server end-to-end |

```bash
pytest -m gate --exitfirst   # the gates
pytest                       # unit tier (integration excluded by default)
pytest -m integration        # integration tier (requires Docker)
```

**Gating discipline:** if a gate test fails, do not merge anything downstream of the
slicer/markers — fix the gate first. The gates encode invariants that make containment
detection *exact*; a red gate means the exactness claim is broken, and every downstream
behavior is suspect.

## Static checks

```bash
ruff check . && ruff format --check .
mypy
lint-imports   # core-purity contract: core modules import stdlib only
```

The import-linter contract enforces that the core (`grid`, `slicer`, `markers`, `ids`,
`model`, `forest`, `fusion`, `selection`, `trace`, `tokens`, `stores.memory`) never
imports the server stack or third-party frameworks. `phorapter.tokens` imports tiktoken
*lazily inside functions* — keep it that way.

## Generated code

`src/phorapter/server/schemas.py` is **generated** from `api/openapi.yaml` (the REST
contract of record) via `hatch run codegen`. Never edit it by hand; change the contract
and regenerate. Hand-written request/response mapping lives in `server/mappers.py`.
CI fails if the running app's OpenAPI output diverges from the authored contract.

## Documentation policy

- Documentation lives in `docs/` and is updated in the same PR as the change it describes.
- Argue properties **intrinsically**: containment is "exact by construction of the
  origin-aligned grid"; upward substitution is "information-preserving because the child's
  bytes are a literal sub-span of the parent". Do not cite unpublished manuscripts,
  submission venues, or unpublished empirical results.
- A shipped-content guard (`scripts/check_shipped_content.py`, run by pre-commit and CI)
  blocks a small set of disallowed strings from entering the tree.

## Commit / PR conventions

- Keep the contract-of-record files (`api/openapi.yaml`, `docs/decisions.md`) in their
  own commits when they change — they are reviewed as contracts, not as code.
- Pre-1.0, the Python API may change on minor versions; the REST `/v1` contract is
  append-only once published.
