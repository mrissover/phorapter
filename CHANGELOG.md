# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: the Python API
may change on minor versions).

## [Unreleased]

### Changed
- Shortened the UUIDv5 namespace seed to `phoropter.impluvium`, which changes
  `PHOROPTER_NAMESPACE` and every derived slice point ID. Harmless now (nothing
  is indexed against the previous namespace); once corpora exist in the wild this
  kind of change would require a reindex.

## [0.1.0] - 2026-07-10

First public release. Phoropter slices documents at multiple sizes on a shared
origin-aligned grid, detects containment between retrieved slices exactly (by
construction, verified with SHA-256 markers), and right-sizes context under a
token budget by discarding contained duplicates and trading small slices up to
their enclosing parents. Ships as an embeddable library and a REST + MCP server.

### Added
- Project scaffold: packaging, lint/type/test tooling, CI, core-purity import contracts.
- Core library: validated multi-size grid, multi-view slicer, structural markers,
  deterministic slice IDs, token counting, and the correctness gates (prefix
  property + cross-implementation parity).
- Storage and embedding: vector-store SPI with in-memory and Qdrant adapters, an
  adapter conformance suite, and the embedder SPI (Ollama, OpenAI-compatible,
  and a deterministic fake).
- Right-sizing engine: containment forest, cross-size rank fusion, and the greedy
  upward selection strategy with a full substitution trace, plus the in-process
  `budget_fit` entry point.
- Server: an OpenAPI 3.1 contract of record with generated DTOs, a FastAPI REST
  surface and a FastMCP surface over a shared async service layer, structured
  logging, optional bearer auth, and the `serve` / `mcp` / `check` CLI.
- Operations: a Dockerfile (tiktoken vocabulary baked in), a dev docker-compose
  stack, and a PyPI trusted-publishing release workflow.
- Evaluation harness (`phoropter eval forest|budget|regress`): forest-density and
  budget-utilization metrics with deterministic regression gating.
