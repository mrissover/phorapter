# syntax=docker/dockerfile:1

# ── build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS build

WORKDIR /build
RUN pip install --no-cache-dir build hatchling

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ── runtime stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Bake the default tiktoken vocabulary into the image so budgeting works offline.
ENV PHOROPTER_TIKTOKEN_CACHE=/opt/phoropter/tiktoken \
    TIKTOKEN_CACHE_DIR=/opt/phoropter/tiktoken \
    PYTHONUNBUFFERED=1

RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --no-cache-dir "$(ls /dist/*.whl)[server,qdrant]"

RUN mkdir -p /opt/phoropter/tiktoken \
    && python -c "import tiktoken; tiktoken.get_encoding('o200k_base').encode('warm the cache')"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 phoropter
USER phoropter

EXPOSE 8000
ENV PHOROPTER_SERVER__HOST=0.0.0.0 \
    PHOROPTER_SERVER__PORT=8000

# The healthcheck runs the same startup validation the CLI exposes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-m", "phoropter", "check"]

ENTRYPOINT ["python", "-m", "phoropter"]
CMD ["serve"]
