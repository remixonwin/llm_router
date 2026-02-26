FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --user -e .

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/src /app/src

ENV PYTHONPATH=/app/src:/root/.local/lib/python3.11/site-packages \
    PYTHONUNBUFFERED=1 \
    ROUTER_HOST=0.0.0.0 \
    ROUTER_PORT=7440 \
    ROUTER_CACHE_DIR=/tmp/llm_router_cache \
    ROUTER_REQUIRE_API_KEY=false \
    ROUTER_LOG_LEVEL=INFO

EXPOSE 7440

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7440/health || exit 1

ENTRYPOINT ["python", "-m", "llm_router.server"]
