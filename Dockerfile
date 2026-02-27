# =============================================================================
# LLM Router Service - Frontend + Backend Combined
# =============================================================================
# This Dockerfile builds a standalone LLM Router service that serves both:
# - React frontend on the root path (/) and (/llm/)
# - FastAPI backend API on (/v1/*), (/providers, /admin, etc.)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Build React frontend
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package files for layer caching
COPY frontend/package*.json ./
RUN npm ci --prefer-offline 2>/dev/null || npm install

# Copy frontend source and build (with container settings for correct base path)
COPY frontend/ ./
ENV CONTAINER_BUILD=true
ENV VITE_ROUTER_BASENAME=/
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Python dependencies builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS python-builder

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.4.29

WORKDIR /build

# Copy package config
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the package and dependencies
RUN uv pip install --no-cache -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Production image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy Python source code
COPY --chown=appuser:appuser src/ ./src/

# Copy built frontend
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist ./frontend/dist

# Create runtime directories
RUN mkdir -p /app/data /app/data/router_cache \
    && chown -R appuser:appuser /app

# Python path for imports
ENV PYTHONPATH="/app/src:/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default environment variables
ENV ROUTER_FRONTEND_DIR=/app/frontend/dist
ENV ROUTER_HOST=0.0.0.0
ENV ROUTER_PORT=8080
ENV ROUTER_LOG_LEVEL=INFO
ENV ROUTER_CACHE_DIR=/app/data/router_cache
ENV ROUTER_ENABLE_OLLAMA_FALLBACK=true

# Expose the service port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${ROUTER_PORT}/health || exit 1

# Run as non-root user
USER appuser

# Start the LLM Router server
CMD ["python", "-m", "llm_router.server", "--host", "0.0.0.0", "--port", "8080"]
