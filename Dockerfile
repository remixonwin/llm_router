FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY src/llm_router src/llm_router
COPY pyproject.toml pyproject.toml
COPY .env.example .env.example

RUN python -m ensurepip && python -m pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir poetry || true

# Install runtime deps via poetry if present otherwise skip
RUN if [ -f pyproject.toml ]; then \
      pip install --no-cache-dir . ; \
    fi

EXPOSE 8001

CMD ["python", "-m", "llm_router.server"]
