# Intelligent LLM Router

A quota-aware LLM routing system that distributes requests across 13+ providers, with automatic failover and local Ollama fallback.

## 🚀 Quick Start

1. **Install [uv](https://docs.astral.sh/uv/)** (if not already installed).
2. **Setup environment**:
   ```bash
   cp .env.example .env
   # Add your API keys (GROQ_API_KEY, GEMINI_API_KEY, etc.)
   ```
3. **Run the server**:
   ```bash
   uv run llm-router
   ```

## 🛠️ API Usage

The router provides an OpenAI-compatible API on port `7544`.

### Chat Completion
```bash
curl http://localhost:7544/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "routing": { "strategy": "cost_optimized" }
  }'
```

### Other Endpoints
- `POST /v1/embeddings`: Get vector embeddings.
- `POST /v1/vision`: Analyze images via URL or base64.
- `GET /v1/models`: List all discovered models.
- `GET /providers`: View real-time quota & latency stats.

## ✨ Features

- **Intelligent Routing**: Strategies for `auto`, `cost_optimized`, `quality_first`, and `latency_first`.
- **Automatic Discovery**: Dynamically fetches model capabilities from providers every hour.
- **Failover & Resilience**: Token-bucket rate limiting, circuit breakers, and automatic provider fallbacks.
- **Two-Tier Cache**: High-performance exact (DiskCache) and semantic (Cosine similarity) caching.
- **Local Fallback**: Uses Ollama as a last resort when all cloud quotas are exhausted.

## 📂 Project Structure

- `src/llm_router/`: Core logic (Router, Quota Manager, Discovery, Cache).
- `tests/`: Comprehensive test suite (`uv run pytest`).
- `pyproject.toml`: Modern dependency management via `uv`.

## 📦 Publishing

This repository ships Python source in `src/llm_router/`. CI builds artifacts and publishes from the main branch. Common publish flows:

- Build source + wheel locally:

```bash
python3 -m pip install --upgrade build twine
python3 -m build  # creates files in dist/
```

- Upload to PyPI (use CI secrets in production):

```bash
python3 -m twine upload dist/*
```

- Build and push Docker images to GitHub Container Registry (example):

```bash
docker build -t ghcr.io/<org>/<repo>:<tag> .
docker push ghcr.io/<org>/<repo>:<tag>
```

CI (GitHub Actions) will normally create the packages and publish them using repository secrets (PYPI_API_TOKEN, GITHUB_TOKEN). After pushing a branch the Actions workflow will run and produce status checks.

## ▶️ Monitor CI/CD

- Web: visit your repository's Actions tab to see workflow runs and logs.
- CLI (GitHub CLI):

```bash
gh run list --limit 5
gh run view <run-id> --log
```

If a workflow fails, open the failing job in Actions to inspect logs, then apply fixes locally and push a new commit. See Troubleshooting below for common issues.

## 🩹 Troubleshooting / Common Fixes

- Lint/test failures: run locally: `python3 -m pip install -e '.[dev]' && ruff --fix src tests && pytest -q`
- Missing secrets in CI: ensure repository has `PYPI_API_TOKEN` and any provider keys stored as secrets.
- Packaging errors: ensure `pyproject.toml` has a proper [project] section and `packages = ['src']` if using setuptools.
