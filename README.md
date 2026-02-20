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

Note: The CLI under `cli/` is experimental and currently has a backend
integration issue. We exclude the CLI from release packaging until the
backend CLI bugs are resolved. See `cli/README.md` for local development.

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
